import os
import time
import jieba
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from openai import OpenAI

# ================= 配置区域 (请填入你的 Key) =================
PINECONE_API_KEY = "sk.."  # 你的 Pinecone Key
OPENAI_API_KEY = "sk.."    # 你的 OpenAI Key
INDEX_NAME = "test-hybrid-sanity-check"
REGION = "us-east-1"         # Pinecone Serverless 的地区 (通常是 us-east-1)
# ==========================================================

# 初始化客户端
pc = Pinecone(api_key=PINECONE_API_KEY)
client = OpenAI(api_key=OPENAI_API_KEY,
                base_url="https://openrouter.ai/api/v1"
                )

# 1. 准备中文分词工具 (适配 BM25)
def get_tokens(text):
    # BM25 默认按空格分词，所以我们需要把中文切开并用空格连接
    # 例如："FastAPI很快" -> "FastAPI 很 快"
    return " ".join(jieba.cut(text))

# 2. 准备测试数据 (模拟 FastAPI 文档片段)
test_data = [
    {"id": "doc1", "text": "FastAPI 是一个现代、快速（高性能）的 Web 框架，基于 Python 3.6+ 类型提示。"},
    {"id": "doc2", "text": "Pydantic 是一个数据验证库，FastAPI 使用它来定义 Schema。"},
    {"id": "doc3", "text": "Uvicorn 是一个基于 uvloop 的 ASGI 服务器，用于运行 FastAPI 应用。"},
    {"id": "doc4", "text": "Starlette 负责处理路由和 WebSocket，FastAPI 继承了它的功能。"},
]

# 3. 初始化并训练 BM25 (Sparse Encoder)
print(">>> 正在训练 BM25...")
bm25 = BM25Encoder()
# 提取所有文本进行简单的训练 (fit)，计算 IDF
corpus = [get_tokens(d["text"]) for d in test_data]
bm25.fit(corpus)
print(">>> BM25 训练完成")

# 4. 创建 Pinecone 索引 (如果不存在)
# 关键点：metric 必须是 "dotproduct" 才能支持混合检索
if not pc.has_index(INDEX_NAME):
    print(f">>> 创建索引 {INDEX_NAME}...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=3072, # text-embedding-3-large 的维度
        metric="dotproduct", # 混合检索的核心！
        spec=ServerlessSpec(cloud="aws", region=REGION)
    )
    # 等待索引初始化
    while not pc.describe_index(INDEX_NAME).status['ready']:
        time.sleep(1)
else:
    print(f">>> 索引 {INDEX_NAME} 已存在，跳过创建。")

index = pc.Index(INDEX_NAME)

# 5. 生成向量并入库 (Upsert)
print(">>> 正在生成向量并入库...")
vectors_to_upsert = []

for item in test_data:
    # A. Dense Vector (OpenAI)
    response = client.embeddings.create(input=item["text"], model="openai/text-embedding-3-large", encoding_format="float")
    dense_vec = response.data[0].embedding
    
    # B. Sparse Vector (BM25)
    # 注意：这里传入分好词的字符串
    sparse_vec = bm25.encode_documents(get_tokens(item["text"]))
    
    vectors_to_upsert.append({
        "id": item["id"],
        "values": dense_vec,
        "sparse_values": sparse_vec,
        "metadata": {"text": item["text"]}
    })

index.upsert(vectors=vectors_to_upsert)
print(">>> 入库完成，等待数据一致性 (休息2秒)...")
time.sleep(2)

# 6. 混合权重计算函数 (官方推荐算法)
def hybrid_scale(dense, sparse, alpha: float):
    """
    Alpha 控制权重:
    1.0 = 纯语义 (OpenAI)
    0.0 = 纯关键词 (BM25)
    0.5 = 混合
    """
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    
    # 稀疏向量乘 (1 - alpha)
    h_sparse = {
        'indices': sparse['indices'],
        'values':  [v * (1 - alpha) for v in sparse['values']]
    }
    # 稠密向量乘 alpha
    h_dense = [v * alpha for v in dense]
    return h_dense, h_sparse

# 7. 执行混合查询 (Hybrid Search)
query_text = "FastAPI 验证库" 
print(f"\n>>> 测试查询: '{query_text}'")

# A. 生成 Query Dense Vector
q_dense_res = client.embeddings.create(input=query_text, model="text-embedding-3-large")
q_dense = q_dense_res.data[0].embedding

# B. 生成 Query Sparse Vector
q_sparse = bm25.encode_queries(get_tokens(query_text))

# C. 调整权重 (Alpha = 0.7, 偏向语义但保留关键词)
hdense, hsparse = hybrid_scale(q_dense, q_sparse, alpha=0.7)

# D. 发送查询
results = index.query(
    vector=hdense,
    sparse_vector=hsparse,
    top_k=2,
    include_metadata=True
)

# 8. 打印结果
print("\n=== 检索结果 ===")
for match in results['matches']:
    print(f"ID: {match['id']}")
    print(f"Score: {match['score']:.4f}")
    print(f"Content: {match['metadata']['text']}\n")

# 清理测试索引 (可选，不想花钱就取消注释)
# pc.delete_index(INDEX_NAME)
# print(">>> 测试索引已删除")