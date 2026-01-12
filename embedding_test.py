import os
import numpy as np
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# --- 1. 模拟你的数据 ---
# 请把 docs/index.md 里的真实内容（大约前500字）粘贴到这里
# 务必保证这和你文件里的内容一模一样
raw_content = """# FastAPI

FastAPI 是一个用于构建 API 的现代、快速（高性能）的 web 框架，使用 Python 并基于标准的 Python 类型提示。

关键特性:

* 快速：可与 NodeJS 和 Go 并肩的极高性能（归功于 Starlette 和 Pydantic）。最快的 Python web 框架之一。
* 高效编码：提高功能开发速度约 200％ 至 300％。
* 更少 bug：减少约 40％ 的人为（开发者）导致错误。
* 智能：极佳的编辑器支持。处处皆可自动补全，减少调试时间。
* 简单：设计的易于使用和学习，阅读文档的时间更短。
* 简短：使代码重复最小化。通过不同的参数声明实现丰富功能。bug 更少。
* 健壮：生产可用级别的代码。还有自动生成的交互式文档。
* 标准化：基于（并完全兼容）API 的相关开放标准：OpenAPI (以前被称为 Swagger) 和 JSON Schema。
"""

# --- 2. 模拟你的切分逻辑 ---
def test_split(text):
    # 你的逻辑是按 ## 切分
    headers_to_split_on = [("##", "section")]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    # 模拟 Parent Document
    doc = Document(page_content=text, metadata={"source": "docs/index.md"})
    return splitter.split_text(doc.page_content)

# --- 3. 计算相似度 (Cosine Similarity) ---
def cosine_similarity(vec_a, vec_b):
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(vec_a, vec_b) / (norm_a * norm_b)

def main():
    print(">>> 正在初始化 Embedding 模型...")
    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-small", 
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    raw_query = "FastAPI 的核心特性有哪些？"
    print(f"\n>>> Query: {raw_query}")

    # 1. 执行切分
    chunks = test_split(raw_content)
    print(f">>> 切分结果: 生成了 {len(chunks)} 个 Chunk")


    
    # ❌ 方式 1: 直接 embedding (你现在的做法)
    print(f"\n>>> [测试 1] 原始 Query: {raw_query}")
    query_vec_1 = embeddings.embed_query(raw_query)
    chunk_vec = embeddings.embed_query(chunks[0].page_content) # 假设只有一个 chunk
    score_1 = cosine_similarity(query_vec_1, chunk_vec)
    print(f"🔥 得分: {score_1:.4f}")

    # ✅ 方式 2: 加上指令前缀 (Instruction)
    # 不同的模型前缀不同，对于 Qwen/Alibaba 系列，通常试用以下几种：
    
    # 前缀 A (通用检索)
    prefix_a = "Represent this query for retrieving relevant documents: "
    query_a = prefix_a + raw_query
    
    # 前缀 B (中文语境)
    prefix_b = "为这个句子生成表示以用于检索相关文章："
    query_b = prefix_b + raw_query

    print(f"\n>>> [测试 2] 加英文前缀 Query: {query_a}")
    query_vec_a = embeddings.embed_query(query_a)
    score_a = cosine_similarity(query_vec_a, chunk_vec)
    print(f"🔥 得分: {score_a:.4f}")

    print(f"\n>>> [测试 3] 加中文前缀 Query: {query_b}")
    query_vec_b = embeddings.embed_query(query_b)
    score_b = cosine_similarity(query_vec_b, chunk_vec)
    print(f"🔥 得分: {score_b:.4f}")

if __name__ == "__main__":
    main()