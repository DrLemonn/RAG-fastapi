import os
import pickle
import uuid
import shutil
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from pprint import pprint

# 假设 cleaner 是你之前定义的模块
# import cleaner 

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_classic.storage import LocalFileStore, EncoderBackedStore # 注意：langchain_classic 可能已弃用，建议用 langchain.storage
from langchain_classic.retrievers import MultiVectorRetriever # 同上

from pinecone import Pinecone, ServerlessSpec

# Load all environment variables
load_dotenv()

# --- 配置项 ---
INDEX_NAME = "fastapi-cleaned"
LOCAL_STORE_PATH = "./parent_docs_store"

# 初始化 Pinecone
pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])

def load_docs(path="./docs"):
    loader = DirectoryLoader(path, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    return loader.load()

def split_markdown_by_h2(parent_doc: Document) -> List[Document]:
    headers_to_split_on = [("##", "section")]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    child_docs = splitter.split_text(parent_doc.page_content)
    for child in child_docs:
        original_meta = child.metadata.copy()
        child.metadata.update(parent_doc.metadata) 
        child.metadata.update(original_meta)
    return child_docs

def get_retriever(docs=None, index_exists=False):
    embeddings = OpenAIEmbeddings(
        model="qwen/qwen3-embedding-8b", 
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    fs = LocalFileStore(LOCAL_STORE_PATH)
    store = EncoderBackedStore(
        store=fs,
        key_encoder=lambda x: x,
        value_serializer=pickle.dumps,
        value_deserializer=pickle.loads
    )

    if index_exists:
        print(f"检测到索引 '{INDEX_NAME}' 已存在，直接加载...")
        vectorstore = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)
    else:
        print(f"索引 '{INDEX_NAME}' 不存在，正在创建...")
        pc.create_index(
            name=INDEX_NAME, 
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region='us-east-1'),
            dimension=4096 
        )
        vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key="doc_id"
    )

    if not index_exists and docs:
        print("正在处理文档并上传...")
        batch_parent_ids = []
        batch_parent_docs = []
        batch_child_docs = []

        for parent_doc in docs:
            parent_id = str(uuid.uuid4())
            child_docs = split_markdown_by_h2(parent_doc)
            for child in child_docs:
                child.metadata["doc_id"] = parent_id
            batch_parent_ids.append(parent_id)
            batch_parent_docs.append(parent_doc)
            batch_child_docs.extend(child_docs)

        retriever.docstore.mset(list(zip(batch_parent_ids, batch_parent_docs)))
        retriever.vectorstore.add_documents(batch_child_docs)
        print(f"成功处理并存储 {len(docs)} 个父文档。")
    
    return retriever

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ==========================================
# 🔥 新增功能：调试向量检索结果
# ==========================================
def debug_retrieval(retriever, query, k=4):
    """
    绕过 MultiVectorRetriever 的合并逻辑，直接查看 VectorStore 返回了什么。
    """
    print(f"\n🔍 [Debug] 正在分析 Query: '{query}'")
    print("=" * 50)
    
    # 直接调用底层的 vectorstore 进行带分数的搜索
    # 注意：Pinecone 的 score 如果是 cosine，通常是 0-1 之间（越接近 1 越相似）
    results = retriever.vectorstore.similarity_search_with_score(query, k=k)
    
    seen_parent_ids = set()
    
    for i, (doc, score) in enumerate(results):
        doc_id = doc.metadata.get("doc_id", "Unknown")
        is_duplicate_parent = doc_id in seen_parent_ids
        seen_parent_ids.add(doc_id)
        
        status = "✅ (将被采用)" if not is_duplicate_parent else "🔻 (将被去重)"
        
        print(f"Chunk #{i+1} | Score: {score:.4f} | Parent ID: {doc_id}")
        print(f"状态: {status}")
        # 先在外部处理好字符串
        preview_text = doc.page_content[:100].replace('\n', ' ')
        print(f"内容片段: {preview_text}...")
        print("-" * 50)
        
    print(f"📊 最终去重后，RAG 将获得 {len(seen_parent_ids)} 个完整的父文档作为上下文。")
    print("=" * 50 + "\n")

def main():
    # 1. 初始化环境
    index_exists = pc.has_index(INDEX_NAME)
    docs = None
    if not index_exists:
        print(">>> 初始化模式 <<<")
        docs = load_docs("./docs")
        # for doc in docs: doc.page_content = cleaner.clean_fastapi_markdown(doc.page_content) 
    
    # 2. 获取 Retriever
    retriever = get_retriever(docs=docs, index_exists=index_exists)

    # 3. 定义 Query
    query = "FastAPI 的核心特性有哪些？"
    print(f"提问: {query}")

    # 4. 🔥 执行调试：查看取出了哪些 Chunk 以及分数
    debug_retrieval(retriever, query, k=5)

    # # 5. 执行正常的 RAG 流程
    # print(">>> 开始执行 RAG 生成...")
    # prompt_temp = """你是一个助手。请根据上下文回答问题。
    # 问题： {question} 
    # 上下文： {context} 
    # 答案："""
    # prompt = PromptTemplate.from_template(prompt_temp)
    # llm = ChatOpenAI(
    #     model="deepseek/deepseek-r1-0528:free", 
    #     openai_api_key=os.getenv("OPENAI_API_KEY"),
    #     base_url="https://openrouter.ai/api/v1", 
    #     temperature=0.6 
    # )

    # rag_chain = (
    #     {"context": retriever | format_docs, "question": RunnablePassthrough()}
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )
    
    # result = rag_chain.invoke(query)
    # print("\n🤖 AI 回答:")
    # pprint(result)

if __name__ == "__main__":
    main()