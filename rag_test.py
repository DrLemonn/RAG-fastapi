import os
import pickle
import uuid
import shutil
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from pprint import pprint

# 假设 cleaner 是你之前定义的模块
import cleaner 

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_classic.storage import LocalFileStore, EncoderBackedStore
from langchain_classic.retrievers import MultiVectorRetriever

from pinecone import Pinecone, ServerlessSpec

# Load all environment variables from .env file
load_dotenv()

# --- 配置项 ---
INDEX_NAME = "fastapi-cleaned"
LOCAL_STORE_PATH = "./parent_docs_store" # 本地存储父文档的文件夹路径

# 初始化 Pinecone
pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])


def load_docs(path="./docs"):
    """加载原始 Markdown 文件"""
    loader = DirectoryLoader(
        path, 
        glob="**/*.md", 
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    return loader.load()

def split_markdown_by_h2(parent_doc: Document) -> List[Document]:
    """
    切分逻辑：按照 ## Header 2 切分，第一个 chunk 包含 # Header 1
    """
    headers_to_split_on = [("##", "section")]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    child_docs = splitter.split_text(parent_doc.page_content)
    
    for child in child_docs:
        original_meta = child.metadata.copy()
        child.metadata.update(parent_doc.metadata) 
        child.metadata.update(original_meta)
        
    return child_docs

def get_retriever(docs=None, index_exists=False):
    """
    构建或加载 Retriever
    :param docs: 文档列表 (仅在初始化时需要)
    :param index_exists: 索引是否存在
    """
    
    # 1. 初始化 Embedding
    embeddings = OpenAIEmbeddings(
        model="qwen/qwen3-embedding-8b", 
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # 2. 设置 DocStore (本地持久化存储)
    # LocalFileStore 存 bytes，EncoderBackedStore 负责把 Document 对象 pickle 成 bytes
    fs = LocalFileStore(LOCAL_STORE_PATH)
    store = EncoderBackedStore(
        store=fs,
        key_encoder=lambda x: x, # key 不需要编码
        value_serializer=pickle.dumps,      # 对应：对象 -> 字节
        value_deserializer=pickle.loads     # 对应：字节 -> 对象
    )

    # 3. 设置 Pinecone VectorStore
    if index_exists:
        print(f"检测到索引 '{INDEX_NAME}' 已存在，直接加载...")
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=embeddings
        )
    else:
        print(f"索引 '{INDEX_NAME}' 不存在，正在创建...")
        pc.create_index(
            name=INDEX_NAME, 
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region='us-east-1'),
            dimension=4096 
        )
        # 创建新实例
        vectorstore = PineconeVectorStore(
            index_name=INDEX_NAME, 
            embedding=embeddings
        )

    # 4. 初始化 MultiVectorRetriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key="doc_id"
    )

    # 5. 如果是初次运行 (index 不存在)，则执行切分和入库
    if not index_exists and docs:
        print("正在处理文档并上传至 Pinecone 和 本地存储...")
        
        # 如果本地存储目录里有脏数据，建议先清空（可选）
        # if os.path.exists(LOCAL_STORE_PATH):
        #     shutil.rmtree(LOCAL_STORE_PATH)
        
        batch_parent_ids = []
        batch_parent_docs = []
        batch_child_docs = []

        for parent_doc in docs:
            parent_id = str(uuid.uuid4())
            
            # 切分 Child Docs
            child_docs = split_markdown_by_h2(parent_doc)
            
            # 注入 ID
            for child in child_docs:
                child.metadata["doc_id"] = parent_id
            
            # 收集数据
            batch_parent_ids.append(parent_id)
            batch_parent_docs.append(parent_doc)
            batch_child_docs.extend(child_docs)

        # 批量入库 - Parent Docs 存本地
        # mset 接受 list of (key, value) tuples
        retriever.docstore.mset(list(zip(batch_parent_ids, batch_parent_docs)))
        
        # 批量入库 - Child Docs 存 Pinecone
        retriever.vectorstore.add_documents(batch_child_docs)
        
        print(f"成功处理并存储 {len(docs)} 个父文档。")
    
    elif index_exists:
        print("跳过数据处理，使用现有索引和本地缓存。")

    return retriever

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain():
    # --- 核心逻辑修改：先检查索引是否存在 ---
    
    index_exists = pc.has_index(INDEX_NAME)
    
    docs = None

    # 只有当索引不存在时，才去加载文件和清洗
    if not index_exists:
        print(">>> 初始化模式：执行全量数据清洗与入库 <<<")
        print("Loading documents...")
        docs = load_docs("./docs")

        print("Cleaning documents...")
        for doc in docs:
            # 执行你的清洗脚本
            doc.page_content = cleaner.clean_fastapi_markdown(doc.page_content, ".")
    else:
        print(">>> 加载模式：直接读取缓存 <<<")
        # 检查本地存储是否存在，如果索引在但本地没文件，可能需要报警或重新跑
        if not os.path.exists(LOCAL_STORE_PATH) or not os.listdir(LOCAL_STORE_PATH):
            print("警告：Pinecone 索引存在，但本地父文档存储为空！可能导致检索结果为空。建议删除 Pinecone 索引重新运行。")

    print("Building/Loading retriever...")
    # 传入 docs (如果是 None 也没关系，因为 index_exists=True 时不会用到 docs)
    retriever = get_retriever(docs=docs, index_exists=index_exists)

    #### RETRIEVAL and GENERATION ####
    print("Runing RAG Chain...")
    
    prompt_temp = """你是一个专门负责问答任务的助手。请结合以下检索到的上下文内容来回答问题。如果你无法从上下文中得到答案，请直接说明你不知道，不要尝试编造。回答字数请控制在三句话以内，并保持言简意赅。
    问题： {question} 
    上下文： {context} 
    答案：
    """
    prompt = PromptTemplate.from_template(prompt_temp)

    llm = ChatOpenAI(
        model="deepseek/deepseek-r1-0528:free", 
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://openrouter.ai/api/v1", 
        temperature=0.6 
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def main():
    rag_chain = get_rag_chain()

    # Question
    print("\n" + "="*30)
    query = "什么是 FastAPI？它的核心特性有哪些？"
    print(f"提问: {query}")
    
    result = rag_chain.invoke(query)
    
    print("-" * 30)
    pprint(result)

if __name__ == "__main__":
    main()