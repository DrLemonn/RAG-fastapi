import os
from dotenv import load_dotenv
from pprint import pprint
import bs4

from langchain_classic import hub
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from pinecone import Pinecone


# Load all environment variables from .env file
load_dotenv()
index_name = os.getenv('PINECONE_INDEX_NAME')

pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
# index = pc.Index(os.environ['PINECONE_INDEX_NAME'])

def load_docs(path="./docs"):
    # 使用 DirectoryLoader 批量加载 Markdown 文件
    loader = DirectoryLoader(
        path, 
        glob="**/*.md", 
        loader_cls=UnstructuredMarkdownLoader
    )
    return loader.load()

def split_docs(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    return text_splitter.split_documents(documents)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain():
    # Embed
    print("Check indexing...")

    embedding = OpenAIEmbeddings(
            model="qwen/qwen3-embedding-8b",  # Your Azure deployment name
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
    )

    # print(f"Debug Key: {os.getenv('OPENAI_API_KEY')[:10]}...")

    if not pc.has_index(index_name):
        print("Loading documents...")
        docs = load_docs("./docs")

        # Split
        print("Spliting documents...")
        splits = split_docs(docs)

        print("Create indexing...")
        vectorstore = PineconeVectorStore.from_documents(
            documents=splits, 
            #embedding=OpenAIEmbeddings(model="openai/text-embedding-3-large"), 
            embedding = embedding,
            index_name=index_name
        )
    else:
        print("Load indexing...")
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embedding
        )


    retriever = vectorstore.as_retriever()

    #### RETRIEVAL and GENERATION ####
    print("Runing RAG...")
    
    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    llm = ChatOpenAI(
        model="deepseek/deepseek-r1-0528:free", # 使用 OpenRouter 的模型标识符
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://openrouter.ai/api/v1", # 指向 OpenRouter 接口
        temperature=0.6 # DeepSeek R1 建议设置一定的随机性以发挥思维链能力
    )

    # Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever


def main():
    rag_chain, _= get_rag_chain()

    # Question
    pprint(rag_chain.invoke("在 FastAPI 中，async def 和普通 def 定义路由的处理逻辑有什么区别？什么时候该用哪种？"))

if __name__ == "__main__":
    main()