import os
from dotenv import load_dotenv
from pprint import pprint

import cleaner_no_dedup as cleaner

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from pinecone import Pinecone, ServerlessSpec

# Load all environment variables from .env file
load_dotenv()

# --- Configuration ---
INDEX_NAME = "fastapi-chunks-only"

# Initialize Pinecone
pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])


def load_docs(path="./docs"):
    """Load raw Markdown files"""
    loader = DirectoryLoader(
        path,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    return loader.load()


def split_markdown_by_h2(doc: Document):
    """
    Split document by ## headers
    """
    headers_to_split_on = [("##", "section")]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    chunks = splitter.split_text(doc.page_content)

    # Preserve original metadata
    for chunk in chunks:
        original_meta = chunk.metadata.copy()
        chunk.metadata.update(doc.metadata)
        chunk.metadata.update(original_meta)

    return chunks


def get_rag_chain():
    # Check if index exists
    index_exists = pc.has_index(INDEX_NAME)

    # Initialize Embedding
    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-large",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Create or load vectorstore
    if index_exists:
        print(f"Loading existing index '{INDEX_NAME}'...")
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=embeddings
        )
    else:
        print(f"Creating new index '{INDEX_NAME}'...")

        # Load and clean documents
        print("Loading documents...")
        docs = load_docs("./docs")

        print("Cleaning documents...")
        for doc in docs:
            doc.page_content = cleaner.clean_fastapi_markdown(doc.page_content, ".")

        # Split all documents into chunks
        print("Splitting documents...")
        all_chunks = []
        for doc in docs:
            chunks = split_markdown_by_h2(doc)
            all_chunks.extend(chunks)

        print(f"Generated {len(all_chunks)} chunks from {len(docs)} documents")

        # Create Pinecone index
        pc.create_index(
            name=INDEX_NAME,
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region='us-east-1'),
            dimension=3072
        )

        # Create vectorstore and add documents
        print("Uploading chunks to Pinecone...")
        vectorstore = PineconeVectorStore.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            index_name=INDEX_NAME
        )

        print(f"Successfully indexed {len(all_chunks)} chunks")

    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    #### RETRIEVAL and GENERATION ####
    print("Running RAG Chain...")

    prompt_temp = """你是一个专门负责问答任务的助手。请结合以下检索到的上下文内容来回答问题。如果你无法从上下文中得到答案，请直接说明你不知道，不要尝试编造。回答字数请控制在三句话以内，并保持言简意赅。
    问题： {question}
    上下文： {context}
    答案：
    """
    prompt = PromptTemplate.from_template(prompt_temp)

    # LLM
    llm = ChatOpenAI(
        model="deepseek/deepseek-r1-0528:free",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0.6
    )

    # Format docs helper
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


def main():
    rag_chain, _ = get_rag_chain()

    # Question
    print("\n" + "="*30)
    query = "FastAPI是什么？有哪些关键特性？"
    print(f"提问: {query}")

    result = rag_chain.invoke(query)

    print("-" * 30)
    pprint(result)


if __name__ == "__main__":
    main()
