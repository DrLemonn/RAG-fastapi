import os
import pickle
import time
import uuid
import jieba
from dotenv import load_dotenv
from pprint import pprint

import cleaner_no_dedup as cleaner

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document

from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder

# Load environment variables
load_dotenv()

# --- Configuration ---
INDEX_NAME = "fastapi-hybrid-multiquery"
BM25_ENCODER_PATH = "bm25_encoder.pkl"
ALPHA = 0.7  # Hybrid search weight: 1.0=pure dense, 0.0=pure sparse, 0.5=balanced

# Initialize Pinecone
pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])


def get_tokens(text):
    """Chinese tokenizer for BM25 - space-separated jieba tokens"""
    return " ".join(jieba.cut(text))


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
    """Split document by ## headers"""
    headers_to_split_on = [("##", "section")]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    chunks = splitter.split_text(doc.page_content)

    # Preserve original metadata
    for chunk in chunks:
        original_meta = chunk.metadata.copy()
        chunk.metadata.update(doc.metadata)
        chunk.metadata.update(original_meta)

    return chunks


def hybrid_scale(dense, sparse, alpha=0.5):
    """
    Scale dense and sparse vectors for hybrid search

    Args:
        dense: Dense vector (list of floats)
        sparse: Sparse vector (dict with 'indices' and 'values')
        alpha: Weight parameter (1.0=pure dense, 0.0=pure sparse, 0.5=balanced)

    Returns:
        Tuple of (scaled_dense, scaled_sparse)
    """
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")

    # Scale sparse vector by (1 - alpha)
    h_sparse = {
        'indices': sparse['indices'],
        'values': [v * (1 - alpha) for v in sparse['values']]
    }
    # Scale dense vector by alpha
    h_dense = [v * alpha for v in dense]

    return h_dense, h_sparse


def setup_hybrid_index(embeddings, force_recreate=False):
    """
    Setup Pinecone index with hybrid search support (dense + sparse vectors)

    Returns:
        tuple: (index, bm25_encoder, chunks) - Pinecone index, fitted BM25 encoder, and document chunks
    """
    index_exists = pc.has_index(INDEX_NAME)

    # Check if BM25 encoder exists
    bm25_exists = os.path.exists(BM25_ENCODER_PATH)

    if index_exists and bm25_exists and not force_recreate:
        print(f"Loading existing index '{INDEX_NAME}' and BM25 encoder...")
        index = pc.Index(INDEX_NAME)

        # Load BM25 encoder
        with open(BM25_ENCODER_PATH, 'rb') as f:
            bm25 = pickle.load(f)

        # We don't have chunks stored, but that's OK for query-only mode
        return index, bm25, None

    else:
        print(f"Creating new hybrid index '{INDEX_NAME}'...")

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

        # Train BM25 encoder on corpus
        print("Training BM25 encoder...")
        bm25 = BM25Encoder()
        corpus = [get_tokens(chunk.page_content) for chunk in all_chunks]
        bm25.fit(corpus)

        # Save BM25 encoder
        print(f"Saving BM25 encoder to {BM25_ENCODER_PATH}...")
        with open(BM25_ENCODER_PATH, 'wb') as f:
            pickle.dump(bm25, f)

        # Create Pinecone index with dotproduct metric (required for hybrid search)
        if index_exists:
            print(f"Deleting existing index '{INDEX_NAME}'...")
            pc.delete_index(INDEX_NAME)
            time.sleep(1)

        print("Creating Pinecone index with hybrid search support...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=3072,  # text-embedding-3-large dimension
            metric="dotproduct",  # REQUIRED for hybrid search
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )

        # Wait for index to be ready
        while not pc.describe_index(INDEX_NAME).status['ready']:
            print("Waiting for index to be ready...")
            time.sleep(1)

        index = pc.Index(INDEX_NAME)

        # Generate and upsert vectors
        print("Generating dense and sparse vectors...")
        vectors_to_upsert = []

        for i, chunk in enumerate(all_chunks):
            # Generate dense vector (OpenAI embedding)
            dense_vec = embeddings.embed_query(chunk.page_content)

            # Generate sparse vector (BM25)
            sparse_vec = bm25.encode_documents(get_tokens(chunk.page_content))

            # Create vector with both dense and sparse
            vectors_to_upsert.append({
                "id": chunk.metadata.get("id", str(uuid.uuid4())),
                "values": dense_vec,
                "sparse_values": sparse_vec,
                "metadata": {
                    "text": chunk.page_content,
                    "source": chunk.metadata.get("source", ""),
                    "section": chunk.metadata.get("section", "")
                }
            })

            # Batch upsert every 100 vectors
            if (i + 1) % 100 == 0 or i == len(all_chunks) - 1:
                print(f"Upserting vectors {i - len(vectors_to_upsert) + 2} to {i + 1}...")
                index.upsert(vectors=vectors_to_upsert)
                vectors_to_upsert = []

        print(f"Successfully indexed {len(all_chunks)} chunks with hybrid vectors")
        print("Waiting for index consistency...")
        time.sleep(2)

        return index, bm25, all_chunks


def hybrid_query(query_text, index, embeddings, bm25, top_k=4, alpha=ALPHA):
    """
    Perform hybrid search query combining dense and sparse vectors

    Args:
        query_text: Query string
        index: Pinecone index
        embeddings: OpenAI embeddings model
        bm25: Fitted BM25 encoder
        top_k: Number of results to return
        alpha: Hybrid weight (1.0=dense only, 0.0=sparse only, 0.5=balanced)

    Returns:
        List of matches from Pinecone
    """
    # Generate dense query vector
    q_dense = embeddings.embed_query(query_text)

    # Generate sparse query vector
    q_sparse = bm25.encode_queries(get_tokens(query_text))

    # Scale vectors by alpha
    hdense, hsparse = hybrid_scale(q_dense, q_sparse, alpha)

    # Single Pinecone query with both dense and sparse
    results = index.query(
        vector=hdense,
        sparse_vector=hsparse,
        top_k=top_k,
        include_metadata=True
    )

    return results['matches']


def generate_query_variations(query, llm, num_variations=3):
    """
    Generate query variations using LLM for multi-query retrieval

    Args:
        query: Original query string
        llm: Language model
        num_variations: Number of variations to generate

    Returns:
        List of query strings including original
    """
    prompt = f"""你是一个帮助生成多种搜索查询的AI助手。
用户会提出一个关于FastAPI文档的问题，你需要生成{num_variations}个不同的提问方式，以便从不同角度检索相关文档。

这些变体应该：
1. 保留原始问题的技术术语（如 OAuth2, Pydantic, FastAPI 等）
2. 使用不同的句式结构
3. 从不同的角度理解问题

原始问题: {query}

请提供{num_variations}个查询变体，每行一个，不要编号，直接输出问题："""

    response = llm.invoke(prompt)

    # Extract variations from response
    variations = [line.strip() for line in response.content.split('\n') if line.strip()]

    # Include original query
    all_queries = [query] + variations[:num_variations]

    return all_queries


def multi_query_hybrid_search(query_text, index, embeddings, bm25, llm, top_k=4, alpha=ALPHA, num_variations=3):
    """
    Perform multi-query hybrid search

    Args:
        query_text: Original query
        index: Pinecone index
        embeddings: OpenAI embeddings
        bm25: Fitted BM25 encoder
        llm: Language model for query generation
        top_k: Number of final results
        alpha: Hybrid search weight
        num_variations: Number of query variations to generate

    Returns:
        List of unique top-k documents
    """
    # Generate query variations
    print(f"\nGenerating query variations for: '{query_text}'")
    variations = generate_query_variations(query_text, llm, num_variations)
    print(f"Query variations:")
    for i, var in enumerate(variations):
        print(f"  {i+1}. {var}")

    # Search with each variation
    all_results = []
    for var in variations:
        results = hybrid_query(var, index, embeddings, bm25, top_k=top_k, alpha=alpha)
        all_results.extend(results)

    # Deduplicate by ID and keep highest score
    seen = {}
    for result in all_results:
        doc_id = result['id']
        if doc_id not in seen or result['score'] > seen[doc_id]['score']:
            seen[doc_id] = result

    # Sort by score and take top_k
    unique_results = list(seen.values())
    unique_results.sort(key=lambda x: x['score'], reverse=True)

    return unique_results[:top_k]


def get_rag_components():
    """
    Initialize and return RAG components

    Returns:
        tuple: (index, embeddings, bm25, llm)
    """
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-large",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Setup hybrid index
    index, bm25, _ = setup_hybrid_index(embeddings)

    # Initialize LLM
    llm = ChatOpenAI(
        model="deepseek/deepseek-r1-0528:free",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0.6
    )

    return index, embeddings, bm25, llm


def rag_answer(query_text, index, embeddings, bm25, llm, top_k=4, alpha=ALPHA):
    """
    Generate answer using multi-query hybrid RAG

    Args:
        query_text: User question
        index: Pinecone index
        embeddings: OpenAI embeddings
        bm25: BM25 encoder
        llm: Language model
        top_k: Number of documents to retrieve
        alpha: Hybrid search weight

    Returns:
        Generated answer string
    """
    # Multi-query hybrid retrieval
    print("\n" + "="*60)
    print("RETRIEVAL PHASE")
    print("="*60)
    docs = multi_query_hybrid_search(query_text, index, embeddings, bm25, llm, top_k=top_k, alpha=alpha)

    print(f"\nRetrieved {len(docs)} unique documents:")
    for i, doc in enumerate(docs):
        print(f"\n  [{i+1}] Score: {doc['score']:.4f}")
        print(f"      Section: {doc['metadata'].get('section', 'N/A')}")
        print(f"      Preview: {doc['metadata']['text'][:100]}...")

    # Format context
    context = "\n\n".join([doc['metadata']['text'] for doc in docs])

    # Generate answer
    print("\n" + "="*60)
    print("GENERATION PHASE")
    print("="*60)

    prompt_text = f"""你是一个专门负责问答任务的助手。请结合以下检索到的上下文内容来回答问题。如果你无法从上下文中得到答案，请直接说明你不知道，不要尝试编造。回答字数请控制在三句话以内，并保持言简意赅。
问题： {query_text}
上下文： {context}
答案：
"""

    response = llm.invoke(prompt_text)

    return response.content


def main():
    """Main function to demonstrate hybrid + multi-query RAG"""
    print("="*60)
    print("Hybrid Search + Multi-Query RAG Pipeline")
    print("="*60)

    # Initialize components
    index, embeddings, bm25, llm = get_rag_components()

    # Test queries
    test_queries = [
        "FastAPI是什么？有哪些关键特性？",
        "如何使用Pydantic进行数据验证？",
        "OAuth2 安全性怎么实现？"
    ]

    for query in test_queries:
        print(f"\n\n{'='*60}")
        print(f"QUERY: {query}")
        print('='*60)

        answer = rag_answer(query, index, embeddings, bm25, llm, top_k=4, alpha=ALPHA)

        print(f"\nFINAL ANSWER:")
        print("-"*60)
        print(answer)
        print("-"*60)


if __name__ == "__main__":
    main()
