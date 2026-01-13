import os
import time
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

# Import all RAG pipelines
import rag_chunks_only
import rag_data_cleaned
import rag_hybrid_multiquery

from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import ContextPrecision, Faithfulness, ResponseRelevancy
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables
load_dotenv()

# Advanced test queries from eval.py
advanced_sample_queries = [
    # --- 语义鸿沟 (用户不知道专业术语) ---
    "怎么保证前端传给我的 JSON 数据里，'price' 字段一定是数字而不是字符串？如果不合法自动报错吗？",
    "我想在好几个接口里复用同一段代码，比如检查用户 Token，但我不想写装饰器，也不想在每个函数里写一遍调用。",

    # --- 错误前提 (用户问题包含错误假设) ---
    "如何在 FastAPI 的 settings.py 里配置内置的 ORM 连接数据库？",
    "为什么我在 path operation 里用了 threading.Thread 去开新线程处理任务，请求还是阻塞了？",

    # --- 多跳推理 (答案需要综合多个概念) ---
    "我要做计算密集型任务，应该用 async def 还是普通的 def？这跟 Node.js 的处理方式一样吗？",
    "我要记录请求处理时间，FastAPI 的中间件和依赖注入都能做拦截，选哪个更好？",

    # --- 边缘情况 ---
    "我有一个字段既可以是 int 也可以是 float，甚至有时候是 string，怎么定义 Schema 能让它通过校验？",
    "我想在服务器启动前预加载一个很大的机器学习模型到内存里，服务关闭时再释放掉，代码应该写在哪里？",
    "如果我想在这个接口上同时用 OAuth2 验证，又要校验一个 API Key header，怎么把这两个安全验证串起来？",
    "Uvicorn 的 workers 数量设置成多少合适？是不是越多越好？跟 CPU 核数有什么关系？"
]

advanced_expected_responses = [
    # 1. JSON 数据校验 (语义鸿沟)
    "你应该使用 Pydantic 模型来定义数据结构。通过创建一个继承自 `pydantic.BaseModel` 的类，并将 'price' 字段声明为 `float` 或 `int` 类型（例如 `price: float`），FastAPI 会自动进行数据校验。如果前端传来的数据类型不匹配（且无法转换），FastAPI 会自动拦截请求并返回 422 Unprocessable Entity 错误，其中包含详细的错误信息，无需手动编写报错逻辑。",

    # 2. 代码复用/Token检查 (语义鸿沟)
    "你应该使用 FastAPI 的依赖注入系统（Dependency Injection）。你可以定义一个普通的函数（例如 `get_current_user`）来包含 Token 检查逻辑，然后在需要的路由函数参数中使用 `Depends(get_current_user)`。这比装饰器更灵活，因为依赖项可以像普通函数一样接收参数，并且可以被 FastAPI 自动处理和测试，同时也支持依赖项的嵌套。",

    # 3. Settings/ORM 配置 (错误前提)
    "这是一个误解。FastAPI 是一个无主见（Unopinionated）的框架，它没有内置的 ORM（对象关系映射）或类似 Django 的 `settings.py` 配置文件。你可以自由选择任何数据库库（如 SQLAlchemy, SQLModel, Tortoise ORM 等）。通常，我们会使用 Pydantic 的 `BaseSettings` 来管理环境变量和配置，并在单独的文件（如 `database.py`）中手动初始化数据库连接。",

    # 4. Threading 阻塞 (错误前提)
    "如果在 `async def` 定义的路由中执行耗时操作（即使是启动线程的开销），仍可能阻塞主事件循环（Event Loop）。FastAPI 的 `async def` 路由运行在单线程循环中。如果你需要进行阻塞性操作或利用多核 CPU，建议不要手动管理 `threading.Thread`，而是将路由定义为普通的 `def`（FastAPI 会自动将其放入外部线程池运行），或者使用 `fastapi.concurrency.run_in_threadpool` 显式将任务派发到线程池。",

    # 5. 计算密集型任务选择 (多跳推理)
    "对于计算密集型（CPU-bound）任务，你应该使用普通的 `def` 定义路由，或者使用 `run_in_process`。因为 `async def` 运行在主事件循环上，计算密集型任务会长时间占用 CPU，导致整个服务无法响应其他请求（阻塞 Loop）。这与 Node.js 类似，Node.js 也是单线程事件循环，如果进行繁重计算也会阻塞整个进程，通常需要 Worker Threads 来解决。但在 FastAPI 中，简单地使用 `def` 就能利用线程池，这通常足以应对非极端的计算需求。",

    # 6. 中间件 vs 依赖注入 (多跳推理)
    "如果目的是记录整个请求的处理时间（包括序列化、验证和网络传输），应该选择中间件（Middleware）（如 `BaseHTTPMiddleware`）。因为中间件作用于请求-响应的整个生命周期，能够捕获从请求到达服务器到响应发送回客户端的完整耗时。而依赖注入（Dependencies）通常在路由匹配和部分验证之后才执行，且无法轻易捕获响应发送后的时间点。",

    # 7. 多种类型字段 (边缘情况)
    "你可以使用 Python `typing` 模块中的 `Union` 类型。在 Pydantic 模型中，将字段定义为 `Union[int, float, str]`。Pydantic 会按照定义的顺序尝试进行类型校验和转换。如果一定要接收任意类型，可以使用 `Any`，但这会失去校验的意义。对于复杂的条件校验，还可以结合 Pydantic 的 `validator` 装饰器进行自定义逻辑判断。",

    # 8. 生命周期/预加载 (边缘情况)
    "代码应该写在 Lifespan（生命周期）事件处理器中。FastAPI 推荐使用 `contextlib.asynccontextmanager` 装饰器创建一个异步上下文管理器，并将其传递给 `FastAPI(lifespan=...)` 参数。在该函数 `yield` 关键字之前的代码会在应用启动（Startup）时执行（适合加载模型），`yield` 之后的代码会在应用关闭（Shutdown）时执行（适合释放内存）。不建议再使用已废弃的 `@app.on_event`。",

    # 9. 组合 OAuth2 和 API Key (边缘情况)
    "你可以在路由操作函数（Path Operation）中同时声明多个依赖项。可以将它们作为参数分别注入，例如 `def route(token: str = Depends(oauth2_scheme), key: str = Depends(api_key_header))`。如果不需要在函数内部使用这些返回值，也可以将它们放入路由装饰器的 `dependencies` 参数列表中，例如 `@app.get('/items', dependencies=[Depends(oauth2_scheme), Depends(verify_api_key)])`，这样 FastAPI 会按顺序执行所有安全验证。",

    # 10. Uvicorn Workers 设置 (边缘情况)
    "Uvicorn 的 workers 数量并非越多越好。官方建议的经验值通常是 CPU 核心数（num_cores）或 CPU 核心数 + 1。因为 Python 的全局解释器锁（GIL）限制了单个进程只能利用一个 CPU 核心，过多的 Worker 只会增加上下文切换的开销并消耗更多内存。在生产环境中，通常配合 Gunicorn 作为进程管理器，通过 `-w` 参数设置 worker 数量，并指定 `-k uvicorn.workers.UvicornWorker` 类。"
]


def evaluate_pipeline(pipeline_name, rag_chain=None, retriever=None, hybrid_components=None):
    """
    Evaluate a single RAG pipeline

    Args:
        pipeline_name: Name of the pipeline (for logging)
        rag_chain: RAG chain (for non-hybrid pipelines)
        retriever: Retriever (for non-hybrid pipelines)
        hybrid_components: Tuple of (index, embeddings, bm25, llm) for hybrid pipeline

    Returns:
        List of evaluation data dictionaries
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {pipeline_name}")
    print('='*60)

    dataset = []
    total_time = 0

    for i, (query, reference) in enumerate(zip(advanced_sample_queries, advanced_expected_responses)):
        print(f"\n[{i+1}/{len(advanced_sample_queries)}] Query: {query[:50]}...")

        start_time = time.time()

        try:
            if hybrid_components:
                # Hybrid pipeline
                index, embeddings, bm25, llm = hybrid_components

                # Get retrieved contexts
                docs = rag_hybrid_multiquery.multi_query_hybrid_search(
                    query, index, embeddings, bm25, llm, top_k=4, alpha=0.5, num_variations=3
                )
                retrieved_contexts = [doc['metadata']['text'] for doc in docs]

                # Generate response
                context = "\n\n".join(retrieved_contexts)
                prompt_text = f"""你是一个专门负责问答任务的助手。请结合以下检索到的上下文内容来回答问题。如果你无法从上下文中得到答案，请直接说明你不知道，不要尝试编造。回答字数请控制在三句话以内，并保持言简意赅。
问题： {query}
上下文： {context}
答案：
"""
                response = llm.invoke(prompt_text)
                response_text = response.content

            else:
                # Standard LangChain pipeline
                relevant_docs = retriever.invoke(query)
                retrieved_contexts = [doc.page_content for doc in relevant_docs]
                response_text = rag_chain.invoke(query)

            elapsed_time = time.time() - start_time
            total_time += elapsed_time

            dataset.append({
                "user_input": query,
                "retrieved_contexts": retrieved_contexts,
                "response": response_text,
                "reference": reference,
                "pipeline": pipeline_name,
                "latency_seconds": round(elapsed_time, 2)
            })

            print(f"  ✓ Completed in {elapsed_time:.2f}s")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            dataset.append({
                "user_input": query,
                "retrieved_contexts": [],
                "response": f"ERROR: {str(e)}",
                "reference": reference,
                "pipeline": pipeline_name,
                "latency_seconds": 0
            })

    avg_latency = total_time / len(advanced_sample_queries)
    print(f"\n{pipeline_name} - Average latency: {avg_latency:.2f}s")

    return dataset


def main():
    print("="*60)
    print("Comparative RAG Pipeline Evaluation")
    print("="*60)
    print("\nComparing 3 pipelines:")
    print("  1. chunks_only - Direct chunk retrieval")
    print("  2. data_cleaned - Parent-child with MultiVectorRetriever")
    print("  3. hybrid_multiquery - Hybrid search + Multi-query")
    print("\nMetrics: ContextPrecision, Faithfulness, ResponseRelevancy")

    # Initialize all pipelines
    print("\n" + "="*60)
    print("INITIALIZATION PHASE")
    print("="*60)

    print("\n[1/3] Loading chunks_only pipeline...")
    chunks_chain, chunks_retriever = rag_chunks_only.get_rag_chain()

    print("\n[2/3] Loading data_cleaned pipeline...")
    cleaned_chain, cleaned_retriever = rag_data_cleaned.get_rag_chain()

    print("\n[3/3] Loading hybrid_multiquery pipeline...")
    hybrid_index, hybrid_embeddings, hybrid_bm25, hybrid_llm = rag_hybrid_multiquery.get_rag_components()

    # Evaluate each pipeline
    print("\n" + "="*60)
    print("EVALUATION PHASE")
    print("="*60)

    all_data = []

    # Evaluate chunks_only
    chunks_data = evaluate_pipeline(
        "chunks_only",
        rag_chain=chunks_chain,
        retriever=chunks_retriever
    )
    print(f"\nchunks_data has {len(chunks_data)} entries")
    all_data.extend(chunks_data)

    # Evaluate data_cleaned
    cleaned_data = evaluate_pipeline(
        "data_cleaned",
        rag_chain=cleaned_chain,
        retriever=cleaned_retriever
    )
    print(f"\ncleaned_data has {len(cleaned_data)} entries")
    all_data.extend(cleaned_data)

    # Evaluate hybrid_multiquery
    hybrid_data = evaluate_pipeline(
        "hybrid_multiquery",
        hybrid_components=(hybrid_index, hybrid_embeddings, hybrid_bm25, hybrid_llm)
    )
    print(f"\nhybrid_data has {len(hybrid_data)} entries")
    all_data.extend(hybrid_data)

    print(f"\nTotal all_data before RAGAS: {len(all_data)} entries")

    # Run RAGAS evaluation
    print("\n" + "="*60)
    print("RAGAS METRICS CALCULATION")
    print("="*60)

    # Separate metadata from evaluation data
    # Add unique row_id to prevent Cartesian product during merge
    metadata_fields = []
    eval_data = []
    for i, item in enumerate(all_data):
        metadata_fields.append({
            'row_id': i,
            'pipeline': item['pipeline'],
            'latency_seconds': item['latency_seconds']
        })
        eval_data.append({
            'row_id': i,
            'user_input': item['user_input'],
            'retrieved_contexts': item['retrieved_contexts'],
            'response': item['response'],
            'reference': item['reference']
        })

    eval_dataset = EvaluationDataset.from_list(eval_data)

    # Initialize evaluator LLM and embeddings
    llm = ChatOpenAI(
        model="google/gemini-3-flash-preview",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0
    )

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )

    evaluator_llm = LangchainLLMWrapper(llm)

    print("\nCalculating RAGAS metrics (this may take a few minutes)...")
    result = evaluate(
        dataset=eval_dataset,
        metrics=[ContextPrecision(), Faithfulness(), ResponseRelevancy()],
        llm=evaluator_llm,
        embeddings=embeddings
    )

    # Convert to DataFrame
    df = result.to_pandas()
    print(f"\nRAGAS result DataFrame has {len(df)} rows")

    # Add metadata back to DataFrame
    # RAGAS preserves order, so we can add columns directly
    metadata_df = pd.DataFrame(metadata_fields)
    print(f"Metadata DataFrame has {len(metadata_df)} rows")

    # Verify same length
    if len(df) != len(metadata_df):
        print(f"WARNING: Length mismatch! df={len(df)}, metadata={len(metadata_df)}")

    # Add metadata columns directly (relies on order preservation)
    df['pipeline'] = metadata_df['pipeline'].values
    df['latency_seconds'] = metadata_df['latency_seconds'].values

    print(f"Final DataFrame has {len(df)} rows")

    # Add summary statistics
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    # Print available columns for debugging
    print("\nAvailable columns:", df.columns.tolist())

    # Find the metric columns (they might have different names)
    metric_cols = [col for col in df.columns if col not in ['user_input', 'retrieved_contexts', 'response', 'reference', 'pipeline', 'latency_seconds', 'row_id']]
    print(f"Metric columns: {metric_cols}")

    # Build aggregation dict dynamically
    agg_dict = {'latency_seconds': 'mean'}
    for col in metric_cols:
        agg_dict[col] = 'mean'

    summary = df.groupby('pipeline').agg(agg_dict).round(4)

    print("\n" + summary.to_string())

    # Export detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_filename = f"eval_comparison_detailed_{timestamp}.csv"
    summary_filename = f"eval_comparison_summary_{timestamp}.csv"

    df.to_csv(detail_filename, index=False, encoding='utf-8-sig')
    summary.to_csv(summary_filename, encoding='utf-8-sig')

    print(f"\n✓ Detailed results exported to: {detail_filename}")
    print(f"✓ Summary results exported to: {summary_filename}")

    # Identify winner
    print("\n" + "="*60)
    print("WINNER ANALYSIS")
    print("="*60)

    # Print best pipeline for each metric
    print("\n")
    for col in metric_cols:
        best_pipeline = summary[col].idxmax()
        best_score = summary.loc[best_pipeline, col]
        print(f"🏆 Best {col}: {best_pipeline} ({best_score:.4f})")

    fastest = summary['latency_seconds'].idxmin()
    print(f"⚡ Fastest: {fastest} ({summary.loc[fastest, 'latency_seconds']:.2f}s)")

    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
