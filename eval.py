import os
import rag_data_cleaned, rag_baseline, rag_chunks_only
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import ContextPrecision, Faithfulness, ResponseRelevancy
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from openai import OpenAI
import pandas as pd

#mode = "data_cleaned"

# 1. 获取生产环境的 Chain 和 Retriever
# rag_chain, retriever= rag_data_cleaned.get_rag_chain() if mode == "data_cleaned" else rag_baseline
rag_chain, retriever= rag_chunks_only.get_rag_chain()

# 2. 准备测试集 (根据你的文档)
sample_queries = [
    "什么是 FastAPI？它的核心特性有哪些？",
    "FastAPI 与 Pydantic 是什么关系？为什么它依赖 Pydantic？",
    "FastAPI 的依赖注入系统（Dependency Injection）是如何工作的？请举例说明 Depends 的用法。",
    "在 FastAPI 中，async def 和普通 def 定义路由的处理逻辑有什么区别？什么时候该用哪种？",
    "FastAPI 官方推荐的 OAuth2 密码流（Password flow）实现步骤是什么？",
    "如何使用 Security 处理具有不同权限作用域（Scopes）的依赖项？",
    "FastAPI 相比于 Flask 和 Django，在性能和开发体验上的核心优势是什么？",
    "为什么在生产环境中通常建议使用 Gunicorn 配合 Uvicorn 工作，而不是直接运行 Uvicorn？",
    "为什么我在路由中使用了 async def，但性能却变慢了（常见于阻塞性 IO 误用）？",
    "在 FastAPI 中，多个依赖项（Nested Dependencies）的执行顺序是怎样的？"
]

expected_responses = [
    "FastAPI 是一个用于构建 API 的现代、高性能的 Python Web 框架，基于标准 Python 类型提示。核心特性包括：1. 高性能：可与 NodeJS 和 Go 并肩（归功于 Starlette 和 Pydantic）；2. 开发速度快：提高功能开发速度约 200% 至 300%；3. 减少 Bug：减少约 40% 的人为错误；4. 直观：强大的编辑器支持和自动补全；5. 简单：易于学习和使用；6. 代码短：最小化代码重复；7. 健壮：生产级代码，自动交互式文档；8. 标准：完全兼容 OpenAPI 和 JSON Schema。",
    
    "FastAPI 与 Pydantic 是深度集成的关系。FastAPI 使用 Pydantic 来进行数据的定义、校验、序列化（数据转换）和自动生成 OpenAPI Schema。它依赖 Pydantic 的原因在于：Pydantic 提供了强大的类型声明能力，使得 FastAPI 能够自动验证传入的请求数据是否符合预期格式，并将复杂的数据库对象或 Python 模型自动转换为 JSON 格式。此外，这种基于类型提示的定义方式直接驱动了 Swagger UI 的自动生成。",
    
    "FastAPI 的依赖注入（DI）系统允许你声明路由函数运行前需要执行的逻辑。FastAPI 负责在运行路由前解析这些依赖，并将其结果注入给函数。用法举例：定义一个函数 `common_params(q: str = None, skip: int = 0)`，然后在路由中使用 `params: dict = Depends(common_params)`。当请求到达时，FastAPI 会先调用 `common_params`，处理 URL 参数，再将返回的字典传递给路由。这有助于代码复用、共享数据库连接和实现安全验证。",
    
    " 在 FastAPI 中，如果你使用 async def 定义路由，FastAPI 会直接在主线程的异步事件循环中调用它；如果你使用普通 def 定义路由，FastAPI 会在一个独立的外部线程池中运行它，以防止阻塞主事件循环。使用建议：如果你使用的第三方库支持 await（如异步数据库驱动），请使用 async def；如果你必须使用同步/阻塞的 IO 操作（如 requests 库或同步数据库连接），请使用普通 def。误在 async def 中调用阻塞代码会直接拖慢整个应用的并发能力。",
    
    "FastAPI 官方推荐的实现步骤如下：1. 创建一个 OAuth2PasswordBearer 实例，并指定 tokenUrl（获取 Token 的端点）；2. 定义一个 Pydantic 模型作为 Token 的响应格式（包含 access_token 和 token_type）；3. 编写一个登录路由，接收 OAuth2PasswordRequestForm，验证用户名密码后生成并返回 JWT；4. 创建一个 get_current_user 依赖项，通过 Depends(oauth2_scheme) 获取 Token，解码并验证用户身份；5. 在需要授权的路径操作中使用该依赖项。",
    
    "处理 Scopes 时，FastAPI 使用 Security 类。1. 在 OAuth2PasswordBearer 中定义 scopes 映射（key 为作用域名称，value 为描述）；2. 在依赖项函数中使用 Security(oauth2_scheme, scopes=['items:read'])，这表示该依赖项需要特定的权限；3. 依赖项可以通过 SecurityScopes 对象获取当前路由请求的所有 scopes，并将其与 JWT token 中解码出的 scopes 进行比对，如果不匹配则抛出 401 或 403 异常。",
    
    "核心优势：1. 性能：基于 ASGI 标准（Starlette），原生支持异步并发，QPS 远超基于 WSGI 的 Flask 和 Django；2. 开发效率：自动生成交互式文档（Swagger/ReDoc），省去手动写 API 文档的时间；3. 类型安全：强制使用类型提示，IDE 自动补全极佳，大幅减少运行时错误；4. 自动校验：内置 Pydantic 校验，无需像 Flask 那样手动解析 JSON 或像 Django 那样写复杂的 Form/Serializer。",
    
    " Gunicorn 是一个成熟的进程管理器（Process Manager），而 Uvicorn 是一个高性能的 ASGI 服务器（Worker）。在生产环境中，Gunicorn 充当 Master 进程，负责监控、重启崩溃的子进程、处理信号量以及平滑重载代码；而 Uvicorn 负责在 Worker 进程内高效地解析异步网络协议。这种结合可以充分利用多核 CPU，并提供生产环境所需的稳定性、可伸缩性和容错性。",
    
    "这是因为在 async def 定义的路由中使用了阻塞性（Blocking）代码。由于 async 路由运行在单线程的事件循环（Event Loop）上，一旦执行了如 time.sleep()、同步数据库驱动查询或同步 HTTP 请求等操作，整个线程会被独占。这意味着在阻塞完成前，服务器无法处理任何其他请求，导致并发性能降为 1。在这种情况下，应将路由改为普通 def，让 FastAPI 将其分发到线程池处理。",
    
    " FastAPI 采用深度优先（Depth-First）的顺序解析依赖树。1. 如果依赖 A 依赖于依赖 B，则 B 会先于 A 执行；2. 在同一个请求中，FastAPI 默认会缓存依赖项的结果。例如，如果路由函数同时依赖了 A 和 B，而 A 和 B 都依赖了 C，那么 C 只会被执行一次，其结果会被缓存并复用给 A 和 B。如果需要每次都重新执行，可以设置 use_cache=False。"
]

# 高级测试用例 - 更贴近真实用户场景
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

dataset = []
for query, reference in zip(advanced_sample_queries, advanced_expected_responses):
    # 模拟 RAG 过程提取数据
    relevant_docs = retriever.invoke(query)
    response = rag_chain.invoke(query)
    
    dataset.append({
        "user_input": query,
        "retrieved_contexts": [doc.page_content for doc in relevant_docs],
        "response": response,
        "reference": reference,
    })

# 3. 执行评估
eval_dataset = EvaluationDataset.from_list(dataset)

llm = ChatOpenAI(
        # model="deepseek/deepseek-r1-0528:free",
        model="google/gemini-3-flash-preview",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://openrouter.ai/api/v1", # 指向 OpenRouter 接口
        temperature=0 
    )

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large", # 确保 OpenRouter 支持这个模型
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1" 
)


evaluator_llm = LangchainLLMWrapper(llm) # 建议评估用更强的模型

result = evaluate(
    dataset=eval_dataset,
    metrics=[ContextPrecision(), Faithfulness(), ResponseRelevancy()],
    llm=evaluator_llm,
    embeddings=embeddings
)

# 将评估结果转换为 pandas DataFrame
df = result.to_pandas()

# 设置显示所有列，方便在终端查看
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 50)


# 如果你想保存到本地以便在 Excel 中分析（强烈推荐）
df.to_csv("rag_eval_details.csv", index=False, encoding='utf-8-sig')
print("\n详细数据已导出至: rag_eval_details.csv")