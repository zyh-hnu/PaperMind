#负责构建和管理LangChain链
"""
- **`chain.py` - 对话链构建器**
    - **职责**: 组装LangChain的各个组件，构建最终的问答链或对话链。
    - **核心函数**: `get_conversation_chain(retriever)`，接收一个检索器，配置好Prompt模板、LLM（智谱AI）、以及对话记忆(Memory)，最终返回一个可以处理用户提问的`chain`对象。
"""
# PaperMind/core/chain.py
"""
这个模块负责构建和管理LangChain链。
它将检索器和Google Gemini语言模型连接起来，形成一个完整的RAG问答链。

性能优化要点（低延迟）：
- 使用 Gemini Flash 系列（默认 gemini-2.5-flash）
- 限制检索文档数量与总上下文字符数（减少提示词体积）
- 截断单文档上下文长度（避免长段文本拖慢推理）
- 限制输出最大 token 数（更快出结果）
- 复用已创建的 LLM 实例（避免重复初始化）
"""

import os
import logging
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.retriever import BaseRetriever
from langchain_google_genai import ChatGoogleGenerativeAI


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- 网络代理设置 ---
# 将代理设置放在全局，以便整个Python会话都可使用
PROXY_URL = "http://127.0.0.1:10808" # ✅ 请确保这是你正确的代理端口
if PROXY_URL:
    os.environ["HTTP_PROXY"] = PROXY_URL
    os.environ["HTTPS_PROXY"] = PROXY_URL
    logger.info(f"已设置代理: {PROXY_URL}")
    
_LLM_CACHE = {}

def _get_cached_llm(model_name: str, api_key: str, temperature: float, max_output_tokens: int) -> ChatGoogleGenerativeAI:
    """基于模型配置复用 LLM 实例以避免重复初始化成本。"""
    cache_key = (model_name, temperature, max_output_tokens)
    if cache_key in _LLM_CACHE:
        return _LLM_CACHE[cache_key]
    

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
        max_output_tokens=max_output_tokens
    )
    _LLM_CACHE[cache_key] = llm
    return llm

def get_conversation_chain(
    retriever: BaseRetriever,
    *,
    model_name: str = "gemini-2.5-flash",
    temperature: float = 1.0,
    max_output_tokens: int = 6000,
    max_docs: int = 5,
    per_doc_char_limit: int = 1000,
    total_context_char_limit: int = 4000,
):
    """
    构建一个完整的、使用Google Gemini的RAG对话链（低延迟优化）。

    参数:
        retriever: 文档检索器
        model_name: Gemini 模型名（默认 gemini-2.5-flash）
        temperature: 采样温度
        max_output_tokens: 回答的最大输出 token 数（越小越快）
        max_docs: 参与上下文拼接的文档数量上限
        per_doc_char_limit: 单文档拼接到提示词的字符上限
        total_context_char_limit: 所有上下文总字符上限
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("未能加载 GOOGLE_API_KEY。请确保你的.env文件在项目根目录且配置正确。")

    llm = _get_cached_llm(
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    prompt_template = (
        "你是一个专业的AI学术论文阅读助手。请根据下面提供的论文片段作为唯一的上下文信息，用清晰、简洁的中文回答用户的问题。"
        "如果上下文中没有足够的信息来回答问题，请直接说“根据提供的资料，我无法回答这个问题。” 不要编造任何上下文之外的信息。\n\n"
        "📌 上下文:\n{context}\n\n"
        "❓ 问题:\n{question}\n\n"
        "🤖 回答:"
    )
    prompt = PromptTemplate.from_template(prompt_template)

    def _truncate_text(text: str, limit: int) -> str:
        if limit <= 0 or len(text) <= limit:
            return text
        return text[:limit]

    def format_docs(docs):
        # 1) 限制文档数量
        selected = docs[: max_docs]
        # 2) 截断每个文档
        truncated = [_truncate_text(d.page_content, per_doc_char_limit) for d in selected]
        # 3) 合并并限制总上下文
        merged = "\n\n".join(truncated)
        if total_context_char_limit and len(merged) > total_context_char_limit:
            merged = merged[: total_context_char_limit]
        return merged

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
