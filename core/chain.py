# PaperMind/core/chain.py

"""
构建RAG对话链 (v4.0 - Rerank Integrated)

核心流程:
1.  **HyDE查询增强**: 接收用户问题，生成假设性文档以优化查询。
2.  **混合检索 (召回)**: 使用增强查询从EnsembleRetriever中“召回”一批候选文档（数量可以多一些，比如15-20个）。
3.  **重排序 (精排)**: 使用BGE-Reranker模型对候选文档进行精准的相关性打分和排序。
4.  **上下文选择与格式化**: 选择重排序后得分最高的N个文档，拼接成最终的上下文。
5.  **最终生成**: 将高质量的上下文和原始问题送入LLM，生成最终答案。
"""

import os
import logging
from typing import List, Dict
import torch

from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.retriever import BaseRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from sentence_transformers.cross_encoder import CrossEncoder
from langchain_core.runnables import RunnableLambda

# --- 全局配置与日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 代理设置 ---
PROXY_URL = "http://127.0.0.1:10808"
if PROXY_URL:
    os.environ["HTTP_PROXY"] = PROXY_URL
    os.environ["HTTPS_PROXY"] = PROXY_URL
    logger.info(f"已设置代理: {PROXY_URL}")

# --- 设备自动检测 ---
def get_optimal_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

DEFAULT_DEVICE = get_optimal_device()

# --- 模型缓存管理器 ---
_LLM_CACHE = {}
_RERANKER_CACHE = {}

def _get_cached_llm(model_name: str, api_key: str, temperature: float) -> ChatGoogleGenerativeAI:
    """基于配置复用LLM实例"""
    cache_key = (model_name, temperature)
    if cache_key not in _LLM_CACHE:
        logger.info(f"正在初始化新的LLM实例: model={model_name}, temp={temperature}")
        _LLM_CACHE[cache_key] = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
        )
    return _LLM_CACHE[cache_key]

def _get_cached_reranker(model_name: str, device: str) -> CrossEncoder:
    """加载并缓存Reranker模型"""
    cache_key = (model_name, device)
    if cache_key not in _RERANKER_CACHE:
        logger.info(f"⏳ 正在加载Reranker模型: {model_name} -> 设备: {device}")
        _RERANKER_CACHE[cache_key] = CrossEncoder(model_name, device=device, max_length=1024)
        logger.info("✅ Reranker模型加载完成。")
    return _RERANKER_CACHE[cache_key]


# --- 核心函数：构建对话链 ---
def get_conversation_chain(
    retriever: BaseRetriever,
    *,
    # LLM & RAG 相关参数
    llm_model_name: str = "gemini-2.5-flash",
    temperature: float = 1.0,
    rerank_model_name: str = r"D:\Project2025\PaperMind\models\bge-reranker-base", 
    rerank_top_n: int = 5, # 最终选择重排后最好的N个文档
    total_context_char_limit: int = 8000
):
    """
    构建一个集成了 HyDE、混合检索 和 Reranker 的高级RAG对话链。
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("未能加载 GOOGLE_API_KEY。请确保.env文件配置正确。")


    def format_context(docs: List[Document]) -> str:
        parts = []
        for i, doc in enumerate(docs[:rerank_top_n], 1):
            source_info = f"[来源: {doc.metadata.get('source', '未知')} | 得分: {doc.metadata.get('score', 'N/A'):.4f}]"
            parts.append(f"【段落 {i}】{source_info}\n{doc.page_content}")
        return "\n\n" + "\n\n".join(parts)

    # 1. 获取LLM和Reranker模型实例
    llm = _get_cached_llm(llm_model_name, api_key, temperature)
    reranker = _get_cached_reranker(rerank_model_name, DEFAULT_DEVICE)

    # 2. HyDE (Hypothetical Document Embeddings) 链
    hyde_template = """
    你是一位AI学术研究助理。请根据用户提出的以下问题，生成一个简洁、专业的段落，就好像它是从一篇相关学术论文中摘录出来的答案一样。
    这个段落将用于进行向量检索，因此它应该包含与问题核心概念相关的关键词、术语和关键信息。
    用户问题: "{question}"
    生成的假设性学术段落:"""
    hyde_prompt = PromptTemplate.from_template(hyde_template)
    hyde_chain = hyde_prompt | llm | StrOutputParser()

    # 3. RAG 最终问答提示
    rag_template = """
    你是一位顶级的AI学术论文阅读助手。请严格根据下面提供的“上下文信息”，用清晰、专业、简洁的中文回答用户的“问题”。
    回答规则:
    1. 你的回答必须完全基于所提供的上下文，禁止利用你的任何先验知识。
    2. 如果上下文中没有足够的信息来回答问题，必须直接回复：“根据提供的资料，我无法回答这个问题。”
    3. 保持答案的客观性和准确性。
    4. 严禁任何形式的猜测、推断或编造。如果你不确定，就说“无法回答”。

    --- 上下文信息 ---
    {context}

    --- 问题 ---
    {question}

    --- 你的回答 ---"""
    rag_prompt = PromptTemplate.from_template(rag_template)

    # 4. 定义重排序函数
    def rerank_documents(inputs: Dict) -> List[Document]:
        """对检索到的文档列表进行重排序"""
        question = inputs["question"]
        docs = inputs["documents"]
        
        if not docs:
            return []
            
        logger.info(f"🔍 Reranker正在对 {len(docs)} 个文档进行精排...")
        
        # 创建 [查询, 文档] 对
        pairs = [(question, doc.page_content) for doc in docs]
        
        # 计算相关性得分
        scores = reranker.predict(pairs, show_progress_bar=False)
        
        # 将得分与文档打包并排序
        scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        
        # 只返回重排序后的文档列表
        reranked_docs = [doc for score, doc in scored_docs]
        
        logger.info("✅ 精排完成！")
        return reranked_docs

    def retrieve_docs(query: str) -> List[Document]:
        return retriever.invoke(query)

    # 5. 构建完整的LCEL执行链
    rag_chain = (
        {
            # 第一部分：获取上下文
            "context": {
                "documents": (lambda x: x['question']) | hyde_chain | RunnableLambda(retrieve_docs),
                "question": lambda x: x['question']
            }
            | RunnableLambda(rerank_documents) # 执行重排序
            | (lambda docs: "\n\n".join(
            f"【段落 {i+1}】[来源: {doc.metadata.get('source', '未知')}]\n{doc.page_content}"
            for i, doc in enumerate(docs[:rerank_top_n])
            ))
            ,
            # 第二部分：传递原始问题
            "question": lambda x: x['question']
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    
    # 封装在一个RunnablePassthrough中，使其接收一个简单的字符串输入
    #final_chain = RunnablePassthrough.assign(question=lambda x: x) | rag_chain

    logger.info("✅ Rerank 增强的RAG对话链构建成功！")
    return rag_chain