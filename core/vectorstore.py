# PaperMind/core/vectorstore.py

"""
高性能混合检索模块 (v3.1 - Production Grade)

核心设计:
1.  **拥抱LangChain生态**: 完全使用官方、经过优化的 BM25Retriever 和 EnsembleRetriever 组件，确保最佳性能和兼容性。
2.  **先进的融合算法**: 利用 EnsembleRetriever 内置的倒数排名融合 (Reciprocal Rank Fusion, RRF) 算法，智能地合并稀疏和稠密检索结果，效果远超手动加权。
3.  **架构解耦**: 本模块专注于“检索器”的构建，完全移除了对LLM的依赖（HyDE逻辑将移至chain.py处理），遵循关注点分离的最佳实践。
4.  **简洁与高效**: 代码更少，功能更强，可维护性更高。
"""
import os
import time
import logging
from typing import List

# 确保在导入torch之前设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from langchain.schema.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema.retriever import BaseRetriever
import torch

# --- 全局配置与日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import asyncio
from concurrent.futures import ThreadPoolExecutor

def retrieve_in_parallel(retrievers, query):
    def _invoke(r):
        return r.invoke(query)
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(_invoke, r) for r in retrievers]
        results = [f.result() for f in futures]
    return results

# --- 设备自动检测 ---
def get_optimal_device() -> str:
    """自动检测并返回最佳可用设备 (CUDA > MPS > CPU)"""
    if torch.cuda.is_available():
        logger.info("✅ 检测到 CUDA 设备，将使用 GPU 加速。")
        return 'cuda'
    if torch.backends.mps.is_available():
        logger.info("✅ 检测到 Apple Silicon (MPS) 设备，将使用 GPU 加速。")
        return 'mps'
    logger.info("⚠️ 未检测到 GPU，将使用 CPU。")
    return 'cpu'

DEFAULT_DEVICE = get_optimal_device()

# --- 模型管理器 (单例模式，避免重复加载) ---
class ModelManager:
    _instance = None
    _models_cache = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_embedding_model(self, model_name: str, device: str) -> HuggingFaceEmbeddings:
        cache_key = f"{model_name}_{device}"
        if cache_key in self._models_cache:
            logger.info(f"🚀 从缓存中复用嵌入模型: {model_name} on {device}")
            return self._models_cache[cache_key]

        logger.info(f"⏳ 正在加载嵌入模型: {model_name} -> 设备: {device}")
        start_time = time.time()
        
        model_kwargs = {'device': device}
        encode_kwargs = {'normalize_embeddings': True, 'batch_size': 128}
        
        if device != 'cpu':
            model_kwargs['torch_dtype'] = torch.float16

        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            show_progress=True
        )
        self._models_cache[cache_key] = embeddings
        logger.info(f"✅ 嵌入模型加载完成，耗时: {time.time() - start_time:.2f} 秒")
        return embeddings

MODEL_MANAGER = ModelManager()

# --- 核心函数：创建混合检索器 ---
def create_hybrid_retriever(
    documents: List[Document],
    top_k: int = 8,
    model_name: str = "BAAI/bge-base-zh-v1.5",
    device: str = DEFAULT_DEVICE,
    bm25_weight: float = 0.5,
    faiss_weight: float = 0.5
) -> BaseRetriever:
    """
    创建并返回一个集成了 BM25 (稀疏) 和 FAISS (稠密) 的高级混合检索器。

    Args:
        documents (List[Document]): LangChain的文档对象列表。
        top_k (int): 每个检索器希望召回的文档数量。
        model_name (str): 用于稠密检索的嵌入模型。
        device (str): 运行嵌入模型的设备。
        bm25_weight (float): BM25 在RRF融合中的权重。
        faiss_weight (float): FAISS 在RRF融合中的权重。

    Returns:
        BaseRetriever: 一个配置好的EnsembleRetriever实例，可直接用于LangChain链。
    """
    if not documents:
        raise ValueError("输入的文档列表不能为空。")

    logger.info("--- 开始创建混合检索器 (BM25 + FAISS) ---")
    start_time = time.time()

    # 1. 创建 BM25 (稀疏) 检索器
    # 使用LangChain官方组件，它经过优化并且兼容整个生态
    logger.info(f"正在初始化 BM25Retriever (k={top_k})...")
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = top_k

    # 2. 创建 FAISS (稠密) 检索器
    logger.info(f"正在初始化 FAISS vectorstore and retriever (k={top_k})...")
    embeddings = MODEL_MANAGER.get_embedding_model(model_name, device)
    
    vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    # 3. 创建 Ensemble (混合) 检索器
    # 使用RRF算法进行智能融合，无需手动处理分数
    logger.info(f"正在初始化 EnsembleRetriever (weights: BM25={bm25_weight}, FAISS={faiss_weight})...")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[bm25_weight, faiss_weight]
    )
    
    logger.info(f"✅ 混合检索器创建成功！总耗时: {time.time() - start_time:.2f} 秒")
    return ensemble_retriever
