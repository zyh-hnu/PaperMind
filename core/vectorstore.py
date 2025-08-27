# PaperMind/core/vectorstore.py

"""
高性能向量数据库模块 (v2.2 - FAISS Optimized)

核心优化点：
1.  **FAISS 优先**: 默认使用FAISS，它在纯CPU和GPU环境下通常比Chroma更快。
2.  **GPU优先与半精度浮点 (FP16)**: 自动检测CUDA/MPS并使用FP16加速。
3.  **智能模型管理**: 通过单例模式缓存已加载的模型，避免重复加载。
4.  **代码简洁**: 提供清晰、直接的函数接口，易于调用和维护。
"""
import os
import time
import logging
from typing import List

# 确保在导入torch之前设置环境变量，以避免潜在的冲突
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

from langchain.schema.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema.retriever import BaseRetriever
import torch

# --- 全局配置与设备检测 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_optimal_device() -> str:
    if torch.cuda.is_available():
        logger.info("✅ 检测到 CUDA 设备，将使用 GPU 加速。")
        return 'cuda'
    if torch.backends.mps.is_available():
        logger.info("✅ 检测到 Apple Silicon (MPS) 设备，将使用 GPU 加速。")
        return 'mps'
    logger.info("⚠️ 未检测到 GPU，将使用 CPU。")
    return 'cpu'

DEFAULT_DEVICE = get_optimal_device()

# --- 模型管理器 ---
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
        encode_kwargs = {'normalize_embeddings': True, 'batch_size': 128} # 增大batch_size以利用GPU
        
        if device != 'cpu':
            model_kwargs['torch_dtype'] = torch.float16

        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            show_progress=True # 显示进度条
        )
        self._models_cache[cache_key] = embeddings
        logger.info(f"✅ 嵌入模型加载完成，耗时: {time.time() - start_time:.2f} 秒")
        return embeddings

MODEL_MANAGER = ModelManager()

def create_vectorstore(chunks: List[str],
                       model_name: str = "BAAI/bge-base-zh-v1.5",
                       device: str = DEFAULT_DEVICE) -> FAISS:
    if not chunks:
        raise ValueError("输入文本块列表不能为空。")

    logger.info(f"--- 开始创建FAISS向量数据库 ---")
    start_time = time.time()
    
    embeddings = MODEL_MANAGER.get_embedding_model(model_name, device)
    
    # FAISS.from_texts 内部已做高效优化
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    
    logger.info(f"✅ FAISS向量数据库创建成功！总耗时: {time.time() - start_time:.2f} 秒")
    return vectorstore

def get_retriever(vectorstore: FAISS, top_k: int = 4) -> BaseRetriever:
    logger.info(f"创建检索器，将返回 top {top_k} 个相关结果。")
    return vectorstore.as_retriever(search_kwargs={"k": top_k})

# --- 单元测试 ---
if __name__ == '__main__':
    from splitter import split_text

    sample_paper_text = ("Abstract: This paper introduces a novel method...\n" * 5 + 
                         "1. Introduction: The field of NLP...\n" * 10 +
                         "2. Methodology: We employed a technique...\n" * 15)
    
    logger.info("--- 步骤1: 切分文本 ---")
    test_chunks = split_text(sample_paper_text, chunk_size=300)
    logger.info(f"✅ 生成 {len(test_chunks)} 个文本块\n")

    logger.info(f"--- 步骤2: 创建向量数据库 (使用设备: {DEFAULT_DEVICE}) ---")
    try:
        vs = create_vectorstore(test_chunks, model_name="BAAI/bge-small-zh-v1.5")
        
        logger.info("\n--- 步骤3: 测试检索 ---")
        retriever = get_retriever(vs, top_k=2)
        query = "What is the novel method?"
        results = retriever.invoke(query)
        
        print("\n检索结果:")
        for doc in results:
            print(f"- {doc.page_content[:150]}...")
            
        logger.info("\n✅ 向量存储模块测试成功！")
            
    except Exception as e:
        logger.error(f"❌ 测试运行失败: {e}")
        logger.info("请确保已安装必要的依赖：pip install faiss-cpu sentence-transformers")