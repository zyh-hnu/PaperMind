# PaperMind/core/vectorstore.py (Updated Version)

"""
这个模块负责向量数据库的所有操作，包括：
1. 初始化嵌入模型。
2. 将文本块创建为向量并存入ChromaDB。
3. 从向量数据库中创建检索器。
"""

# PaperMind/core/vectorstore.py (Updated for Chroma Import)
# vectorstore.py 开头添加
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 清华镜像站
# 临时测试脚本
import requests
print(requests.get('https://hf-mirror.com').status_code)  # 应该返回 200

from typing import List
from langchain.schema.document import Document
# V-- [CHANGE] Updated import statement for Chroma --V
from langchain_community.vectorstores import Chroma
# A-- [CHANGE] Updated import statement for Chroma --A
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema.retriever import BaseRetriever

# ... a-rest of the file remains exactly the same ...
EMBEDDING_MODEL_NAME = 'BAAI/bge-large-zh-v1.5'
PERSIST_DIRECTORY = "./db"

def create_vectorstore(chunks: List[str]) -> Chroma:
    # ... function code is identical ...
    print("--- 开始创建向量数据库 ---")
    print(f"加载嵌入模型: {EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("嵌入模型加载完成。")
    documents = [Document(page_content=chunk) for chunk in chunks]
    print("开始计算文本嵌入并存入数据库...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    print("--- 向量数据库创建成功！---")
    return vectorstore

def get_retriever(vectorstore: Chroma, top_k: int = 4) -> BaseRetriever:
    # ... function code is identical ...
    print(f"从向量数据库中获取检索器，配置为返回 top {top_k} 个结果。")
    return vectorstore.as_retriever(search_kwargs={"k": top_k})

if __name__ == '__main__':
    # ... test code is identical ...
    from splitter import split_text
    sample_paper_text = """...""" # Same test code as before
    # (The test code from the previous step remains the same)
    from splitter import split_text

    sample_paper_text = """
Abstract
This is the abstract of the paper. We propose a new model called PaperMind.

1. Introduction
This is the introduction section. It provides background information and states the problem. The main contribution is a novel RAG-based system for scientific literature.

2. Conclusion
This is the conclusion. We found that PaperMind significantly improves reading efficiency.
    """
    
    print("--- 步骤1: 使用 splitter.py 切分文本 ---")
    chunks = split_text(sample_paper_text)
    if not chunks:
        raise ValueError("文本切分失败，无法继续测试。")
    print(f"成功将文本切分为 {len(chunks)} 个块。")

    print("\n--- 步骤2: 使用 create_vectorstore 创建向量数据库 ---")
    vector_store = create_vectorstore(chunks)
    if not isinstance(vector_store, Chroma):
        raise TypeError("创建向量数据库失败，返回类型不正确。")
    print("向量数据库实例已成功创建。")
    
    print("\n--- 步骤3: 使用 get_retriever 获取检索器 ---")
    retriever = get_retriever(vector_store)
    if not isinstance(retriever, BaseRetriever):
        raise TypeError("获取检索器失败，返回类型不正确。")
    print("检索器实例已成功获取。")

    print("\n--- 步骤4: 测试检索功能 ---")
    query = "What is PaperMind?"
    print(f"测试查询: '{query}'")
    
    retrieved_docs = retriever.invoke(query)
    
    print(f"\n检索到了 {len(retrieved_docs)} 个相关文档:")
    for i, doc in enumerate(retrieved_docs):
        print(f"\n--- 相关文档 {i+1} ---")
        print(doc.page_content)
    
    if any("PaperMind" in doc.page_content for doc in retrieved_docs):
        print("\n\n测试成功！检索到的文档中包含了关键词'PaperMind'。")
    else:
        print("\n\n测试失败！检索到的文档中未能找到关键信息。")