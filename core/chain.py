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
"""

import os
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.retriever import BaseRetriever
from langchain_google_genai import ChatGoogleGenerativeAI

# --- 网络代理设置 ---
# 将代理设置放在全局，以便整个Python会话都可使用
PROXY_URL = "http://127.0.0.1:10808" # ✅ 请确保这是你正确的代理端口
if PROXY_URL:
    os.environ["HTTP_PROXY"] = PROXY_URL
    os.environ["HTTPS_PROXY"] = PROXY_URL

def get_conversation_chain(retriever: BaseRetriever):
    """
    构建一个完整的、使用Google Gemini的RAG对话链。
    """
    # [修正] 关键改动：将API Key的检查移入函数内部！
    # 这样只有在创建链的时候才会检查，而不是在导入模块时。
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("未能加载 GOOGLE_API_KEY。请确保你的.env文件在项目根目录且配置正确。")
    
    # 初始化Gemini Pro模型
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # ✅ 关键修复：使用稳定且正确的免费模型名称
        google_api_key=api_key, # 明确传递api_key参数，更稳健
        temperature=0.1,
    )

    prompt_template = """
    请根据以下上下文信息，简洁、准确地回答用户的问题。
    如果你在上下文中找不到答案，请明确说“根据提供的文档，我无法回答这个问题”，不要编造信息。
    请使用中文进行回答。

    上下文:
    {context}

    问题:
    {question}

    回答:
    """
    
    prompt = PromptTemplate.from_template(prompt_template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# --- 单元测试 ---
if __name__ == '__main__':
    # [修正] 关键改动：只在直接运行此文件进行测试时，才加载.env
    from dotenv import load_dotenv
    print("正在以测试模式运行 chain.py...")
    # load_dotenv() 默认会查找当前工作目录下的.env文件
    # 当你在项目根目录运行 `python core/chain.py` 时，这会正确加载 D:/Project2025/PaperMind/.env
    if load_dotenv():
        print(".env 文件加载成功！")
    else:
        print("警告: 未找到 .env 文件，请确保它在项目根目录。")

    from splitter import split_text
    from vectorstore import create_vectorstore, get_retriever

    sample_text = """
    PaperMind是一个基于RAG架构的AI科研助手。它的核心功能是允许用户与多篇论文进行对话。
    该系统由一位优秀的工程师在2025年开发，旨在提升科研效率。
    """
    
    print("\n--- 步骤1: 切分文本 ---")
    chunks = split_text(sample_text)
    
    print("--- 步骤2: 创建向量库和检索器 ---")
    vs = create_vectorstore(chunks)
    retriever = get_retriever(vs)
    
    print("--- 步骤3: 构建RAG链 (使用Google Gemini) ---")
    rag_chain = get_conversation_chain(retriever)
    
    print("--- 步骤4: 测试链的调用 ---")
    question = "PaperMind的核心功能是什么？"
    print(f"测试问题: {question}")
    
    response = rag_chain.invoke(question)
    
    print("\n模型的回答:")
    print(response)
    
    if "RAG" in response and "对话" in response:
        print("\n测试成功！模型根据上下文正确回答了问题。")
    else:
        print("\n测试失败！模型的回答不符合预期。")