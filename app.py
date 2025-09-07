# PaperMind/app.py

import streamlit as st
import time
import logging
from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入我们自己编写的核心逻辑模块
from core import loader, splitter, vectorstore, chain

# --- 页面配置 ---
st.set_page_config(
    page_title="PaperMind - 你的AI论文阅读助手",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 应用主函数 ---
def main():
    """
    PaperMind应用的主函数，负责渲染UI和协调后端逻辑。
    """
    load_dotenv()

    st.title("PaperMind 🧠")
    st.caption("与你的学术论文进行智能对话，由RAG和LLM强力驱动")

    # --- UI 侧边栏 ---
    with st.sidebar:
        st.header("🛠️ 控制面板")
        st.write("请上传你的PDF文档，点击“开始处理”按钮，即可开启智能问答。")

        pdf_file = st.file_uploader("上传你的PDF文件", type="pdf", accept_multiple_files=False)

        if st.button("开始处理", use_container_width=True, type="primary"):
            if pdf_file is not None:
                with st.spinner("🚀 AI引擎启动中，请稍候..."):
                    try:
                        # --- 这是全新的、正确的处理流程 ---
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        # 1. 加载PDF文本
                        #st.info("步骤 1/4: 正在提取PDF文本...")
                        status_text.text("步骤 1/4: 正在提取PDF文本...")
                        raw_text = loader.load_pdf_text(pdf_file)
                        progress_bar.progress(25)

                        # 2. 切分文本
                        #st.info("步骤 2/4: 正在智能切分文档...")
                        status_text.text("步骤 2/4: 正在智能切分文档...")
                        text_chunks = splitter.split_text(raw_text)
                        progress_bar.progress(50)
                        
                        # <-- FIX 1: 将纯文本块转换为LangChain的Document对象 ---
                        # 这是适配新版vectorstore的关键一步
                        documents = [Document(page_content=chunk) for chunk in text_chunks]

                        # 3. 创建混合检索器 (同时完成向量化和BM25索引)
                        status_text.text("步骤 3/4: 正在构建混合检索知识库...")
                        #st.info("步骤 3/4: 正在构建混合检索知识库...")
                        # <-- FIX 2: 调用正确的函数，并召回更多文档以供Reranker筛选 ---
                        hybrid_retriever = vectorstore.create_hybrid_retriever(
                            documents=documents,
                            top_k=20,  # 召回20篇，让Reranker有材料可选
                            model_name="BAAI/bge-base-zh-v1.5",  # 显式声明
                            device=vectorstore.DEFAULT_DEVICE,   # 使用模块内定义的设备
                        )
                        progress_bar.progress(75)

                        # 4. 创建带Reranker的RAG对话链并存入session_state
                        #st.info("步骤 4/4: 正在构建AI对话链...")
                        status_text.text("步骤 4/4: 正在构建AI对话链...")
                        # <-- FIX 3: 将创建好的混合检索器直接传入 ---
                        st.session_state.conversation_chain = chain.get_conversation_chain(
                            retriever=hybrid_retriever
                        )
                        
                        # 初始化聊天记录
                        if "chat_history" not in st.session_state:
                            st.session_state.chat_history = []
                        progress_bar.progress(100)
                        st.success("处理完成！现在可以开始提问了。")
                        
                        progress_bar.empty()
                        status_text.empty()

                    except Exception as e:
                        st.error(f"处理文档时发生致命错误: {e}")
                        logger.error(f"处理文档时发生致命错误: {e}", exc_info=True)
                        st.stop()
            else:
                st.warning("⚠️ 请先上传一个PDF文件。")

    # --- 主聊天界面 ---
    st.header("💬 对话窗口")

    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 如果没有对话链，显示引导信息
    if st.session_state.conversation_chain is None:
        st.info("请在左侧上传文档并点击“开始处理”以启动对话。")

    # 显示历史聊天记录
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 获取用户输入
    if user_question := st.chat_input("请在此输入你关于文档的问题..."):
        if st.session_state.conversation_chain is None:
            st.warning("⚠️ 请先在左侧上传并处理一个PDF文档。")
        else:
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            with st.spinner("🤖 AI正在思考，请稍候..."):
                try:
                    # 调用我们最终的、带Reranker的RAG链
                    response = st.session_state.conversation_chain.invoke({"question": user_question})
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    with st.chat_message("assistant"):
                        st.markdown(response)
                        
                except Exception as e:
                    error_message = f"获取AI回答时发生错误: {e}"
                    st.error(error_message)
                    logger.error(error_message, exc_info=True)

if __name__ == '__main__':
    main()