#Streamlit应用主入口（前端UI）
"""
- **功能**: **这是你项目的“脸面”**，所有与用户界面（UI）相关的代码都在这里。
- **主要职责**:
    1. 使用Streamlit构建页面布局（标题、侧边栏、上传按钮、聊天窗口等）。
    2. 接收用户的操作，例如上传文件、输入问题。
    3. 调用`core/`模块中封装好的函数来处理这些操作（例如，调用`core.loader`来加载文件，调用`core.chain`来获取答案）。
    4. 将后端返回的结果美观地展示在前端页面上。
- **开发思路**: 让`app.py`保持“轻量”，它只做“传达”和“展示”的工作，不涉及复杂的业务逻辑。
"""
# PaperMind/app.py

import streamlit as st
from dotenv import load_dotenv

# 导入我们自己编写的核心逻辑模块
from core import loader, splitter, vectorstore, chain

# --- 页面配置 ---
# st.set_page_config 必须是第一个被调用的Streamlit命令
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
    # 在应用启动时加载环境变量，这是最佳实践
    load_dotenv()

    st.title("PaperMind 🧠")
    st.caption("与你的学术论文进行智能对话，由RAG和LLM强力驱动")

    # --- UI 侧边栏 ---
    with st.sidebar:
        st.header("🛠️ 控制面板")
        st.write("请上传你的PDF文档，点击“开始处理”按钮，即可开启智能问答。")

        # PDF文件上传器
        # accept_multiple_files=False 确保用户一次只上传一个文件
        pdf_file = st.file_uploader("上传你的PDF文件", type="pdf", accept_multiple_files=False)

        # "处理"按钮
        if st.button("开始处理", use_container_width=True):
            if pdf_file is not None:
                # 使用 st.spinner 显示一个美观的加载状态
                with st.spinner("🚀 AI引擎启动中，请稍候..."):
                    try:
                        # 1. 加载PDF文本
                        st.info("步骤 1/4: 正在提取PDF文本...")
                        raw_text = loader.load_pdf_text(pdf_file)

                        # 2. 切分文本
                        st.info("步骤 2/4: 正在智能切分文档...")
                        chunks = splitter.split_text(raw_text)

                        # 3. 创建向量数据库
                        st.info("步骤 3/4: 正在构建知识向量库...")
                        vector_store_instance = vectorstore.create_vectorstore(chunks)

                        # 4. 创建RAG对话链并存入session_state
                        st.info("步骤 4/4: 正在构建AI对话链...")
                        # st.session_state 就像一个全局字典，可以在应用的多次重载之间保持数据
                        st.session_state.conversation_chain = chain.get_conversation_chain(
                            vectorstore.get_retriever(vector_store_instance)
                        )
                        
                        # 初始化聊天记录
                        st.session_state.chat_history = []
                        
                        st.success("处理完成！现在可以开始提问了。")
                        
                    except Exception as e:
                        st.error(f"处理文档时发生致命错误: {e}")
                        st.stop() # 出现错误时停止应用
            else:
                st.warning("⚠️ 请先上传一个PDF文件。")

    # --- 主聊天界面 ---
    st.header("💬 对话窗口")

    # 初始化 session_state 中的变量 (如果它们不存在的话)
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None
        st.info("请在左侧上传文档并点击“开始处理”以启动对话。")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 显示历史聊天记录
    for message in st.session_state.chat_history:
        # st.chat_message 会根据role参数显示不同角色的头像
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 获取用户输入
    if user_question := st.chat_input("请在此输入你关于文档的问题..."):
        # 首先检查对话链是否已准备好
        if st.session_state.conversation_chain is None:
            st.warning("⚠️ 请先在左侧上传并处理一个PDF文档。")
        else:
            # 将用户问题添加到历史记录并显示
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            # 获取AI回答
            with st.spinner("🤖 AI正在思考，请稍候..."):
                try:
                    # 调用我们已经构建好的RAG链来获取回答
                    response = st.session_state.conversation_chain.invoke(user_question)
                    
                    # 将AI回答添加到历史记录并显示
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    with st.chat_message("assistant"):
                        st.markdown(response)
                        
                except Exception as e:
                    st.error(f"获取AI回答时发生错误: {e}")

# --- 启动应用的入口 ---
# 这是一个Python的标准写法，确保main()只在直接运行此脚本时被调用
if __name__ == '__main__':
    main()