# PaperMind: 你的AI科研论文阅读与分析助手 🚀

[Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)

[Framework](https://img.shields.io/badge/Framework-Streamlit-red)

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

**PaperMind** 是一个基于检索增强生成（RAG）技术的智能应用，旨在解决科研工作者、学生和工程师在处理海量学术论文时面临的信息过载和阅读效率低下的核心痛点。

通过本工具，你可以与单篇或多篇论文进行深度对话，快速提取核心信息、对比不同研究的异同，并以前所未有的效率完成文献综述（Literature Review）工作。

### 🎬 应用演示

*(在这里放一张展示应用核心功能的GIF动图会非常加分！)*

#加图片

*(这是一个示例图片，请替换为你自己应用的截图或GIF)*

---

### ✨ 核心功能

- **📝 单篇论文深度问答**: 上传一篇PDF论文，即可就其内容进行自由提问，例如“这篇论文的核心贡献是什么？”或“请解释一下文中的XX算法”。
- **📚 多论文知识库构建**: 将多篇相关论文（例如同一个课题的所有参考文献）上传，自动构建一个专属的向量知识库。
- **🧠 跨文档综合分析**: 基于构建的多论文知识库进行提问，实现跨论文的信息整合与对比。例如：“总结一下这些论文中解决过拟合问题的主流方法及其优缺点。”
- **🔍 答案来源可追溯**: 每一个由AI生成的答案都会附上其所依据的原文片段，确保信息的准确性和可信度。
- **🌐 (规划中) arXiv API集成**: 输入关键词，自动拉取最新的预印本论文并加入知识库，时刻追踪前沿动态。

---

### 🎯 项目动机

作为一名研究生，我深刻体会到科研工作的挑战性。每周都需要阅读大量的前沿论文，但传统阅读方式耗时巨大，且读过的知识点容易遗忘、难以串联。通用的大语言模型（如ChatGPT）因其训练数据的时效性问题，无法对最新的研究成果进行分析。

为了解决这一痛点，我决定利用RAG技术，打造一个真正能为科研工作提效的AI助手，将研究者从繁琐的“信息搬运”中解放出来，更专注于“创新思考”。

---

### 🛠️ 技术栈

| 分类              | 技术                                                       |
| --------------- | -------------------------------------------------------- |
| **前端**          | `Streamlit`                                              |
| **后端API**       | `Python`, `FastAPI` (规划中)                                |
| **AI / RAG 核心** | `LangChain`, `Zhipu AI (GLM-4)`, `Sentence-Transformers` |
| **向量数据库**       | `ChromaDB`                                               |
| **数据处理**        | `PyMuPDF`                                                |

### ⚙️ 安装与运行

### 1. 克隆项目

```bash
git clone <https://github.com/YOUR_USERNAME/PaperMind.git>
cd PaperMind
```

### 2. 创建并激活虚拟环境

```bash
# for macOS/Linux
python3 -m venv venv
source venv/bin/activate

# for Windows
python -m venv venv
.\\venv\\Scripts\\activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

复制 `.env.example` 文件并重命名为 `.env`。

```bash
cp .env.example .env
```

然后，在 `.env` 文件中填入你的智谱AI API密钥。

```
ZHIPUAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"
```

### 5. 运行应用

```bash
streamlit run app.py
```

现在，在你的浏览器中打开 `http://localhost:8501` 即可开始使用！

---

### 🚀 未来规划 (Roadmap)

- [ ] **优化UI/UX**: 提升交互体验，增加会话历史记录的可视化管理。
- [ ] **增强PDF解析**: 优化对论文中**表格、图片和公式**的识别与提取能力。
- [ ] **知识图谱可视化**: 提取论文中的关键实体（如模型、数据集），并以图谱形式展示它们之间的关联。
- [ ] **集成Zotero/Mendeley**: 与主流文献管理工具打通，直接读取文献库。
- [ ] **部署上线**: 将应用部署到云平台，方便随时随地访问。

---

### 📜 开源许可

本项目采用 [MIT License](https://www.notion.so/zyhi71/LICENSE) 开源。