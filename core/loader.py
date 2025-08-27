#负责加载和解析文档
# PaperMind/core/loader.py
"""
- **`loader.py` - 加载器**
    - **职责**: 从数据源读取信息。
    - **核心函数**: `load_pdf_text(pdf_file)`，接收一个PDF文件对象，返回提取出的纯文本。
"""

import fitz  # PyMuPDF
from typing import IO

def load_pdf_text(pdf_file: IO[bytes]) -> str:
    """
    从一个PDF文件对象中提取所有文本内容。

    这个函数专门设计用于处理由Streamlit的file_uploader上传的文件对象，
    这些对象是以二进制I/O流的形式存在的。

    参数:
        pdf_file (IO[bytes]): 一个二进制文件对象，例如由 st.file_uploader() 返回的对象。

    返回:
        str: 包含PDF所有页面文本的单个字符串。

    异常:
        - 如果文件不是一个有效的PDF或已损坏，PyMuPDF可能会抛出异常。
          调用此函数的上层代码（如app.py）应该处理这些异常。
    """
    all_text = []
    try:
        # PyMuPDF可以直接从字节流中打开PDF文件
        with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
            # 遍历PDF的每一页
            for page in doc:
                # 提取当前页的文本
                all_text.append(page.get_text())
        
        # 将所有页面的文本合并成一个长字符串，用换行符分隔
        return "\n".join(all_text)
    
    except Exception as e:
        # 如果在处理过程中发生任何错误，打印错误信息并返回一个空字符串
        # 在实际应用中，这里应该使用更完善的日志记录
        print(f"Error processing PDF file: {e}")
        return ""