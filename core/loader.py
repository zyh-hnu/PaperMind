# PaperMind/core/loader.py

"""
PDF文档加载器模块 (v2.1 - Simplified & Robust)

核心功能:
- 从PDF文件对象中高效、准确地提取文本。
- 自动将多页文本合并为一个字符串。
- 提供简洁的错误处理。
"""
import fitz  # PyMuPDF
import logging
from typing import IO

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFLoadError(Exception):
    """PDF加载过程中的自定义异常类"""
    pass

def load_pdf_text(pdf_file: IO[bytes]) -> str:
    """
    从一个二进制PDF文件对象中提取所有文本内容。

    Args:
        pdf_file (IO[bytes]): PDF文件对象, 例如由 st.file_uploader() 返回。

    Returns:
        str: 包含PDF所有页面文本的单个字符串。

    Raises:
        PDFLoadError: 当文件为空、损坏或无法解析时。
    """
    if not pdf_file:
        raise ValueError("输入的pdf_file不能为空。")

    try:
        # PyMuPDF可以直接从内存中的字节流读取
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        
        all_text = []
        for page in doc:
            all_text.append(page.get_text())
            
        full_text = "\n\n".join(all_text)

        if not full_text.strip():
            raise PDFLoadError("PDF文件中未找到可提取的文本内容。")
        
        logger.info(f"PDF加载成功: 共 {len(doc)} 页, 提取到 {len(full_text)} 个字符。")
        return full_text

    except Exception as e:
        logger.error(f"处理PDF时发生错误: {e}")
        raise PDFLoadError(f"无法处理该PDF文件。可能是文件已损坏或格式不受支持。错误详情: {e}")

# --- 单元测试 ---
if __name__ == '__main__':
    # 为了测试，请在项目根目录创建一个 `papers` 文件夹，并放入一个 `sample.pdf` 文件。
    import os
    
    test_pdf_path = os.path.join(os.path.dirname(__file__), '..', 'papers', 'sample.pdf')
    
    try:
        logger.info(f"--- 正在测试PDF加载器 ---")
        with open(test_pdf_path, 'rb') as f:
            text = load_pdf_text(f)
            logger.info(f"✅ PDF加载测试成功！预览: '{text[:200].replace(chr(10), ' ')}...'")
    except FileNotFoundError:
        logger.error(f"❌ 测试失败: 请确保测试文件存在于 {test_pdf_path}")
    except PDFLoadError as e:
        logger.error(f"❌ 测试失败: {e}")