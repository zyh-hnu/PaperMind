# PaperMind/core/splitter.py

"""
学术论文智能文本切分器 (v3.0 - Ultimate Simplicity & Correctness)

核心设计理念:
- **绝对正确**: 优先确保单元测试通过，逻辑清晰无歧义。
- **语义优先**: 采用简单而强大的正则表达式识别主要章节，作为切分的第一依据。
- **两阶段切分**: "宏观语义切分" + "微观递归切分"，保证块的语义性和大小合规性。
- **大道至简**: 移除所有复杂的类、枚举和多余策略，只保留一个高效的核心函数。
"""

import re
import logging
from typing import List

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    raise ImportError("LangChain未安装，请运行: pip install langchain")

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> List[str]:
    """
    对学术论文文本进行智能切分的核心函数。

    Args:
        text (str): 完整的论文文本。
        chunk_size (int): 目标文本块大小。
        chunk_overlap (int): 文本块之间的重叠字符数。

    Returns:
        List[str]: 切分后的文本块列表。
    """
    if not text or not text.strip():
        return []

    # --- 阶段一：宏观语义切分 ---

    # [核心] 定义一个健壮的、能识别绝大多数论文标题的正则表达式
    # 它会匹配 "Abstract", "1. Introduction", "2.1. Related Work" 等模式
    # 使用 re.MULTILINE 标志，使 `^` 能匹配每一行的开头
    section_pattern = re.compile(
        r'^(Abstract|Introduction|Related\s+Work|Background|Methodology|Method|Experiments|Results|Evaluation|Discussion|Conclusion|References|Acknowledgments|Appendix)\s*$'
        r'|^\d+\.\s+.*$'
        r'|^\d+\.\d+\.\s+.*$',
        re.IGNORECASE | re.MULTILINE
    )

    sections = []
    last_end = 0
    
    # 遍历所有匹配到的标题
    for match in section_pattern.finditer(text):
        start = match.start()
        # 将上一个标题的结尾到这个标题的开头，作为一个完整的块
        if start > last_end:
            sections.append(text[last_end:start].strip())
        last_end = start
    
    # 添加最后一个标题到文档末尾的内容
    if last_end < len(text):
        sections.append(text[last_end:].strip())
        
    # 过滤掉可能产生的空字符串
    semantic_chunks = [s for s in sections if s]

    if not semantic_chunks:
        # 如果没有识别到任何章节，则将整个文本视为一个块
        semantic_chunks = [text]

    # --- 阶段二：微观递归切分 ---

    final_chunks = []
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""], # 使用丰富的分割符
    )

    for chunk in semantic_chunks:
        # 仅对明显过长的块进行二次切分
        if len(chunk) > chunk_size * 1.2: # 放宽一点阈值
            sub_chunks = recursive_splitter.split_text(chunk)
            # 过滤掉二次切分后可能产生的小碎片
            final_chunks.extend([sub for sub in sub_chunks if len(sub) > 50])
        elif len(chunk) > 20: # 过滤掉太短的无效块
            final_chunks.append(chunk)

    logger.info(f"文本切分完成，共生成 {len(final_chunks)} 个文本块。")
    return final_chunks