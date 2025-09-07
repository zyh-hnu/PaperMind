# PaperMind/core/splitter.py

"""
学术论文智能文本切分器 (v4.1 - Final Polished Version)

核心设计理念:
- **精准的正则匹配**: 采用经过优化的单一正则表达式，稳健地识别中英文、带编号、自定义等多种标题格式。
- **语义完整性优先**: 确保章节标题与其内容始终位于同一文本块的开头，为RAG提供高质量的上下文。
- **两阶段切分策略**: “宏观语义切分”+“微观递归切分”，完美平衡语义与尺寸。
- **生产就绪**: 日志输出到文件，代码结构清晰，注释完备。

注意: 需要安装 LangChain -> pip install langchain langchain-text-splitters
"""

import re
import logging
from typing import List

# 尝试导入 LangChain 分割器（兼容新旧版本）
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False

if not LANGCHAIN_AVAILABLE:
    raise ImportError("LangChain未安装，请运行: pip install langchain langchain-text-splitters")

# 配置日志（输出到项目根目录的 splitter.log 文件）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='splitter.log',
    filemode='w'  # 每次运行覆盖日志
)
logger = logging.getLogger(__name__)

# 中英文常见学术章节关键词
SECTION_KEYWORDS_EN = [
    'abstract', 'introduction', 'related work', 'background',
    'methodology', 'method', 'methods', 'experimental setup', 'experiments', 'results', 'evaluation',
    'discussion', 'conclusion', 'conclusions', 'summary', 'future work',
    'references', 'acknowledgments', 'appendix', 'appendices'
]

SECTION_KEYWORDS_CN = [
    '摘要', '引言', '绪论', '相关工作', '背景', '方法论', '方法', '实验',
    '结果', '评估', '讨论', '结论', '参考文献', '致谢', '附录',
    '总结', '未来工作'
]


def build_section_pattern() -> re.Pattern:
    """
    构建一个健壮、高效的章节标题正则表达式。

    该模式能识别以下所有情况:
    - 纯关键词标题: "Abstract", "摘要" (通常单独一行)
    - 数字编号 + 任意标题: "1. Introduction", "2.1 相关工作", "3.3.1 My Custom Title"
    """
    # 关键词部分
    en_words = '|'.join(re.escape(k) for k in SECTION_KEYWORDS_EN)
    cn_words = '|'.join(re.escape(k) for k in SECTION_KEYWORDS_CN)
    keywords = f'(?:{en_words}|{cn_words})'

    # 模式1: 纯关键词标题，通常独占一行。
    # e.g., "Abstract", "摘要"
    pattern1 = rf'^\s*{keywords}\s*$'

    # 模式2: 数字编号标题，后面可以跟任何内容。
    # e.g., "1. Introduction", "2.1. 相关工作", "3.3 My Model"
    num_pattern = r'\d+(\.\d+)*\.'
    pattern2 = rf'^\s*{num_pattern}\s*[A-Z].*$'

    # 将两个核心模式组合起来，确保从行首匹配
    full_pattern = f'({pattern1})|({pattern2})'
    
    return re.compile(full_pattern, re.IGNORECASE | re.MULTILINE)


def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> List[str]:
    """
    对学术论文文本进行智能切分的核心函数。
    """
    if not text or not text.strip():
        logger.warning("输入文本为空，返回空列表。")
        return []

    logger.info(f"开始内容感知分块，目标块大小: {chunk_size}")

    # --- 阶段一：宏观语义切分 ---
    section_pattern = build_section_pattern()
    matches = list(section_pattern.finditer(text))

    semantic_chunks = []

    if not matches:
        logger.warning("未识别到任何章节标题，全文将作为一个整体进行递归切分。")
        semantic_chunks.append(text.strip())
    else:
        logger.info(f"识别到 {len(matches)} 个章节标题，开始进行语义切分。")

        # 1. 处理第一个标题前的内容（如论文标题、作者信息、摘要前的部分）
        first_match_start = matches[0].start()
        if first_match_start > 0:
            preface = text[:first_match_start].strip()
            if preface:
                semantic_chunks.append(preface)

        # 2. 遍历所有标题，确保每个块从标题开始，到下一个标题前结束
        for i, match in enumerate(matches):
            start_index = match.start()
            end_index = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chunk = text[start_index:end_index].strip()
            if chunk:
                semantic_chunks.append(chunk)

    logger.info(f"宏观语义切分完成，共生成 {len(semantic_chunks)} 个语义块。")

    # --- 阶段二：微观递归切分 ---
    final_chunks = []
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    for chunk in semantic_chunks:
        # 仅对明显过长的块进行二次切分，增加10%缓冲
        if len(chunk) > chunk_size * 1.1:
            sub_chunks = recursive_splitter.split_text(chunk)
            # 过滤掉二次切分后可能产生的过短碎片
            final_chunks.extend([s for s in sub_chunks if len(s.strip()) > 50])
        elif len(chunk.strip()) > 20: # 过滤掉本身就过短的语义块
            final_chunks.append(chunk)

    logger.info(f"微观递归切分完成，最终生成 {len(final_chunks)} 个文本块。")
    return final_chunks

# --- 单元测试 ---
# --- 用于本地测试的示例代码 ---
if __name__ == '__main__':
    sample_text = """
    My Awesome Paper
    Author: A.I. Gemini

    Abstract
    This is the abstract. It summarizes the paper in a few sentences. It is short.

    1. Introduction
    This is the introduction. It provides background information and states the research question.
    The introduction is usually longer than the abstract. We will make this part intentionally long to test the recursive splitting.
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed non risus. Suspendisse lectus tortor, dignissim sit amet, adipiscing nec, ultricies sed, dolor. Cras elementum ultrices diam. Maecenas ligula massa, varius a, semper congue, euismod non, mi. Proin porttitor, orci nec nonummy molestie, enim est eleifend mi, non fermentum diam nisl sit amet erat. Duis semper. Duis arcu massa, scelerisque vitae, consequat in, pretium a, enim. Pellentesque congue. Ut in risus volutpat libero pharetra tempor. Cras vestibulum bibendum augue. Praesent egestas leo in pede. Praesent blandit odio eu enim. Pellentesque sed dui ut augue blandit sodales. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae; Aliquam nibh. Mauris ac mauris sed pede pellentesque fermentum. Maecenas adipiscing ante non diam.

    1.1. Background
    This is a subsection on background. It goes into more detail.

    2. Method
    This section describes the methodology used in the research. It is very important.
    3. Conclusion and Future Work
    This is the conclusion. We summarize our findings. We also suggest future work.

    References
    [1] A. Person, "A Great Paper", 2023.
    """

    # 设置一个较小的 chunk_size 以便在示例中触发递归切分
    chunks = split_text(sample_text, chunk_size=250, chunk_overlap=50)

    print(f"\n--- 切分结果 (共 {len(chunks)} 块) ---\n")
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} (长度: {len(chunk)}) ---")
        print(chunk)
        print("\n" + "="*40 + "\n")