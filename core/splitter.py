#负责文本的切分策略
# PaperMind/core/splitter.py
"""
- **`splitter.py` - 切分器**
    - **职责**: 将长文本按照特定策略切分成小块 (Chunks)。
    - **核心函数**: `split_text(text)`，接收文本，返回一个由文本块组成的列表。未来你可以把“按章节切分”的复杂逻辑实现在这里。
"""

import re
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    将输入的文本进行两阶段的、保留语义的切分。

    阶段一: 根据论文的章节标题进行语义切分。
    阶段二: 对过长的章节块进行递归字符切分。

    参数:
        text (str): 从PDF中提取的完整文本。
        chunk_size (int): 每个文本块的目标最大长度。
        chunk_overlap (int): 文本块之间的重叠字符数。

    返回:
        List[str]: 切分后的文本块列表。
    """
    if not text:
        return []

    # --- 阶段一：语义边界切分 ---
    # 定义匹配章节标题的正则表达式。
    # 这个表达式会匹配 "1. Introduction", "2.1. Related Work", "Abstract", "References" 等模式。
    # 使用 re.MULTILINE 标志，使 '^' 能够匹配每一行的开头。
    # 使用正向先行断言 `(?=...)` 来在匹配标题的位置进行切分，同时保留标题在后续文本的开头。
    semantic_split_pattern = r"(?=\n\d+\.\s.*|\n\d+\.\d+\.\s.*|\nAbstract|\nIntroduction|\nRelated Work|\nMethodology|\nExperiments|\nConclusion|\nReferences)"
    
    # 根据正则表达式进行初步切分
    semantic_chunks = re.split(semantic_split_pattern, text)
    # re.split 可能会在开头产生一个空字符串，需要过滤掉
    semantic_chunks = [chunk.strip() for chunk in semantic_chunks if chunk.strip()]

    # --- 阶段二：递归字符切分 ---
    # 初始化LangChain的递归字符切分器
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # LangChain默认的分割符就很适合通用文本
    )

    final_chunks = []
    for semantic_chunk in semantic_chunks:
        # 如果初步切分出的章节块仍然大于目标大小，则对其进行二次切分
        if len(semantic_chunk) > chunk_size:
            smaller_chunks = recursive_splitter.split_text(semantic_chunk)
            final_chunks.extend(smaller_chunks)
        else:
            # 如果章节块本身就不大，就直接作为一个整体
            final_chunks.append(semantic_chunk)

    return final_chunks

# --- 单元测试 ---
if __name__ == '__main__':
    # 创建一个模拟的、包含多种章节格式的论文文本
    sample_paper_text = """
Abstract
This is the abstract of the paper. It is a short summary of the entire work, usually less than 250 words. We propose a new model called PaperMind.

1. Introduction
This is the introduction section. It provides background information and states the problem. The introduction is often quite long, so it might need to be split into smaller pieces by the recursive splitter. Let's add more text here to ensure its length exceeds the chunk size. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. This makes the introduction long enough.

2. Related Work
This section discusses previous research related to our work. It is usually shorter than the introduction.

3. Methodology
This is the methodology section. It describes the methods and techniques used in the research.
3.1. Our Proposed Model
Here we detail our PaperMind model architecture. This is a subsection.
3.2. Datasets
Here we describe the datasets used for experiments.

4. Conclusion
This is the conclusion. It summarizes the findings and suggests future work. It is typically short.

References
[1] A. Vaswani et al., "Attention is all you need," 2017.
[2] J. Devlin et al., "BERT: Pre-training of deep bidirectional transformers for language understanding," 2018.
    """

    print("--- 正在测试文本切分功能 ---")
    
    # 使用默认参数调用我们的核心函数
    chunks = split_text(sample_paper_text)
    
    if chunks:
        print(f"\n成功将文本切分成了 {len(chunks)} 个块。\n")
        
        for i, chunk in enumerate(chunks):
            print(f"--- Chunk {i+1} (Length: {len(chunk)}) ---")
            # 打印每个块的前100个字符和后50个字符作为预览
            print(chunk[:100].strip() + "...")
            # print("...")
            # print(chunk[-50:].strip())
            print("-" * 20)
            
    else:
        print("切分失败，没有生成任何文本块。")