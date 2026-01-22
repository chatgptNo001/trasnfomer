"""
数据预处理模块（伪代码）
目标：把原始文本转成模型可用的 token 序列和训练批次。
"""

from typing import List, Dict, Tuple


def clean_text(text: str) -> str:
    """
    简化清洗：去除多余空白、统一大小写等。
    真实场景会包含去噪、去重、敏感内容过滤等步骤。
    """
    return " ".join(text.lower().split())


def tokenize(text: str) -> List[str]:
    """
    伪代码：用空格分词。
    真实 LLM 通常使用 BPE / SentencePiece 等子词切分方法。
    """
    return text.split(" ")


def build_vocab(tokens: List[str]) -> Dict[str, int]:
    """
    构建词表：把 token 映射到整数 ID。
    词表大小会影响模型容量与泛化能力。
    """
    vocab = {"<pad>": 0, "<unk>": 1}
    for token in tokens:
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def numericalize(tokens: List[str], vocab: Dict[str, int]) -> List[int]:
    """
    把 token 转成 ID，未知词使用 <unk>。
    """
    unk_id = vocab["<unk>"]
    return [vocab.get(token, unk_id) for token in tokens]


def create_batches(token_ids: List[int], seq_len: int, batch_size: int) -> List[Tuple[List[int], List[int]]]:
    """
    构造训练样本：
    - 输入序列 x: token_ids[i : i+seq_len]
    - 目标序列 y: token_ids[i+1 : i+seq_len+1]
    这是经典的“下一个词预测”训练方式。
    """
    batches = []
    for i in range(0, len(token_ids) - seq_len - 1, seq_len):
        x = token_ids[i : i + seq_len]
        y = token_ids[i + 1 : i + seq_len + 1]
        batches.append((x, y))
        if len(batches) == batch_size:
            yield batches
            batches = []
    if batches:
        yield batches
