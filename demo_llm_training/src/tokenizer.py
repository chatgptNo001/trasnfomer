"""
简易分词器（伪代码级别）。
目标：把文本拆成 token，并映射为 id。
"""

from collections import Counter
from typing import List, Dict, Tuple


class SimpleTokenizer:
    """
    用空格分词的最小化分词器。
    真实 LLM 通常使用 BPE/WordPiece 等子词算法。
    """

    def __init__(self) -> None:
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.vocab_size: int = 0

    def build_vocab(self, texts: List[str], min_freq: int = 1) -> None:
        """
        构建词表：统计词频，并过滤低频词。
        """
        counter = Counter()
        for line in texts:
            tokens = line.strip().split()
            counter.update(tokens)

        vocab = [tok for tok, freq in counter.items() if freq >= min_freq]
        # 添加特殊符号
        vocab = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"] + vocab

        self.token_to_id = {tok: i for i, tok in enumerate(vocab)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}
        self.vocab_size = len(vocab)

    def encode(self, text: str) -> List[int]:
        """
        编码：token -> id
        """
        tokens = text.strip().split()
        return [self.token_to_id.get(tok, self.token_to_id["<UNK>"]) for tok in tokens]

    def decode(self, ids: List[int]) -> str:
        """
        解码：id -> token
        """
        return " ".join(self.id_to_token.get(i, "<UNK>") for i in ids)

    def build_dataset(self, texts: List[str]) -> List[Tuple[List[int], List[int]]]:
        """
        构建“下一词预测”训练样本。
        输入序列：<BOS> x1 x2 x3
        目标序列：x1 x2 x3 <EOS>
        """
        dataset = []
        for line in texts:
            ids = [self.token_to_id["<BOS>"]] + self.encode(line) + [self.token_to_id["<EOS>"]]
            inputs = ids[:-1]
            targets = ids[1:]
            dataset.append((inputs, targets))
        return dataset
