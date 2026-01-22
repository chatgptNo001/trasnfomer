"""
模型结构（伪代码）：
- 词嵌入（Embedding）
- 简化的自注意力（Self-Attention）
- 线性层输出词表分布

注：这里用“伪代码 + 数学说明”来帮助理解。
"""

from typing import List, Tuple
import random


class ToyTransformer:
    """
    迷你版 Transformer。
    参数（权重）代表“知识存储”。
    """

    def __init__(self, vocab_size: int, hidden_size: int = 8) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # 伪代码：随机初始化参数（实际中使用更精细的初始化）
        self.embeddings = [
            [random.uniform(-0.1, 0.1) for _ in range(hidden_size)]
            for _ in range(vocab_size)
        ]
        self.attention_wq = [
            [random.uniform(-0.1, 0.1) for _ in range(hidden_size)]
            for _ in range(hidden_size)
        ]
        self.attention_wk = [
            [random.uniform(-0.1, 0.1) for _ in range(hidden_size)]
            for _ in range(hidden_size)
        ]
        self.attention_wv = [
            [random.uniform(-0.1, 0.1) for _ in range(hidden_size)]
            for _ in range(hidden_size)
        ]
        self.output_w = [
            [random.uniform(-0.1, 0.1) for _ in range(vocab_size)]
            for _ in range(hidden_size)
        ]

    def forward(self, token_ids: List[int]) -> List[List[float]]:
        """
        前向传播：输入 token 序列，输出每个位置的 logits。

        伪公式（简化）：
        - e_t = Embedding(token_t)
        - Q = e_t W_q, K = e_t W_k, V = e_t W_v
        - Attention(Q,K,V) = softmax(QK^T / sqrt(d)) V
        - logits = attention_output W_o
        """
        # 这里返回随机 logits 作为伪代码占位
        logits = []
        for _ in token_ids:
            logit = [random.uniform(-1.0, 1.0) for _ in range(self.vocab_size)]
            logits.append(logit)
        return logits

    def update_parameters(self, grads: Tuple[float, ...], lr: float) -> None:
        """
        参数更新（伪代码）：
        theta = theta - lr * gradient

        这里不做具体矩阵更新，只是演示“知识内化”。
        """
        _ = grads
        _ = lr
        # 真实实现会对 embeddings / attention / output_w 等进行更新
        return
