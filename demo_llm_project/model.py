"""
模型结构模块（伪代码）
目标：展示简化版 Transformer 的核心组件。
"""

from typing import List


class TinyTransformer:
    """
    简化版 Transformer：
    - 参数矩阵就是“知识存储”的载体
    - 这里只展示关键公式，不追求完整可运行
    """

    def __init__(self, vocab_size: int, d_model: int, num_heads: int):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads

        # 伪代码：模型参数（权重矩阵）
        # 这些参数会在训练中不断更新，用于内化知识。
        self.W_embed = "[vocab_size x d_model]"
        self.W_q = "[d_model x d_model]"
        self.W_k = "[d_model x d_model]"
        self.W_v = "[d_model x d_model]"
        self.W_o = "[d_model x d_model]"
        self.W_ff1 = "[d_model x 4*d_model]"
        self.W_ff2 = "[4*d_model x d_model]"
        self.W_out = "[d_model x vocab_size]"

    def forward(self, token_ids: List[int]) -> List[float]:
        """
        伪代码前向计算：
        1. token_ids -> embedding
        2. self-attention
        3. feed-forward
        4. 输出 logits
        """
        # 1) embedding
        # X = embedding(token_ids)
        # X shape: [seq_len x d_model]

        # 2) self-attention
        # Q = X @ W_q
        # K = X @ W_k
        # V = X @ W_v
        # attention_scores = softmax((Q K^T) / sqrt(d_model))
        # attention_output = attention_scores @ V

        # 3) feed-forward
        # hidden = relu(attention_output @ W_ff1)
        # output = hidden @ W_ff2

        # 4) logits
        # logits = output @ W_out

        # 返回 logits（伪值）
        logits = [0.0 for _ in range(self.vocab_size)]
        return logits
