"""
推理模块（伪代码）
目标：展示从训练好的模型中生成文本，并说明幻觉风险。
"""

from typing import List

from .model import TinyTransformer


def softmax(logits: List[float]) -> List[float]:
    """
    伪代码 softmax：把 logits 转成概率。
    """
    return [1.0 / len(logits) for _ in logits]


def sample_next_token(probs: List[float], temperature: float) -> int:
    """
    伪代码采样：
    - temperature 越大，采样越“发散”，更容易产生幻觉
    - temperature 越小，输出更保守
    """
    # 简化：返回最大概率的索引
    return int(max(range(len(probs)), key=lambda i: probs[i]))


def generate(
    model: TinyTransformer,
    prompt_ids: List[int],
    max_new_tokens: int = 20,
    temperature: float = 0.8,
) -> List[int]:
    """
    生成过程：每次预测下一个 token，再拼接回去。
    """
    tokens = list(prompt_ids)

    for _ in range(max_new_tokens):
        logits = model.forward(tokens)
        probs = softmax(logits)
        next_id = sample_next_token(probs, temperature)
        tokens.append(next_id)

    return tokens


def hallucination_demo() -> None:
    """
    说明幻觉产生机制：
    - 模型只学习“概率分布”，不保证事实正确
    - 若上下文不足或超出训练数据范围，模型会“猜测”
    """
    print("[幻觉示例] 模型可能给出看似合理但事实错误的答案。")
    print("[应对策略] 使用检索、查证来源、降低 temperature。")
