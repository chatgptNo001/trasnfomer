"""
工具函数：数学公式、softmax、损失计算、可视化。
"""

from typing import List
import math


def softmax(logits: List[float]) -> List[float]:
    """
    softmax(z_i) = exp(z_i) / sum_j exp(z_j)
    """
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]


def cross_entropy(probs: List[float], target_index: int) -> float:
    """
    交叉熵损失：
    L = -log(p_target)
    """
    p = max(probs[target_index], 1e-12)
    return -math.log(p)


def moving_average(values: List[float], window: int = 5) -> List[float]:
    """
    简单移动平均，用于观察损失趋势。
    """
    if not values:
        return []
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start : i + 1]
        smoothed.append(sum(chunk) / len(chunk))
    return smoothed


def ascii_plot(values: List[float], width: int = 40) -> None:
    """
    纯文本“可视化”：把损失映射成条形图。
    """
    if not values:
        return
    max_val = max(values)
    min_val = min(values)
    span = max(max_val - min_val, 1e-8)
    for i, v in enumerate(values):
        bar_len = int(((v - min_val) / span) * width)
        bar = "#" * bar_len
        print(f"step {i:02d}: {bar} ({v:.4f})")
