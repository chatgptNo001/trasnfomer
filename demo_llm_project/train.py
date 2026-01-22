"""
训练流程模块（伪代码）
目标：展示完整训练循环、损失函数、梯度更新和日志输出。
"""

from typing import Iterable, List, Tuple

from .model import TinyTransformer
from .utils import log_metrics, ascii_loss_curve


def cross_entropy_loss(logits: List[float], target_id: int) -> float:
    """
    交叉熵损失（伪代码）：
    L = -sum(y * log(p))
    其中 y 是 one-hot 标签，p 是 softmax(logits) 输出概率。
    """
    # 伪实现：返回一个占位值
    return 1.0


def train_one_epoch(
    model: TinyTransformer,
    batches: Iterable[List[Tuple[List[int], List[int]]]],
    learning_rate: float,
) -> List[float]:
    """
    训练单个 epoch，返回损失曲线。
    """
    loss_history = []

    for step, batch in enumerate(batches, start=1):
        # 伪代码：累积 batch 的损失
        batch_loss = 0.0
        for x, y in batch:
            # 前向：得到 logits
            logits = model.forward(x)

            # 计算损失（以最后一个 token 为例）
            target_id = y[-1]
            loss = cross_entropy_loss(logits, target_id)
            batch_loss += loss

            # 反向传播（伪代码）
            # grad = dL/dθ
            # θ = θ - lr * grad
            # 在真实框架中由 autograd + optimizer 完成

        batch_loss = batch_loss / max(len(batch), 1)
        loss_history.append(batch_loss)

        # 输出日志，帮助理解训练动态
        log_metrics(step=step, loss=batch_loss)

    return loss_history


def train(
    model: TinyTransformer,
    batches: Iterable[List[Tuple[List[int], List[int]]]],
    learning_rate: float,
    epochs: int,
) -> None:
    """
    训练主循环：
    展示损失随训练迭代逐步下降，体现“知识内化”。
    """
    all_losses = []
    for epoch in range(1, epochs + 1):
        print(f"\n=== Epoch {epoch}/{epochs} ===")
        loss_history = train_one_epoch(model, batches, learning_rate)
        all_losses.extend(loss_history)

    # 简单的 ASCII 曲线展示
    ascii_loss_curve(all_losses)

    # 泛化提醒：如果训练损失很低但新数据表现差，说明过拟合
    print("\n[提示] 训练损失下降 ≠ 真实泛化能力提高，需要验证集评估。")
