"""
日志与可视化模块（伪代码）
目标：用简单打印帮助理解训练过程。
"""

from typing import List


def log_metrics(step: int, loss: float) -> None:
    """
    打印训练日志。
    """
    print(f"Step {step:04d} | loss = {loss:.4f}")


def ascii_loss_curve(loss_history: List[float]) -> None:
    """
    输出一个简易的 ASCII 曲线，让初学者看到损失下降趋势。
    """
    print("\nLoss Curve (ASCII)")
    for i, loss in enumerate(loss_history[:20], start=1):
        bars = "#" * max(1, int((1.0 / (loss + 0.01))))
        print(f"{i:02d}: {bars}")
