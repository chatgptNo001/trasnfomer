"""
LLM 训练流程演示（Python 伪代码）。

核心问题对应位置：
1) 学习目标：下一词预测，最小化交叉熵损失。
2) 学习过程：前向 → 损失 → 反向 → 更新。
3) 知识存储：参数（权重）更新 = 内化。
4) 泛化与局限：从训练数据学统计模式，但会出现幻觉。
"""

from pathlib import Path
from typing import List, Tuple

from tokenizer import SimpleTokenizer
from model import ToyTransformer
from utils import softmax, cross_entropy, moving_average, ascii_plot


def load_corpus(path: Path) -> List[str]:
    """
    读取语料，每行一条样本。
    """
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def compute_loss_and_grads(logits: List[List[float]], targets: List[int]) -> Tuple[float, Tuple[float, ...]]:
    """
    计算损失与梯度（伪代码）。

    交叉熵损失（单步）：
        L = -log(p_target)
    整体损失：
        L_total = (1/T) * sum_t L_t

    反向传播梯度（伪代码）：
        dL/dz = p - y
    其中 z 是 logits，p 是 softmax 概率。
    """
    total_loss = 0.0
    for step_logits, target_id in zip(logits, targets):
        probs = softmax(step_logits)
        total_loss += cross_entropy(probs, target_id)

    mean_loss = total_loss / max(1, len(targets))

    # 这里返回虚拟梯度：真实实现会计算每个参数的梯度
    grads = (mean_loss,)
    return mean_loss, grads


def train_demo() -> None:
    """
    训练主流程：
    - 数据预处理
    - 模型初始化
    - 训练循环（日志可视化）
    """
    corpus_path = Path(__file__).parent.parent / "data" / "sample_corpus.txt"
    texts = load_corpus(corpus_path)

    # 1) 数据预处理
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(texts)
    dataset = tokenizer.build_dataset(texts)

    print("[INFO] 词表大小:", tokenizer.vocab_size)
    print("[INFO] 数据样本数:", len(dataset))

    # 2) 模型初始化
    model = ToyTransformer(vocab_size=tokenizer.vocab_size, hidden_size=8)

    # 3) 训练循环（伪代码）
    losses: List[float] = []
    epochs = 3
    lr = 0.1

    for epoch in range(epochs):
        print(f"\n[Epoch {epoch + 1}]------------------------")
        for inputs, targets in dataset:
            # 前向传播
            logits = model.forward(inputs)
            # 计算损失与梯度
            loss, grads = compute_loss_and_grads(logits, targets)
            # 参数更新（知识内化）
            model.update_parameters(grads, lr=lr)
            losses.append(loss)
            print(f"loss={loss:.4f} | input={tokenizer.decode(inputs)}")

    # 4) 简易可视化/日志
    print("\n[INFO] 损失趋势（移动平均）")
    smoothed = moving_average(losses, window=3)
    ascii_plot(smoothed, width=30)

    # 5) 推理演示（泛化与幻觉）
    prompt = "小猫 喜欢"
    prompt_ids = tokenizer.encode(prompt)
    logits = model.forward(prompt_ids)
    next_token_probs = softmax(logits[-1])

    # 取最高概率 token（贪心）
    predicted_id = max(range(len(next_token_probs)), key=lambda i: next_token_probs[i])
    predicted_token = tokenizer.id_to_token[predicted_id]

    print("\n[INFER] 提示词:", prompt)
    print("[INFER] 预测下一个词:", predicted_token)

    # 6) 幻觉机制说明（打印提示）
    print("\n[NOTE] 幻觉机制：")
    print("- 模型只学到了统计关联，不具备真实世界验证能力。")
    print("- 当提示过于模糊或训练数据不足时，模型仍会生成“看似合理”的词。")
    print("- 应对方式：增加明确上下文、使用检索或事实校验、设置拒答策略。")


if __name__ == "__main__":
    train_demo()
