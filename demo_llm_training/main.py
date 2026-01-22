"""
LLM 训练流程演示（Python 伪代码版）

目标：帮助初学者理解
1) 学习目标
2) 学习过程
3) 知识存储与内化
4) 泛化能力与局限性（含幻觉机制）

注意：本文件是“可读性优先”的伪代码示例，不用于真实训练。
"""

from dataclasses import dataclass
from typing import List, Tuple
import math
import random


# ----------------------------
# 1. 数据准备与预处理
# ----------------------------

def load_toy_corpus(path: str) -> List[str]:
    """加载简单文本语料（toy data）。"""
    with open(path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def build_vocab(corpus: List[str]) -> Tuple[dict, dict]:
    """
    构建词表：word -> id 与 id -> word。
    真实 LLM 往往使用 BPE / WordPiece 等子词切分。
    """
    words = sorted(set(" ".join(corpus).split()))
    word_to_id = {word: idx for idx, word in enumerate(words)}
    id_to_word = {idx: word for word, idx in word_to_id.items()}
    return word_to_id, id_to_word


def encode_text(text: str, word_to_id: dict) -> List[int]:
    """将文本编码为 token id 序列。"""
    return [word_to_id[word] for word in text.split()]


def create_next_token_pairs(token_ids: List[int]) -> List[Tuple[int, int]]:
    """
    构造 (input_token, next_token) 训练对。
    这是最简化的“下一个词预测”训练目标。
    """
    pairs = []
    for i in range(len(token_ids) - 1):
        pairs.append((token_ids[i], token_ids[i + 1]))
    return pairs


# ----------------------------
# 2. 模型结构（极简版）
# ----------------------------

@dataclass
class TinyLM:
    """
    极简“语言模型”，用一个矩阵 W 表示 token 的嵌入 + 输出权重。

    在真实 LLM 中，模型通常包括：
    - Token Embedding
    - 多层 Transformer Block（自注意力 + 前馈网络）
    - LayerNorm、残差连接等

    这里用一个权重矩阵 W (vocab_size x vocab_size)
    直接生成下一个 token 的 logits，便于理解。
    """

    vocab_size: int
    weights: List[List[float]]  # 简化的权重矩阵

    @staticmethod
    def initialize(vocab_size: int) -> "TinyLM":
        """随机初始化权重矩阵。"""
        weights = [
            [random.uniform(-0.1, 0.1) for _ in range(vocab_size)]
            for _ in range(vocab_size)
        ]
        return TinyLM(vocab_size=vocab_size, weights=weights)

    def forward(self, input_id: int) -> List[float]:
        """
        前向传播：
        logits = W[input_id]
        即根据输入 token 的权重行生成下一个词的分数。
        """
        return self.weights[input_id]

    def predict(self, input_id: int) -> int:
        """推理：取 logits 最大值作为预测 token。"""
        logits = self.forward(input_id)
        return int(max(range(self.vocab_size), key=lambda i: logits[i]))


# ----------------------------
# 3. 损失函数与梯度更新
# ----------------------------

def softmax(logits: List[float]) -> List[float]:
    """数值稳定版 softmax。"""
    max_logit = max(logits)
    exps = [math.exp(l - max_logit) for l in logits]
    total = sum(exps)
    return [e / total for e in exps]


def cross_entropy_loss(probs: List[float], target_id: int) -> float:
    """
    交叉熵损失：
    L = -log(p_target)
    """
    return -math.log(probs[target_id] + 1e-12)


def compute_gradients(
    probs: List[float], target_id: int, vocab_size: int
) -> List[float]:
    """
    计算对 logits 的梯度：
    dL/dlogits = probs - one_hot(target)
    """
    grads = probs[:]
    grads[target_id] -= 1.0
    return grads


def update_weights(
    model: TinyLM, input_id: int, grads: List[float], lr: float
) -> None:
    """
    梯度下降更新：
    W[input_id] = W[input_id] - lr * grads

    公式：
    W = W - η * ∇W
    """
    for i in range(model.vocab_size):
        model.weights[input_id][i] -= lr * grads[i]


# ----------------------------
# 4. 训练流程
# ----------------------------

@dataclass
class TrainingConfig:
    epochs: int = 5
    lr: float = 0.1
    log_interval: int = 5


def train(model: TinyLM, pairs: List[Tuple[int, int]], config: TrainingConfig) -> None:
    """
    训练循环：
    1) 前向传播
    2) 计算损失
    3) 反向传播（梯度）
    4) 参数更新（知识存储）
    """
    step = 0
    for epoch in range(config.epochs):
        total_loss = 0.0
        random.shuffle(pairs)
        for input_id, target_id in pairs:
            logits = model.forward(input_id)
            probs = softmax(logits)
            loss = cross_entropy_loss(probs, target_id)
            grads = compute_gradients(probs, target_id, model.vocab_size)
            update_weights(model, input_id, grads, config.lr)

            total_loss += loss
            step += 1
            if step % config.log_interval == 0:
                print(
                    f"[epoch={epoch + 1}] step={step:03d} loss={loss:.4f} "
                    f"avg_loss={total_loss / step:.4f}"
                )

        print(f"Epoch {epoch + 1} done. avg_loss={total_loss / len(pairs):.4f}")


# ----------------------------
# 5. 推理与泛化/幻觉演示
# ----------------------------


def infer(model: TinyLM, seed_text: str, word_to_id: dict, id_to_word: dict) -> str:
    """
    简单推理：输入一个词，预测下一个词。
    """
    input_id = word_to_id.get(seed_text)
    if input_id is None:
        return "<OOV>（模型没有见过该词，预测不可靠）"
    pred_id = model.predict(input_id)
    return id_to_word[pred_id]


def demonstrate_generalization_and_hallucination(
    model: TinyLM, word_to_id: dict, id_to_word: dict
) -> None:
    """
    演示：
    - 泛化能力：模型在已见过的模式上预测合理结果
    - 幻觉：模型在稀有/未见过模式上“猜测”输出
    """
    print("\n[泛化能力演示]")
    print("输入: 我  -> 预测:", infer(model, "我", word_to_id, id_to_word))
    print("输入: 学习 -> 预测:", infer(model, "学习", word_to_id, id_to_word))

    print("\n[幻觉演示]")
    print("输入: 火星 -> 预测:", infer(model, "火星", word_to_id, id_to_word))
    print("说明：模型会根据已学模式输出一个词，但未必真实正确。")


# ----------------------------
# 6. 主流程
# ----------------------------


def main() -> None:
    """
    主流程概览：
    1. 数据预处理
    2. 模型初始化
    3. 训练
    4. 推理与泛化/幻觉说明
    """
    corpus = load_toy_corpus("data/sample_data.txt")
    word_to_id, id_to_word = build_vocab(corpus)

    # 编码与构造训练样本
    token_ids = []
    for sentence in corpus:
        token_ids.extend(encode_text(sentence, word_to_id))
    pairs = create_next_token_pairs(token_ids)

    # 初始化模型
    model = TinyLM.initialize(vocab_size=len(word_to_id))

    # 训练配置
    config = TrainingConfig(epochs=3, lr=0.3, log_interval=4)
    train(model, pairs, config)

    # 推理演示
    print("\n[推理演示]")
    for word in ["我", "喜欢", "模型"]:
        print(f"输入: {word} -> 预测: {infer(model, word, word_to_id, id_to_word)}")

    demonstrate_generalization_and_hallucination(model, word_to_id, id_to_word)


if __name__ == "__main__":
    main()
