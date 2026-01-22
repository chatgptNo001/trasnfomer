"""
LLM 训练流程演示（伪代码）
把数据预处理、模型、训练、推理流程串起来。
"""

from .data_preprocess import clean_text, tokenize, build_vocab, numericalize, create_batches
from .model import TinyTransformer
from .train import train
from .inference import generate, hallucination_demo


def main() -> None:
    # 1) 准备数据（伪数据）
    raw_text = "LLM is a language model. It learns patterns from data."
    cleaned = clean_text(raw_text)
    tokens = tokenize(cleaned)
    vocab = build_vocab(tokens)
    token_ids = numericalize(tokens, vocab)

    # 2) 创建训练批次
    seq_len = 4
    batch_size = 2
    batches = create_batches(token_ids, seq_len=seq_len, batch_size=batch_size)

    # 3) 初始化模型（参数即知识存储）
    model = TinyTransformer(vocab_size=len(vocab), d_model=32, num_heads=4)

    # 4) 训练模型
    train(model, batches, learning_rate=1e-3, epochs=2)

    # 5) 推理生成
    prompt_ids = token_ids[:seq_len]
    output_ids = generate(model, prompt_ids, max_new_tokens=5)
    print("\n生成的 token ids:", output_ids)

    # 6) 幻觉演示
    hallucination_demo()


if __name__ == "__main__":
    main()
