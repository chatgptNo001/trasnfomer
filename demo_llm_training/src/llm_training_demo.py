"""
LLM 训练流程演示（Python 伪代码）

目标：用尽量直观的方式展示“从数据到推理”的完整流程。
注意：这是“可运行风格的伪代码”，关注概念，不追求真实可执行。
"""

# =========================================
# 0) 配置与演示用超参数
# =========================================
VOCAB_SIZE = 50        # 词表大小（示意）
EMBED_DIM = 16         # 词向量维度（示意）
HIDDEN_DIM = 32        # 模型隐藏维度（示意）
MAX_LEN = 8            # 每条样本的最大长度（示意）
LEARNING_RATE = 0.1    # 学习率（示意）
EPOCHS = 3             # 训练轮数（演示）

# =========================================
# 1) 数据预处理（分词、构建词表、生成训练样本）
# =========================================
# 目标：把文本转成模型可读的“数字序列”。
# 简化示例：把每行看作“空格分词”后的序列。

def load_corpus(path):
    """读取语料（伪代码）。"""
    lines = read_lines(path)  # 假设 read_lines 返回字符串列表
    return [line.strip().split(" ") for line in lines]


def build_vocab(tokenized_lines, vocab_size):
    """构建词表，给每个词分配一个ID。"""
    # 统计频次 -> 排序 -> 选前 vocab_size
    word_freq = count_words(tokenized_lines)
    most_common = take_top_k(word_freq, k=vocab_size - 2)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, word in enumerate(most_common, start=2):
        vocab[word] = i
    return vocab


def encode(tokens, vocab, max_len):
    """把词序列转成ID序列，并做截断/填充。"""
    ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]
    ids = ids[:max_len]
    while len(ids) < max_len:
        ids.append(vocab["<PAD>"])
    return ids


def create_training_pairs(token_ids):
    """
    生成 (上下文 -> 下一个词) 训练样本。
    例：输入 ["我","喜欢","学习"]
        训练对：(["我"],"喜欢"), (["我","喜欢"],"学习")
    """
    pairs = []
    for i in range(1, len(token_ids)):
        context = token_ids[:i]
        target = token_ids[i]
        pairs.append((context, target))
    return pairs


# =========================================
# 2) 模型架构（极简 Transformer 风格示意）
# =========================================
# 真实 LLM 会更复杂，这里只保留“可解释性”。

class TinyTransformer:
    """
    一个极简的“Embedding + 简化注意力 + 输出层”的示意模型。
    参数是模型“知识”的存储位置。
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim):
        # 词向量矩阵：把离散词ID映射到连续向量
        self.W_embed = random_matrix(vocab_size, embed_dim)

        # 注意力/前馈层权重（简化）
        self.W_hidden = random_matrix(embed_dim, hidden_dim)
        self.W_out = random_matrix(hidden_dim, vocab_size)

    def forward(self, input_ids):
        """
        前向传播：输入上下文 -> 输出下一个词的概率分布
        简化：对上下文做平均后进入线性层
        """
        embeddings = [self.W_embed[idx] for idx in input_ids]
        context_vec = average(embeddings)
        hidden = relu(context_vec @ self.W_hidden)
        logits = hidden @ self.W_out
        probs = softmax(logits)
        return probs, (context_vec, hidden)


# =========================================
# 3) 损失函数与梯度更新（关键公式）
# =========================================
# 常见目标：交叉熵损失
#   L = -log p(y_true | x)
# 梯度下降更新：
#   W := W - lr * dL/dW


def cross_entropy_loss(probs, target_id):
    """交叉熵损失（伪代码）。"""
    return -log(probs[target_id] + 1e-9)


def backward_and_update(model, cache, probs, target_id, lr):
    """
    反向传播与参数更新（高度简化）。
    这里只示意“误差 -> 梯度 -> 更新”的过程。
    """
    context_vec, hidden = cache

    # 误差：预测分布与真实标签的差异
    # dL/dlogits = probs; 真实位置减 1
    grad_logits = probs
    grad_logits[target_id] -= 1

    # 参数更新（伪代码）：
    # W_out := W_out - lr * (hidden^T @ grad_logits)
    model.W_out -= lr * outer(hidden, grad_logits)

    # 继续反传到隐藏层（简化）
    grad_hidden = grad_logits @ transpose(model.W_out)
    grad_hidden = relu_backward(grad_hidden, hidden)

    # 更新 W_hidden
    model.W_hidden -= lr * outer(context_vec, grad_hidden)

    # 更新词向量：把梯度分配到上下文中的每个词
    for idx in context_vec_source_ids(context_vec):
        model.W_embed[idx] -= lr * grad_hidden


# =========================================
# 4) 训练循环（学习过程 + 日志可视化）
# =========================================

def train(model, dataset, epochs, lr):
    """
    训练循环：
    for epoch:
        for (context, target):
            forward -> loss -> backward -> update
    """
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for context_ids, target_id in dataset:
            probs, cache = model.forward(context_ids)
            loss = cross_entropy_loss(probs, target_id)
            total_loss += loss
            backward_and_update(model, cache, probs, target_id, lr)

        avg_loss = total_loss / max(len(dataset), 1)
        # 训练日志：帮助直观理解“损失下降”
        print(f"[Epoch {epoch}] avg_loss = {avg_loss:.4f}")
        print("  可视化(越短越好): " + "█" * int(avg_loss * 10))


# =========================================
# 5) 推理（生成过程）
# =========================================
# 推理时：
#   给定上下文 -> 输出概率 -> 选择下一个词 -> 拼成文本


def generate(model, prompt_ids, max_steps=5):
    """基于上下文生成后续词（伪代码）。"""
    output = list(prompt_ids)
    for _ in range(max_steps):
        probs, _ = model.forward(output)
        next_id = sample_from_distribution(probs)
        output.append(next_id)
    return output


# =========================================
# 6) 知识存储、泛化与幻觉（说明性演示）
# =========================================
# - 知识存储：体现在参数更新（W_embed/W_hidden/W_out 的变化）。
# - 泛化：模型学到的是“统计规律”，能在新组合上做合理预测。
# - 幻觉：当上下文缺乏事实约束时，模型仍会输出“看似合理”的词。


def explain_generalization_and_hallucination():
    """
    用日志提示用户：模型何时会泛化成功，何时可能产生幻觉。
    """
    print("[分析] 泛化：模型能在相似上下文中给出合理预测。")
    print("[分析] 幻觉：若上下文不足或问题超出训练分布，模型仍会编造答案。")
    print("[建议] 对关键事实进行核验，或要求模型提供来源。")


# =========================================
# 7) 入口流程（伪代码流程演示）
# =========================================

# Step 1: 数据准备
corpus = load_corpus("data/sample_corpus.txt")
vocab = build_vocab(corpus, VOCAB_SIZE)

# Step 2: 构建训练样本
all_pairs = []
for tokens in corpus:
    token_ids = encode(tokens, vocab, MAX_LEN)
    all_pairs.extend(create_training_pairs(token_ids))

# Step 3: 初始化模型
model = TinyTransformer(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM)

# Step 4: 训练
train(model, all_pairs, EPOCHS, LEARNING_RATE)

# Step 5: 推理演示
prompt = ["机器", "学习"]
prompt_ids = encode(prompt, vocab, MAX_LEN)[:2]
output_ids = generate(model, prompt_ids, max_steps=3)
print("[生成示例]", decode(output_ids, vocab))

# Step 6: 泛化与幻觉说明
explain_generalization_and_hallucination()

"""
运行后你会看到：
- 每个 epoch 的平均损失下降（学习过程）
- 生成文本的示例（推理过程）
- 泛化与幻觉的解释性日志（认知边界）
"""
