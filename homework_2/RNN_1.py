"""RNN，词为单位"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
import numpy as np
import random
import os
import nltk

nltk.download('punkt')

# 定义提取段落的函数
def extract_paragraphs(corpus_dir, num_paragraphs, max_tokens):
    paragraphs = []
    labels = []

    # 遍历语料库中的每个txt文件
    for novel_file in os.listdir(corpus_dir):
        if novel_file.endswith('.txt'):
            novel_path = os.path.join(corpus_dir, novel_file)

            # 读取小说内容
            with open(novel_path, 'r', encoding='gbk', errors='ignore') as file:
                novel_text = file.read()

            # 根据换行符分割成段落
            novel_paragraphs = novel_text.split('\n')

            # 随机抽取一定数量的段落
            random.shuffle(novel_paragraphs)
            for paragraph in novel_paragraphs:
                # 如果段落长度不超过max_tokens，则添加到数据集中
                if len(paragraph.split()) <= max_tokens:
                    paragraphs.append(paragraph)
                    labels.append(novel_file[:-4])  # 小说文件名作为标签
                    if len(paragraphs) == num_paragraphs:
                        break

    return paragraphs, labels

# 语料库路径
corpus_dir = r'C:\Users\HP\PycharmProjects\pythonProject5\data'

# 参数设置
num_paragraphs = 1000
max_tokens = [20, 100, 500, 1000, 3000]

# 10次交叉验证
for k in max_tokens:
    print(f"Max tokens: {k}")
    paragraphs, labels = extract_paragraphs(corpus_dir, num_paragraphs, k)

    # 创建分词器
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(paragraphs)

    # 将文本转换为序列
    sequences = tokenizer.texts_to_sequences(paragraphs)

    # 设置词汇表大小
    vocab_size = len(tokenizer.word_index) + 1

    # 对序列进行填充，使它们具有相同的长度
    max_length = max([len(seq) for seq in sequences])
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

    # 拆分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

    # 创建标签编码器
    label_encoder = LabelEncoder()

    # 对训练集和测试集的标签进行编码
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # 构建 RNN 模型
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=100, input_length=max_length),
        Bidirectional(LSTM(64)),
        Dense(64, activation='relu'),
        Dense(len(set(labels)), activation='softmax')
    ])

    # 编译模型
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 训练模型
    model.fit(X_train, y_train_encoded, epochs=10, batch_size=32, validation_data=(X_test, y_test_encoded))
