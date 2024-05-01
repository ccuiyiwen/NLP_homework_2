'''#将 RNN 模型改为使用词嵌入，并重新训练模型'''
"""RNN，字为单位"""


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from gensim import corpora, models
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

# 定义LDA模型
def train_lda_model(paragraphs, num_topics):
    # 创建字典和语料库
    dictionary = corpora.Dictionary([word_tokenize(paragraph) for paragraph in paragraphs])
    corpus = [dictionary.doc2bow(word_tokenize(paragraph)) for paragraph in paragraphs]

    # 训练LDA模型
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary)

    return lda_model

# 设置主题数量
num_topics = 10

# 10次交叉验证
for k in max_tokens:
    print(f"Max tokens: {k}")
    paragraphs, labels = extract_paragraphs(corpus_dir, num_paragraphs, k)

    lda_model = train_lda_model(paragraphs, num_topics=num_topics)

    # 将段落表示为主题分布
    topics_distribution = np.zeros((len(paragraphs), num_topics))
    for i, paragraph in enumerate(paragraphs):
        bow_vector = lda_model.id2word.doc2bow(word_tokenize(paragraph))
        topics = lda_model[bow_vector]
        for topic in topics:
            topics_distribution[i, topic[0]] = topic[1]

    # 拆分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(topics_distribution, labels, test_size=0.2, random_state=42)
    print(type(X_train))
    print(type(X_test))
    print(type(y_train))
    print(type(y_test))
    # 创建标签编码器
    label_encoder = LabelEncoder()

    # 对训练集和测试集的标签进行编码
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # 构建 RNN 模型
    model = Sequential([
        Embedding(input_dim=num_topics, output_dim=100, input_length=num_topics),
        Bidirectional(LSTM(64)),
        Dense(64, activation='relu'),
        Dense(len(set(labels)), activation='softmax')
    ])

    # 编译模型
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 训练模型
    model.fit(X_train, y_train_encoded, epochs=10, batch_size=32, validation_data=(X_test, y_test_encoded))