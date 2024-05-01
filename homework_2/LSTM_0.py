'''LSTM 字为单位'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional
from sklearn.model_selection import KFold
from gensim import corpora, models
from nltk.tokenize import word_tokenize
from imblearn.over_sampling import RandomOverSampler
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

    # 从 extract_paragraphs 函数返回的 labels
    all_labels_set = set(labels)

    lda_model = train_lda_model(paragraphs, num_topics=num_topics)

    # 将段落表示为主题分布
    topics_distribution = np.zeros((len(paragraphs), num_topics))
    for i, paragraph in enumerate(paragraphs):
        bow_vector = lda_model.id2word.doc2bow(word_tokenize(paragraph))
        topics = lda_model[bow_vector]
        for topic in topics:
            topics_distribution[i, topic[0]] = topic[1]

    # 拆分数据集为训练集和测试集
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(topics_distribution):
        X_train, X_test = topics_distribution[train_index], topics_distribution[test_index]
        y_train, y_test = np.array(labels)[train_index], np.array(labels)[test_index]

        # 创建标签映射字典
        label_to_index = {label: i for i, label in enumerate(set(labels))}
        index_to_label = {i: label for label, i in label_to_index.items()}

        # 将标签转换为整数形式
        y_train_encoded = np.array([label_to_index[label] for label in y_train])
        y_test_encoded = np.array([label_to_index[label] for label in y_test])

        # 确保测试集中的标签在所有标签集合中
        unknown_labels = set(y_test) - all_labels_set
        if unknown_labels:
            # 将未知标签映射为一个通用的 "其他" 标签
            y_test = [label if label in all_labels_set else '其他' for label in y_test]

        # 过采样
        oversample = RandomOverSampler()
        X_train_over, y_train_over = oversample.fit_resample(X_train, y_train_encoded)

        # 构建LSTM模型
        model = Sequential([
            Embedding(input_dim=num_topics, output_dim=100, input_length=num_topics),
            Bidirectional(LSTM(64)),
            Dense(64, activation='relu'),
            Dense(len(set(labels)), activation='softmax')
        ])

        # 编译模型
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # 现在使用过采样后的数据进行模型的训练
        model.fit(X_train_over, y_train_over, epochs=10, batch_size=32, validation_data=(X_test, y_test_encoded))
