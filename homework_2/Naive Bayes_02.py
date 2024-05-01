"""分类器为朴素贝叶斯"""
"""以词为单位"""
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models
import numpy as np
import random
import os
import nltk
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

nltk.download('punkt')


def extract_paragraphs(corpus_dir, num_paragraphs, max_tokens):
    paragraphs = []
    labels = []

    # 遍历语料库中的每个txt文件
    for novel_file in os.listdir(corpus_dir):
        if novel_file.endswith('.txt'):
            novel_path = os.path.join(corpus_dir, novel_file)

            # 读取小说内容
            with open(novel_path, 'r', encoding='gbk', errors='ignore') as file:  # 使用gbk编码
                novel_text = file.read()

            # 根据换行符分割成段落
            novel_paragraphs = novel_text.split('\n')

            # 随机抽取一定数量的段落
            random.shuffle(novel_paragraphs)
            for paragraph in novel_paragraphs:
                # 如果段落长度不超过max_tokens，则添加到数据集中
                words = nltk.word_tokenize(paragraph)
                if len(words) <= max_tokens:
                    paragraphs.append(words)
                    labels.append(novel_file[:-4])  # 小说文件名作为标签
                    if len(paragraphs) == num_paragraphs:
                        return paragraphs, labels

    return paragraphs, labels


# 语料库路径
corpus_dir = r'C:\Users\HP\PycharmProjects\pythonProject5\data'

# 参数设置
num_paragraphs = 1000
max_tokens = [20, 100, 500, 1000, 3000]

# 提取段落
paragraphs_all = {}
labels_all = {}
for k in max_tokens:
    paragraphs, labels = extract_paragraphs(corpus_dir, num_paragraphs, k)
    paragraphs_all[k] = paragraphs
    labels_all[k] = labels


# 定义LDA模型
def train_lda_model(paragraphs, num_topics):
    # 创建字典和语料库
    dictionary = corpora.Dictionary(paragraphs)
    corpus = [dictionary.doc2bow(words) for words in paragraphs]

    # 训练LDA模型
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary)

    return lda_model


# 设置主题数量
num_topics = 10

# 使用朴素贝叶斯分类器
classifier = MultinomialNB()

# 使用Pipeline将向量化和分类器组合起来
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000)),
    ('classifier', classifier)
])

# 10次交叉验证
for k in max_tokens:
    print(f"Max tokens: {k}")
    paragraphs = paragraphs_all[k]
    labels = labels_all[k]

    lda_model = train_lda_model(paragraphs, num_topics=num_topics)

    # 将段落表示为主题分布
    topics_distribution = np.zeros((len(paragraphs), num_topics))
    for i, words in enumerate(paragraphs):
        bow_vector = lda_model.id2word.doc2bow(words)
        topics = lda_model[bow_vector]
        for topic in topics:
            topics_distribution[i, topic[0]] = topic[1]

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(topics_distribution, labels, test_size=0.2, random_state=42)

    # 使用10次交叉验证进行评估
    cv_scores = cross_val_score(classifier, X_train, y_train, cv=10)
    print("Cross-validation scores:", cv_scores)
    print("Mean cross-validation score:", np.mean(cv_scores))

    # 训练朴素贝叶斯分类器
    classifier.fit(X_train, y_train)

    # 计算整体训练准确度
    train_pred = classifier.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    print("Train accuracy:", train_acc)

    # 测试准确度
    test_accuracy = classifier.score(X_test, y_test)
    print("Test accuracy:", test_accuracy)
