import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from gensim import corpora, models
from nltk.tokenize import word_tokenize
import numpy as np
import random
import os
import nltk

nltk.download('punkt')


# Define a function to extract paragraphs
def extract_paragraphs(corpus_dir, num_paragraphs, max_tokens):
    paragraphs = []
    labels = []

    # Iterate through each txt file in the corpus
    for novel_file in os.listdir(corpus_dir):
        if novel_file.endswith('.txt'):
            novel_path = os.path.join(corpus_dir, novel_file)

            # Read the content of the novel
            with open(novel_path, 'r', encoding='gbk', errors='ignore') as file:
                novel_text = file.read()

            # Tokenize the novel into paragraphs
            novel_paragraphs = novel_text.split('\n')

            # Shuffle and randomly select a certain number of paragraphs
            random.shuffle(novel_paragraphs)
            for paragraph in novel_paragraphs:
                # If the paragraph length does not exceed max_tokens, add it to the dataset
                if len(paragraph.split()) <= max_tokens:
                    paragraphs.append(paragraph)
                    labels.append(novel_file[:-4])  # Use the novel filename as the label
                    if len(paragraphs) == num_paragraphs:
                        break

    return paragraphs, labels


# Corpus directory
corpus_dir = r'C:\Users\HP\PycharmProjects\pythonProject5\data'

# Parameters
num_paragraphs = 1000
max_tokens = [20, 100, 500, 1000, 3000]


# Define the LDA model
def train_lda_model(paragraphs, num_topics):
    # Create a dictionary and a corpus
    dictionary = corpora.Dictionary([word_tokenize(paragraph) for paragraph in paragraphs])
    corpus = [dictionary.doc2bow(word_tokenize(paragraph)) for paragraph in paragraphs]

    # Train the LDA model
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary)

    return lda_model


# Number of topics
num_topics = 10

# 10-fold cross-validation
for k in max_tokens:
    print(f"Max tokens: {k}")
    paragraphs, labels = extract_paragraphs(corpus_dir, num_paragraphs, k)

    # Labels from the extract_paragraphs function
    all_labels_set = set(labels)

    lda_model = train_lda_model(paragraphs, num_topics=num_topics)

    # Represent paragraphs as topic distributions
    topics_distribution = np.zeros((len(paragraphs), num_topics))
    for i, paragraph in enumerate(paragraphs):
        bow_vector = lda_model.id2word.doc2bow(word_tokenize(paragraph))
        topics = lda_model[bow_vector]
        for topic in topics:
            topics_distribution[i, topic[0]] = topic[1]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(topics_distribution, labels, test_size=0.2, random_state=42)

    # Build a random forest classifier model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Calculate training accuracy
    train_accuracy = model.score(X_train, y_train)
    print("Train accuracy:", train_accuracy)

    # Calculate testing accuracy
    test_accuracy = model.score(X_test, y_test)
    print("Test accuracy:", test_accuracy)
