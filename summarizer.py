import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import networkx as nx

nltk.download("stopwords")

def read_textfile(file_path):
    f = open(file_path, "r")
    data = f.readlines()
    text_data = data[0].split(". ")

    article_text = []

    for sentence in text_data:
        line = sentence.replace("[^a-zA-Z]", " ").split(" ") # text should be english
        article_text.append(line)
    
    article_text.pop() 
    
    return article_text

def write_textfile(file_name, summarized):
    f = open(file_name, "w+")

    for sentence in summarized:
        f.write(sentence + ".\n")
    
    f.close()

def get_similarity(sentence_1, sentence_2, stopwords):
    lower_sentence1, lower_sentence2 = [], []

    for word in sentence_1:
        lower_sentence1.append(word.lower())

    for word in sentence_2:
        lower_sentence2.append(word.lower())
 
    total_words = list(set(lower_sentence1 + lower_sentence2))

    N = len(total_words)
    v1, v2 = [0] * N, [0] * N
    
    # word2vec
    for word in lower_sentence1:
        if word not in stopwords:
            idx = total_words.index(word)
            v1[idx] += 1
 
    for word in lower_sentence2:
        if word not in stopwords:
            idx = total_words.index(word)
            v2[idx] += 1
    
    similarity = 1 - cosine_distance(v1, v2)
    
    return similarity


def generate_summary(file_path, k, stop_words):
    result = []

    article_text = read_textfile(file_path)

    N = len(article_text)
    similarity_matrix = np.zeros((N, N))
 
    for i in range(N):
        for j in range(i+1, N):
            similarity = get_similarity(article_text[i], article_text[j], stop_words)

            similarity_matrix[i][j], similarity_matrix[j][i] = similarity, similarity

    # Page rank
    ranks = nx.pagerank(nx.from_numpy_array(similarity_matrix))

    ranked_sentence = sorted(((ranks[i], s) for i, s in enumerate(article_text)), reverse=True)

    for i in range(k):
      result.append(" ".join(ranked_sentence[i][1]))

    write_textfile("summarized_article.txt", result)

def main():
    stop_words = stopwords.words('english')
    if stop_words is None:
        stop_words = []

    generate_summary(file_path="sample_article.txt", k=2, stop_words=stop_words)

if __name__ == "__main__":
    main()