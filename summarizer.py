import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import networkx as nx
import argparse
from rouge import Rouge

class ExtractiveSummarizer:
    def __init__(self, args):
        self.args = args
        self.reference_text = ""
        self.summarized_text = ""

        nltk.download("stopwords")
        self.stop_words = stopwords.words('english')
        if self.stop_words is None:
            self.stop_words = []

    def read_textfile(self, file):
        f = open(file, "r")
        data = f.readlines()
        text_data = data[0].split(". ")
        self.reference_text = data[0]

        article_text = []

        for sentence in text_data:
            line = sentence.replace("[^a-zA-Z]", " ").split(" ") # text should be english
            article_text.append(line)
        
        article_text.pop() 
        
        return article_text

    def write_textfile(self, summarized):
        f = open(self.args.output_file, "w+")

        for sentence in summarized:
            f.write(sentence + ".\n")
        
        f.close()

    def get_similarity(self, sentence_1, sentence_2):
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
            if word not in self.stop_words:
                idx = total_words.index(word)
                v1[idx] += 1
    
        for word in lower_sentence2:
            if word not in self.stop_words:
                idx = total_words.index(word)
                v2[idx] += 1
        
        similarity = 1 - cosine_distance(v1, v2)
        
        return similarity
    
    def calculate_k(self, N):
        k = N // 3

        if N % 3 != 0:
            k += 1
        
        return k

    def generate_summary(self):
        result = []
        article_text = self.read_textfile(self.args.input_file)

        N = len(article_text)
        similarity_matrix = np.zeros((N, N))
    
        for i in range(N):
            for j in range(i+1, N):
                similarity = self.get_similarity(article_text[i], article_text[j])

                similarity_matrix[i][j], similarity_matrix[j][i] = similarity, similarity

        # Page rank
        ranks = nx.pagerank(nx.from_numpy_array(similarity_matrix))

        ranked_sentence = sorted(((ranks[i], s) for i, s in enumerate(article_text)), reverse=True)

        if self.args.k < 1 or self.args.k > N:
            k = self.calculate_k(N)
        else:
            k = self.args.k

        for i in range(k):
            sentence = " ".join(ranked_sentence[i][1])
            result.append(sentence)
            self.summarized_text += sentence + ". "

        self.write_textfile(result)

    # Using Rouge to Evaluate abstractive Summary
    def evaluate(self):
        r = Rouge()
        scores = r.get_scores(self.summarized_text, self.reference_text)
        print(scores)
        
def main():
    parser = argparse.ArgumentParser(description='Article Summarizer')

    parser.add_argument('--k', dest="k", type=int, default=0, 
                        help='set number of output sentences')

    parser.add_argument('--i', dest="input_file", type=str, default="sample_article.txt",
                        help='article text file to summarize - default sample_article.txt')

    parser.add_argument('--o', dest="output_file", type=str, default="summarized_sample_article.txt",
                        help='summarized article text file  - default summarized_sample_article.txt')

    args = parser.parse_args()
    summarizer = ExtractiveSummarizer(args)
    summarizer.generate_summary()
    summarizer.evaluate()

if __name__ == "__main__":
    main()