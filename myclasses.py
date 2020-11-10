from typing import List
from math import log

class TfidfVectorizer ():
    def __init__(self):
        self.features = []
    
    def get_feature_names(self) -> List[str]:
        """Returns list of features names"""
        return(self.features)
    
    def count_matrix(self, corpus: List[str]) -> List[List[int]]:
        """'Learn the vocabulary dictionary and return document-term matrix"""
        features = {}

        # creating features dict
        for s in corpus:
            for w in s.lower().split(' '):
                if w not in features:
                    features[w] = len(features)

        # filling count_matrix
        count_matrix = []

        for i, s in enumerate(corpus):
            count_matrix.append([0] * len(features))
            for w in s.lower().split(' '):
                count_matrix[i][features[w]] += 1

        self.features = [x[0] for x in sorted(list(features.items()), key=lambda x: x[1])]

        return count_matrix
    
    def tf_transform(self, count_matrix : List[List[int]]) -> List[List[float]]:
        tf_matrix = []
        for l in count_matrix:
            n = sum(l)
            tf_matrix.append([round(x / n, 3) for x in l])
        return tf_matrix
    
    def idf_transform(self, count_matrix: List[List[int]]) -> List[float]:
        idf_matrix = []
        n_docs = len(count_matrix)
        n_features = len(count_matrix[0])
        words = list(map(list, zip(*count_matrix)))
        idf_matrix = [ round(log((n_docs + 1) / (sum(map(lambda x: int(x > 0), words[i])) + 1)) + 1,1)  for i in range(n_features)]
        return idf_matrix
    
    def fit_transform(self, corpus: List[str]) -> List[List[float]]:
        count_matrix = self.count_matrix(corpus)
        tf_matrix = self.tf_transform(count_matrix)
        idf = self.idf_transform(count_matrix)
        tfidf_matrix = [[round(x*y,3) for x,y in list(zip(*[tf,idf]))] for tf in tf_matrix]
        return tfidf_matrix