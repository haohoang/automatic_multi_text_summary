import numpy as np
import pandas as pd
import heapq
import utils
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import FastText
import logging


class Summary():
    """
    Attribute:
        - data: list of list - article, position, raw_text
        - df: DataFrame with row: sentence, column = "article, position, raw_text"
            article: own sentence
            position of the sentence in the article
            raw_text: sentence before preprocess
        - model: word embedding
        - rel_position: True if sort sentences in final summary based on average of
                      all sentences' position in a cluster
                      Default: True
        - n_clusters: number of clusters in Kmeans
        - lsa: True if use LSA after Kmean, default=False
        - mmr: True if use MMR after LSA or Kmeans, default=False
        - random_state: default = 42
        - lamda: only used when mmr is used, default = 0.6
        - n_component: only use when lsa is True, default=20
        - n_iter: only use when lsa is True. default = 10
    """

    def __init__(self, data=None, model=FastText(), rel_position=True, n_clusters=16, lsa=False, mmr=False,
                 random_state=42, lamda=0.6, n_components=20, n_iter=10):
        self.df = pd.DataFrame(data, columns=["article", "position", "raw_text"])
        self.model = model
        self.rel_position = rel_position
        self.n_clusters = n_clusters
        self.lsa = lsa
        self.mmr = mmr
        self.random_state = 42
        self.lamda = lamda
        self.n_components = n_components
        self.n_iter = n_iter

    def clean_sent(self):
        logging.info("Start cleaning sentences")
        self.df["clean_text"] = ""
        for index, row in self.df.iterrows():
            if index % 10 == 0:
                logging.info("Cleaned " + str(index) + "sentences")
            temp = utils.clean_not_remain_tag(row["raw_text"])
            if len(temp) == 0:
                logging.info("Delete sentence" + str(index) + "because of zero length")
                self.df.drop(labels=[index], inplace=True)
            else:
                self.df.at[index, "clean_text"] = temp
        logging.info("Finish cleaning sentences")
        logging.info("Reindex ")
        self.df.index = range(len(self.df))

    def compute_vector(self, sent):
        """model: word embedding
           sent: list of words
           return vector of sent"""
        vector = np.zeros(self.model.get_dimension())
        for word in sent:
            vector += self.model.get_word_vector(word)
        return vector

    def sent2vec(self):
        logging.info("Start computing vectors")
        self.df["vector"] = None
        for index, row in self.df.iterrows():
            if index % 10 == 0:
                logging.info("Computed " + str(index) + "vector sentences")
            self.df.at[index, "vector"] = self.compute_vector(row["raw_text"])
        logging.info("Finishing compute vectors")

    def trainKmean(self):
        self.clean_sent()
        self.sent2vec()
        self.kmean = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.kmean.fit(self.df['vector'].tolist())
        self.df["label"] = self.kmean.labels_
        self.choose_centers()
        self.kmean_summary()

    def choose_centers(self):
        logging.info("Choosing center in each cluster")
        self.df['choose'] = False
        self.relative_position = list()
        for i in range(self.n_clusters):
            index = self.df['label'] == i
            center = self.kmean.cluster_centers_[i]
            minDistance = self.kmean.inertia_
            centeri = None
            for idx, row in self.df[index].iterrows():
                if np.linalg.norm(row['vector'] - center) <= minDistance:
                    centeri = idx
            self.df.at[centeri, 'choose'] = True
            if self.rel_position:
                self.relative_position.append(self.df[index]['position'].mean())
        logging.info("Finish choosing")

    def kmean_summary(self):
        logging.info("Choosing center-sentences for summary")
        index = self.df['choose'] == True
        self.summary = self.df[index].sort_values(['label'])
        if self.rel_position:
            self.summary['relative_position'] = np.array(self.relative_position)
            self.summary = self.summary.sort_values(['relative_position'])
        else:
            self.summary = self.summary.sort_values(['position'])
        self.summary.index = range(self.n_clusters)
        logging.info("Finish choosing")

    def build_vocab(self):
        logging.info("Building vocab")
        corpus = self.summary['clean_text'].tolist()
        for idx, sent in enumerate(corpus):
            corpus[idx] = " ".join(sent)
        return corpus

    def LSA(self):
        logging.info("Start LSA")
        corpus = self.build_vocab()
        logging.info("Compute tf-idf")
        vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True)
        X = vectorizer.fit_transform(corpus)
        svd = TruncatedSVD(n_components=self.n_components, n_iter=self.n_iter, random_state=self.random_state)
        Y = svd.fit_transform(X)
        A = Y.mean(axis=1)
        index = heapq.nlargest(self.n_components, range(len(A)), A.take)
        logging.info("Finish LSA")
        return index

    def MMR(self):
        logging.info("Start MMR")
        if self.lsa:
            indices = self.LSA()
            self.summary = self.summary.iloc[indices]
            self.summary.index = range(self.n_components)
        self.summary['choose'] = 0

        # choose first sentence for final summary
        S = self.summary[['position']].idxmin()
        first_sent = S.get(key='position')
        logging.info("First sentence'position in summary is " + str(first_sent))
        self.summary.at[first_sent, 'choose'] = 1

        # compute tf-idf
        corpus = self.build_vocab()
        vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True)
        X = vectorizer.fit_transform(corpus)
        logging.info("Vocabulary include" + str(vectorizer.idf_.shape) + "words")
        self.idf = pd.DataFrame(vectorizer.idf_, index=vectorizer.get_feature_names())
        count = CountVectorizer()
        X = count.fit_transform(corpus)
        self.tf = pd.DataFrame(X.toarray(), columns=count.get_feature_names())

        logging.info("Ranking")
        # ranking
        for i in range(15):
            max_sim = -100000000
            max_idx = -1
            S1 = self.summary['choose'] == 0
            S2 = self.summary['choose'] == 2
            for idx_v, row in self.summary[S1].iterrows():
                sim1 = self.sim(first_sent, idx_v)
                max_sim2 = 0
                if i > 0:
                    for idx_u, row_u in self.summary[S2].iterrows():
                        sim2 = self.sim(idx_u, idx_v)
                        max_sim2 = max(max_sim2, sim2)
                simV = self.lamda * (sim1 - (1 - self.lamda) * max_sim2)
                if simV >= max_sim:
                    max_sim = simV
                    max_idx = idx_v
            self.summary.at[max_idx, 'choose'] = 2

        self.summary.at[first_sent, 'choose'] = 2
        index = self.summary['choose'] == 2
        self.summary = self.summary[index]
        logging.info("Finish MMR")

    def sim(self, idx_u, idx_v):
        """ idx_u: index of sentence u
            idx_v: index of sentence v
        Compute similarity between sentence u and sentence v"""
        tu_so = 0
        mau_so = 0
        for word in self.summary.at[idx_v, 'clean_text']:
            tu_so += self.tf.at[idx_u, word] * self.tf.at[idx_v, word] * (self.idf.at[word, 0] ** 2)
        for word in self.summary.at[idx_u, 'clean_text']:
            mau_so += (self.tf.at[idx_u, word] * self.idf.at[word, 0]) ** 2
        mau_so = mau_so ** 0.5
        return tu_so / mau_so

    def save_summary(self, filename):
        tom_tat = self.summary['raw_text'].tolist()
        with open(filename, 'w') as f:
            for sent in tom_tat:
                f.write(sent + "\n")

    def summary(self, filename):
        self.trainKmean()
        if self.mmr:
            self.MMR()
        self.save_summary(filename)


