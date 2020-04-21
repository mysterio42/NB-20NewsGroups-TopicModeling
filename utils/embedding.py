import glob
import os

import joblib
import pandas as pd
from numpy import linalg
from sklearn.feature_extraction.text import \
    (CountVectorizer, TfidfTransformer, TfidfVectorizer)

from utils.model import generate_model_name

EMBEDDING_DIR = 'embeddings/'


def normalize_l2(vec):
    vec_l2norm = linalg.norm(vec, 2)
    return vec / vec_l2norm


def latest_modified_embedding():
    """
    returns latest trained weight
    :return: model weight trained the last time
    """
    embedding_files = glob.glob(EMBEDDING_DIR + '*')
    latest = max(embedding_files, key=os.path.getctime)
    return latest


def dump_embedding(embedding):
    path = EMBEDDING_DIR + 'tf-idf-' + generate_model_name(5) + '.pkl'

    with open(path, 'wb') as f:
        joblib.dump(value=embedding, filename=f, compress=3)
        print(f'Embedding saved at {path}')


def load_embedding(path):
    with open(path, 'rb') as f:
        return joblib.load(filename=f)


class Embedding:
    """


    Examples
    --------
    >>> embedding = Embedding(docs=docs,norm='l2')
    >>>
    >>>  print(embedding.tfidf.toarray()[0])
    >>>
    """

    def __init__(self, docs, norm=None):

        self.data = docs
        self.norm = norm

        self.cv = CountVectorizer(analyzer='word', ngram_range=(1, 1))
        self.tr = TfidfTransformer(norm=self.norm)

        self._embed()

    def _embed(self):
        self.tf = self.cv.fit_transform(self.data)
        self.tfidf = self.tr.fit_transform(self.tf)

    def show_embedding(self):
        return list(zip(self.data, self.tfidf.toarray()))

    def embed_unseen(self, data):
        tf = self.cv.transform(data)
        return self.tr.transform(tf)

    def to_csv(self, path,markdown=False):

        self.path = path

        feature_extraction = {}

        feature_extraction['word'] = self.cv.get_feature_names()

        for i, el in enumerate(self.tf.toarray()):
            feature_extraction['tf_sen_' + str(i)] = el

        feature_extraction['idf'] = self.tr.idf_

        for i, el in enumerate(self.tfidf.toarray()):
            feature_extraction['tf_idf_sen_' + str(i)] = el

        df = pd.DataFrame(data=feature_extraction)

        del feature_extraction
        if markdown:
            with open(path + '.txt', 'w') as f:
                f.write(df.to_markdown())
        else:
            df.to_csv(path + '.csv', index=False)

    def to_l2(self):
        if not self.norm:
            df = pd.read_csv(self.path + '.csv')
            vec = df['tf_idf_sen_1'].to_numpy()
            print(vec)
            vec = normalize_l2(vec)
            print(vec)


class FastEmbedding:
    """


    Examples
    --------
    >>> fast_emb = FastEmbedding(docs=docs)
    >>> print(fast_emb.tfidf.toarray()[0])
    """

    def __init__(self, docs, norm=None):
        self.to_tfidf_vec = TfidfVectorizer(norm=norm)
        self.data = docs
        self._embed()

    def _embed(self):
        self.tfidf = self.to_tfidf_vec.fit_transform(self.data)

    def embed_unseen(self, data):
        return self.to_tfidf_vec.transform(data)
