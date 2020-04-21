from utils.data import get_clean_data
from utils.embedding import Embedding

DATA_DIR = 'data/'
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
categories.sort()

features_train, labels_train, features_test, labels_test = get_clean_data(categories)
features_train, labels_train, features_test, labels_test = features_train[:3], labels_train[:3], features_test[
                                                                                                 :3], labels_test[:3]
emb = Embedding(docs=features_train, norm='l2')

embed_features_test = emb.embed_unseen(features_test)
emb.to_csv(DATA_DIR + 'NewsGroups-partial-train-embedding')
emb.to_csv(DATA_DIR + 'NewsGroups-partial-train-embedding', markdown=True)
