import argparse

from utils.data import get_data, get_whole_data
from utils.embedding import FastEmbedding
from utils.embedding import dump_embedding, load_embedding, latest_modified_embedding
from utils.model import load_model, latest_modified_weight
from utils.model import train_model, predict_model, find_optimal_alpha
from utils.plot import plot_pca, plot_tsne, plot_optimal_alpha


def parse_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected')

    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str2bool, default=True,
                        help='True: Load trained model  '
                             'False: Train model default: False')

    parser.print_help()

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    categories.sort()

    if args.load:

        embedding = load_embedding(latest_modified_embedding())
        model = load_model(latest_modified_weight())

        sentence = ['Software engineering is getting hotter and hotter nowadays']
        embed_sentence = embedding.embed_unseen(sentence)

        ans = predict_model(model, embed_sentence, categories)

        print(f'{sentence[0]} - {ans}')
    else:

        features_train, labels_train, features_test, labels_test = get_data(categories)

        X, y = get_whole_data(categories)
        embedding_alpa = FastEmbedding(docs=X, norm='l2')
        alpha_range, scores, opt_alpha, opt_score = find_optimal_alpha(embedding_alpa.tfidf, y)
        plot_optimal_alpha(alpha_range, scores, opt_alpha, opt_score)
        del X, y

        fast_emb = FastEmbedding(docs=features_train, norm='l2')

        embed_features_test = fast_emb.embed_unseen(features_test)
        dump_embedding(fast_emb)

        plot_pca(fast_emb.tfidf.toarray(), labels_train, 'train')
        plot_pca(embed_features_test.toarray(), labels_test, 'test')
        plot_tsne(fast_emb.tfidf.toarray(), labels_train, 'train')
        plot_tsne(embed_features_test.toarray(), labels_test, 'test')

        model = train_model(opt_alpha, fast_emb.tfidf, labels_train, embed_features_test, labels_test)

        sentence = ['Software engineering is getting hotter and hotter nowadays']
        embed_sentence = fast_emb.embed_unseen(sentence)
        ans = predict_model(model, embed_sentence, categories)
        print(f'{sentence[0]} - {ans}')
