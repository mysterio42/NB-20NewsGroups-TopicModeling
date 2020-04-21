from sklearn.datasets import fetch_20newsgroups


def get_data(categories):
    training = fetch_20newsgroups(subset='train',
                                  categories=categories,
                                  shuffle=True)
    testing = fetch_20newsgroups(subset='test',
                                 categories=categories)
    return training.data, training.target, testing.data, testing.target


def get_whole_data(categories):
    training = fetch_20newsgroups(subset='train',
                                  categories=categories,
                                  shuffle=True)
    testing = fetch_20newsgroups(subset='test',
                                 categories=categories)

    x = []
    y = []
    x.extend(training.data)
    x.extend(testing.data)
    y.extend(training.target)
    y.extend(testing.target)
    return x, y


def get_clean_data(categories):
    training = fetch_20newsgroups(subset='train',
                                  categories=categories,
                                  remove=('headers', 'footers', 'quotes'),
                                  shuffle=True)
    testing = fetch_20newsgroups(subset='test',
                                 remove=('headers', 'footers', 'quotes'),
                                 categories=categories)

    return training.data, training.target, testing.data, testing.target
