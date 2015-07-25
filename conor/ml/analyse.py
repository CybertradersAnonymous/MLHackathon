import numpy as np
from sklearn import preprocessing

import loaders


def load_data_nick():
    raw_data = loaders.load_train_nick()
    classifier = raw_data['Target_Return'].values
    data = raw_data.drop('Target_Return')
    return classifier, data


def load_data_tom():
    raw_data = loaders.load_train_tom()
    classifier = raw_data[['AdjPrice', 'Company']]
    data = raw_data.drop(['Sector', 'Industry'], axis=1)
    return classifier, data


def main():
    classifier, data = load_data_tom()

    companies = set(data['Company'].values)

    filter = lambda df, company: df.loc[data['Company'] == company, :].drop('Company', 1).values

    results = np.empty(len(companies))

    for idx, company in enumerate(companies):
        c_data = filter(data, company)
        c_classifier = filter(classifier, company)
        c_data_scaled = preprocessing.scale(c_data)
        c_classifier_scaled = preprocessing.scale(c_classifier)

        results[idx] = support_vector_machine(c_data_scaled, c_classifier_scaled)

    print('Mean sum squared: {}'.format(results.mean()))

    # naive_bayes(data, classifier)
    # decision_tree(data, classifier)
    # k_nearest_neighbors(data, classifier)
    # logistic_regression(data, classifier)

    # k_means(data, classifier)
    # naive_bayes_2(data, classifier)
    # ada_boost(data, classifier)


def normalize(data, sample):
    normalized_data = np.array([(i-min(i))/(max(i)-min(i)) for i in data.T]).T
    minmax = np.array([[min(i),max(i)] for i in data.T])
    normalized_sample = [(sample[i]-minmax[i][0])/(minmax[i][1]-minmax[i][0])  for i in range(len(sample))]
    return normalized_data, normalized_sample


def support_vector_machine(data, classifier):
    from sklearn import svm
    clf = svm.SVR()
    clf.fit(data, classifier[:, 0])

    predictions = clf.predict(data)

    sq_error = ((classifier[:, 0] - predictions)**2).mean()
    return sq_error


if __name__ == '__main__':
    main()
