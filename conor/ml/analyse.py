import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm, tree, preprocessing

import loaders


def load_data_nick():
    raw_data = loaders.load_train_nick()
    classifier = raw_data['Target_Return'].values
    data = raw_data.drop('Target_Return')
    return classifier, data


def load_data_tom():
    train_data = loaders.load_train_tom('../../data/train.csv')
    test_data = loaders.load_train_tom('../../data/test.csv')

    train_returns = calculate_returns(train_data['AdjPrice'].values)
    train_data['Returns'] = pd.Series(train_returns, index=train_data.index)

    test_returns = calculate_returns(test_data['AdjPrice'].values)
    test_data['Returns'] = pd.Series(test_returns, index=test_data.index)

    train_classifier = train_data[['Returns', 'Company']][1:]
    train_dataset = train_data.drop(['Sector', 'Industry', 'AdjPrice'], axis=1)[1:]

    test_classifier = test_data[['Returns', 'Company']][1:]
    test_dataset = test_data.drop(['Sector', 'Industry', 'AdjPrice'], axis=1)[1:]

    return train_classifier, train_dataset, test_classifier, test_dataset


def main():
    train_classifier, train_data, test_classifier, test_data = load_data_tom()

    # companies = set(train_data['Company'].values)
    companies = ['Alcoa Inc']

    filter = lambda df, company: df.loc[df['Company'] == company, :].drop('Company', 1).values

    results = np.empty(len(companies))

    for idx, company in enumerate(companies):
        c_train_data = filter(train_data, company)
        c_train_classifier = filter(train_classifier, company)
        c_train_data_scaled = preprocessing.scale(c_train_data)
        c_train_classifier_scaled = preprocessing.scale(c_train_classifier)

        c_test_data = filter(test_data, company)
        c_test_classifier = filter(test_classifier, company)
        c_test_data_scaled = preprocessing.scale(c_test_data)
        c_test_classifier_scaled = preprocessing.scale(c_test_classifier)

        results[idx] = analyse(svm.SVR(), c_train_data_scaled, c_test_data_scaled,
                               c_train_classifier_scaled, c_test_classifier_scaled)

    print('Mean sum squared: {}'.format(results.mean()))

    # naive_bayes(data, classifier)
    # decision_tree(data, classifier)
    # k_nearest_neighbors(data, classifier)
    # logistic_regression(data, classifier)

    # k_means(data, classifier)
    # naive_bayes_2(data, classifier)
    # ada_boost(data, classifier)


def lag(data, empty_term=0.):
    lagged = np.roll(data, 1, axis=0)
    lagged[0] = empty_term
    return lagged


def calculate_returns(prices):
    lagged_pnl = lag(prices)
    returns = (prices - lagged_pnl) / lagged_pnl

    # All values prior to our position opening in pnl will have a
    # value of inf. This is due to division by 0.0
    returns[np.isinf(returns)] = 0.
    # Additionally, any values of 0 / 0 will produce NaN
    returns[np.isnan(returns)] = 0.
    return returns


def analyse(func, train_data, test_data, classifier, actual):
    func.fit(train_data, classifier[:, 0])

    predictions = func.predict(test_data)

    plt.scatter(predictions, actual[:, 0])
    plt.show()

    sq_error = ((actual[:, 0] - predictions)**2).mean()

    return sq_error


if __name__ == '__main__':
    main()
