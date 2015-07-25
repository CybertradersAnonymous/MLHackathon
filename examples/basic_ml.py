# Example from:
# http://en.wikipedia.org/wiki/Naive_Bayes_classifier
# Naive Bayes, decision tree, logistic regression, svm, knn, k-means
# the colums of v are arranged in gender, heigth (foot), weight, foot size

from numpy import *
from pylab import plot, show

v=array([[0, 6, 180,    12],
[0,     5.92,   190,    11],
[0,     5.58,   170,    12],
[0,     5.92,   165,    10],
[1,     5,      100,    6],
[1      , 5.5,  150,    8],
[1, 5.42,       130,    7],
[1,     5.75,   150,    9]])

data = v[:,1:]
classifier = v[:,0]
sample = [5.75, 160,    8]

def print_result(func, sample):
    if func.predict(sample):
        print 'The person is FEMALE','says: %s'%func.__class__
        return 1
    else:
        print 'The person is MALE','says: %s'%func.__class__
        return 0


def normalize(data, sample):
    normalized_data = array([(i-min(i))/(max(i)-min(i)) for i in data.T]).T
    minmax = array([[min(i),max(i)] for i in data.T])
    normalized_sample = [(sample[i]-minmax[i][0])/(minmax[i][1]-minmax[i][0])  for i in range(len(sample))]
    return normalized_data, normalized_sample

def naive_bayes(data,classifier,sample):
    from scipy.stats import norm
    idx_male = array([i for i in range(len(classifier)) if classifier[i]==0])
    idx_female = array([i for i in range(len(classifier)) if classifier[i]==1])
    mean_male = mean(data[idx_male,:],0)
    std_male = std(data[idx_male,:],0)
    mean_female = mean(data[idx_female,:],0)
    std_female = std(data[idx_female,:],0)

    probs_female = []
    for i in range(len(mean_female)):
        probs_female.append( norm.pdf(sample[i],mean_female[i],std_female[i]))

    probs_male = []
    for i in range(len(mean_male)):
        probs_male.append( norm.pdf(sample[i],mean_male[i],std_male[i]))

    p_male = cumprod(probs_male)[-1] * 0.5
    p_female = cumprod(probs_female)[-1] * 0.5
    if p_male > p_female:
        print 'The person is MALE',     'says: Naive Bayes'
        return 0
    else:
        print 'The person is FEMALE','says: : Naive Bayes'
        return 1


def decision_tree(data,classifier,sample):
    from sklearn import tree
    clf = tree.DecisionTreeRegressor()
    clf.fit(data,classifier)
    print_result(clf,sample)

def k_nearest_neighbors(data,classifier,sample):
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=2)
    data,sample = normalize(data,sample)
    clf.fit(data, classifier)
    print_result(clf,sample)

def logistic_regression(data,classifier,sample):
    from sklearn.linear_model import LogisticRegression
    clf =  LogisticRegression(penalty='l2')
    clf.fit(data,classifier)
    print_result(clf,sample)

def support_vector_machine(data,classifier,sample):
    from sklearn import svm
    clf = svm.SVC()
    data,sample = normalize(data,sample)
    clf.fit(data,classifier)
    print_result(clf,sample)

def k_means(data,classifier,sample):
    from sklearn.cluster import KMeans
    clf = KMeans(n_clusters=2)
    data,sample = normalize(data,sample)
    clf.fit(data,classifier)
    print_result(clf,sample)

def naive_bayes_2(data,classifier,sample):
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(data, classifier)
    print_result(clf,sample)

def ada_boost(data,classifier,sample):
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.cluster import KMeans
    from sklearn.naive_bayes import GaussianNB
    func = GaussianNB()
    func = DecisionTreeRegressor()
    func = KMeans(n_clusters=2)
    clf = AdaBoostRegressor(func,n_estimators=300,random_state=random.RandomState(1))
    clf.fit(data,classifier)
    print_result(clf,[sample])


if __name__=='__main__':

    naive_bayes(data,classifier,sample)
    decision_tree(data,classifier,sample)
    k_nearest_neighbors(data,classifier,sample)
    logistic_regression(data,classifier,sample)
    support_vector_machine(data,classifier,sample)
    k_means(data,classifier,sample)
    naive_bayes_2(data,classifier,sample)
    ada_boost(data,classifier,sample)

