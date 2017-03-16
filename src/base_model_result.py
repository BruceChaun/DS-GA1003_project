import numpy as np
from csv import reader
import model

def feature_normalize(train, test):
    """
    Rescale the data so that each feature in the training set is in
    the interval [0,1], and apply the same transformations to the test 
    set, using the statistics computed on the training set.

    Args: 
        train - training set, a 2D numpy array of size (num_instances, num_features)
        test  - test set, a 2D numpy array of size (num_instances, num_features)

    Returns: 
        train_normalized - training set after normalization 
        test_normalized  - test set after normalization
    """
    # get max and min for each column
    train_max = train.max(axis=0)
    train_min = train.min(axis=0)

    # discard column which has the constant value
    indicator = (train_max != train_min)
    train = train[:,indicator]
    test = test[:,indicator]
    train_max = train_max[indicator]
    train_min = train_min[indicator]

    # normalize
    train_normalized = (train - train_min) / (train_max - train_min)
    test_normalized = (test - train_min) / (train_max - train_min)

    return train_normalized, test_normalized

def load_data(path, tau=0.3):
    """
    load data and generate features and label from existing data attributes

    NOTE:
        This is for baseline. Currently, we choose 10 features.
        Read code comments for more details

    @param path: string, from which we read data
    @param tau: float, threshold of labeling whether a product review is 
                helpful or not, compared with the value of helpful/total.
    @return feature: 2D numpy array (num_instances, num_features)
    @return label: binary value
    """
    data_file = open(path, "rb")
    data = reader(data_file, delimiter=",", quotechar='"')

    feature = []
    label = []

    for row in data:
        try:
            attr = []
            attr.append(float(row[6])) # score
            attr.append(float(row[17])) # positive words
            attr.append(float(row[18])) # negative words
            attr.append(float(row[22])) # number of total user reviews
            attr.append(1. * float(row[23]) / float(row[21])) # review sequence %
            attr.append(float(row[24])) # score relative to average rating
            attr.append(float(row[25])) # variance of rating

            num_sent = float(row[26]) # number of sentences
            attr.append(num_sent) 
            num_word_token = float(row[27]) # number of word tokens
            attr.append(num_word_token) 
            if num_sent == 0:
                attr.append(0)
            else:
                attr.append(1. * num_word_token / num_sent) # words per sentence
            feature.append(np.array(attr))

            # set label
            helpful = float(row[8])
            if helpful == 0:
                label.append(0)
            else:
                ratio = 1. * helpful / float(row[9])
                if ratio > tau:
                    label.append(1)
                else:
                    label.append(0)
        except ValueError:
            print row

    return np.array(feature), label


X_train, y_train = load_data("../data/train.csv")
X_valid, y_valid = load_data("../data/valid.csv")
X_test, y_test = load_data("../data/test.csv")

X_train_norm, X_valid = feature_normalize(X_train, X_valid)
_, X_test = feature_normalize(X_train, X_test)


def ababoost_model():
    m = model.AdaBoost(X_train_norm, y_train)
    m.train()
    acc = m.eval(X_valid, y_valid)
    print "Accuracy = %.6f" % acc

if __name__ == "__main__":
    ababoost_model()
