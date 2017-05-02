from __future__ import print_function
import numpy as np
from csv import reader
from sklearn.manifold import TSNE

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

def load_pca_csv_data(path, tau=0.5, skip_header=True):
    """
    load data and generate features and label from existing data attributes

    NOTE:
        This is for baseline. Currently, we choose 10 features.
        Read code comments for more details

    @param path: string, from which we read data
    @param tau: float, threshold of labeling whether a product review is 
                helpful or not, compared with the value of helpful/total.
    @param skip_header: bool, True if you want to skip the header in the 
                first row, False otherwise.
    @return feature: 2D numpy array (num_instances, num_features)
    @return label: binary value
    """
    data_file = open(path, "rb")
    data = reader(data_file, delimiter=",", quotechar='"')

    if skip_header:
        next(data)

    feature = []
    label = []

    for row in data:
        try:
            attr = row[-46:] # hard coded, last 46 cols are pca-ed data
            for i in range(len(attr)):
                attr[i] = float(attr[i])
            feature.append(np.array(attr))

            # set label
            helpful = float(row[2])
            if helpful == 0:
                label.append(0)
            else:
                ratio = 1. * helpful / float(row[3])
                if ratio > tau:
                    label.append(1)
                else:
                    label.append(0)
        except ValueError:
            print(row)

    return np.array(feature), np.array(label)


def get_score_from_confusion_matrix(confusion_matrix):
    """
    @param confusion_matrix: a numpy array, 2*2
    @return accuracy, f1_score
    """
    _sum = np.sum(confusion_matrix)

    accuracy = 0.0
    for i in range(confusion_matrix.shape[0]):
        accuracy += 1. * confusion_matrix[i,i] / _sum

    precision = 1. * confusion_matrix[0,0] / np.sum(confusion_matrix[:,0])
    recall = 1. * confusion_matrix[0,0] / np.sum(confusion_matrix[0,:])
    f1_score = 2. * precision * recall / (precision + recall)

    return accuracy, f1_score


def tnse_reduction(data, n_components):
    """
    Manifold learning

    reference:

        http://scikit-learn.org/stable/modules/manifold.html

    @param data: numpy array, (num_samples, num_features)
    @param n_components: int, dimension of embedded space
    @return tnsed_data: numpy array, (num_samples, num_features)
    """
    tsne = TSNE(n_components=n_components, random_state=0)
    tsned_data = tsne.fit_transform(data)
    return tsned_data


