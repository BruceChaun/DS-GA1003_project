from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle
import numpy as np


class Model(object):
    """
    Base model class, define some common functions
    """
    def __init__(self, X, y):
        """
        class constructor, set member variables

        Usage: model = Model(X, y)
        """
        self.X = X # training data, numpy type
        self.y = y # training label, numpy type
        self.labels = np.unique(y)

    def train(self):
        """
        model training

        Use self.X and self.y to supervised learning

        Usage: model.train()
        """
        raise NotImplementedError("train() method in class Model must be overriden")

    def eval(self, X, y):
        """
        Evaluate performance (e.g. accuracy) on data set (X, y)

        Usage: acc = model.eval(X, y)
        """
        raise NotImplementedError("eval() method in class Model must be overriden")

    def save(self, path):
        """
        save the trained model in disk

        Usage: model.save("path/to/save")

        @param path: string, indicating the file to save the model
        """
        pickle.dump(self.model, open(path, "wb"))

    def load(self, path):
        """
        load a saved model from disk

        Usage: trained_model = model.load("path/to/load")

        @param path: string, indicating the file to load the model
        """
        self.model = pickle.load(open(path, "rb"))
        return self.model


class AdaBoost(Model):
    """
    AdaBoost classifier
    """
    def __init__(self, X, y, T=100):
        super(AdaBoost, self).__init__(X, y)
        self.T = T

    def train(self):
        self.model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4), \
                algorithm="SAMME", n_estimators=self.T)
        self.model.fit(self.X, self.y)
        
    def eval(self, X, y):
        """
        return a confusion matrix
        """
        pred = self.model.predict(X)
        n = len(pred)
        
        TP = TN = FP = FN = 0

        for i in range(n):
            if y[i] == self.labels[1]:
                if pred[i] == self.labels[1]:
                    TP += 1
                else:
                    FN += 1
            elif pred[i] == self.labels[0]:
                if pred[i] == self.labels[0]:
                    TN += 1
                else:
                    FP += 1

        confusion_matrix = np.array([TP, FN, FP, TN]).reshape([2,2])

        return confusion_matrix


