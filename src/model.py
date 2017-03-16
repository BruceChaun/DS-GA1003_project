from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle


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

    def model(self):
        """
        retrieve the trained model

        Usage: trained_model = model.model()
        """
        return self.model

    def save(self, path):
        """
        save the trained model in disk

        Usage: model.save("path/to/save")

        @param path: string, indicating the file to save the model
        """
        pickle.dump(self.model(), open(path, "wb"))

    def load(self, path):
        """
        load a saved model from disk

        Usage: model.load("path/to/load")

        @param path: string, indicating the file to load the model
        """
        self.model = pickle.load(open(path, "rb"))


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
        pred = self.model.predict(X)
        n = len(pred)
        acc = 0

        for i in range(n):
            if pred[i] == y[i]:
                acc += 1

        return 1. * acc / n


