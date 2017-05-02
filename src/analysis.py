from __future__ import print_function
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import re
from sklearn.ensemble import AdaBoostClassifier
import model
import numpy as np


def feature_importance(model):
    importance = np.array(model.feature_importances_)
    indices = np.argsort(importance)[::-1]

    plt.figure(figsize=(25, 15))
    plt.title("Feature Importance")
    plt.bar(range(len(indices)), importance[indices], align="center")
    plt.xticks(range(len(indices)), indices+1)
    plt.grid()
    plt.xlabel("Feature Indices")
    plt.ylabel("Importance")
    plt.savefig("feature_importance.png")
    plt.close()


def plot_train(filename):
    plots = [[], [], []]
    pattern = re.compile("acc: (\d+\.\d*)")

    with open(filename, "rb") as f:
        for line in f:
            m = pattern.findall(line)
            if m:
                plots[0].append(float(m[0]))
                plots[1].append(float(m[1]))
                plots[2].append(float(m[2]))

    epochs = len(plots[0])

    plt.figure(figsize=(25, 15))
    for i in range(len(plots)):
        plt.plot(range(1, 1+epochs), plots[i])
    plt.legend(["training", "validation", "test"], 
            loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig("acc.png")
    plt.close()


if __name__ == "__main__":
    #adaboost = model.AdaBoost(None, None)
    #adaboost.load("./model/adaboost_1_101.pkl")
    #feature_importance(adaboost.model)
    plot_train("adaboost.out")
