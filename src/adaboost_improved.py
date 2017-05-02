from __future__ import print_function
import numpy as np
import model
import utils
import pickle
import os


folder = "../data/pca/"
model_path = "model/"

if not os.path.exists(model_path):
    os.mkdir(model_path)

tau = 0.8
X_train, y_train = utils.load_pca_csv_data(folder + "train.csv", tau)
X_valid, y_valid = utils.load_pca_csv_data(folder + "validate.csv", tau)
X_test, y_test = utils.load_pca_csv_data(folder + "test.csv", tau)


def run_epoch(epoch):
    m = model.AdaBoost(X_train, y_train, epoch)
    m.train()

    confusion_matrix_train = m.eval(X_train, y_train)
    confusion_matrix_valid = m.eval(X_valid, y_valid)
    confusion_matrix_test = m.eval(X_test, y_test)
    print(confusion_matrix_valid)
    
    train_acc, train_f1 = utils.get_score_from_confusion_matrix(confusion_matrix_train)
    valid_acc, valid_f1 = utils.get_score_from_confusion_matrix(confusion_matrix_valid)
    test_acc, test_f1 = utils.get_score_from_confusion_matrix(confusion_matrix_test)

    return train_acc, train_f1, valid_acc, valid_f1, test_acc, test_f1, m


def run_ababoost_model():
    n_components = 2
    begin = 1
    end = begin + 100
    epochs = list(range(begin, end))

    best_model = None
    best_valid_acc = 0

    for epoch in epochs:
        vals = run_epoch(epoch)

        print("Epoch {:3} | training acc: {:5.4f} | training f1: {:5.4f} | \
                valid acc: {:5.4f} | test f1: {:5.4f} | \
                test acc: {:5.4f} | test f1: {:5.4}".format(epoch, 
                    vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]))

        if epoch > 1 and vals[2] > best_valid_acc:
            best_model = vals[6]
            best_valid_acc = vals[2]

    model_name = model_path + "adaboost_{}_{}.pkl"
    best_model.save(model_name.format(begin, end))


if __name__ == "__main__":
    run_ababoost_model()
