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

X_train, y_train = utils.load_pca_csv_data(folder + "train.csv")
X_valid, y_valid = utils.load_pca_csv_data(folder + "validate.csv")
X_test, y_test = utils.load_pca_csv_data(folder + "test.csv")


def run_epoch(epoch):
    m = model.AdaBoost(X_train, y_train, epoch)
    m.train()

    confusion_matrix_train = m.eval(X_train, y_train)
    confusion_matrix_valid = m.eval(X_valid, y_valid)
    confusion_matrix_test = m.eval(X_test, y_test)
    
    train_acc = utils.get_accuracy_from_confusion_matrix(confusion_matrix_train)
    valid_acc = utils.get_accuracy_from_confusion_matrix(confusion_matrix_valid)
    test_acc = utils.get_accuracy_from_confusion_matrix(confusion_matrix_test)

    return train_acc, valid_acc, test_acc, m


def run_ababoost_model():
    n_components = 2
    begin = 301
    end = begin + 100
    epochs = list(range(begin, end))

    train_history = []
    valid_history = []
    test_history = []
    best_model = None
    best_valid_acc = 0

    for epoch in epochs:
        train_acc, valid_acc, test_acc, model = run_epoch(epoch)

        train_history.append(train_acc)
        valid_history.append(valid_acc)
        test_history.append(test_acc)

        if epoch > 1 and valid_acc > best_valid_acc:
            best_model = model
            best_valid_acc = valid_acc

    model_name = model_path + "adaboost_{}_{}.pkl"
    pickle.dump(best_model, open(model_name.format(begin, end), "wb"))

    train_hist_name = model_path + "train_{}_{}.txt".format(begin, end)
    np.savetxt(train_hist_name, train_history, delimiter=",", fmt="%1.6f")

    valid_hist_name = model_path + "valid_{}_{}.txt".format(begin, end)
    np.savetxt(valid_hist_name, valid_history, delimiter=",", fmt="%1.6f")


if __name__ == "__main__":
    run_ababoost_model()
