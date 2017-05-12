'''
Created on May 2, 2017
@author: Fanglin Chen
This module fits the logistic regression and the SVM model.
'''

import numpy as np
import pandas as p
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier

path = '/Users/chenfanglin/fc1315/ml_project/DS-GA1003_project/'

def dummy(data, column, threshold):
    """
    Transform a continuous variable into a dummy variable.
    
    Args:
        merge_data - merge_data set, a 2D numpy array of shape (num_instances, num_features)
        column - variable to be transformed, a column in the DataFrame
        threshold - number to be compared with variable values, a number
    Returns:
        a 1D numpy array of shape (num_instances) 
    """
    return (data[column] >= threshold)

def split_X_y(data):
    """
    Split the merge_data into raw features, PCA features, and output.
    
    Args:
        merge_data - merge_data set, a DataFrame of shape (num_instances, num_features)
    Returns:
        X_raw - raw features, a 2D numpy array of shape (num_instances, num_raw_features)
        X_pca - PCA features, a 2D numpy array of shape (num_instances, num_pca_features)
        y - output, a 1D numpy array of shape (num_instances)
    """
    data = p.read_csv(path + data + '.csv')
    data['helpful'] = data['helpful']/data['total']   #Percentage of helpfulness votes
    y = data.apply(dummy, axis=1, args=('helpful',0.8))   #The output is 1 if helpfulness percentage >= 0.8 and 0 otherwise
    X = data.drop(['helpful', 'sub_cat1', 'voted'], axis=1)   #Drop non-numerical features
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.mean(axis=0), inplace=True)
    X_raw = X.ix[:,:83]   #The first 82 columns are raw features
    X_pca = X.ix[:,83:]   #The last 46 columns are PCA features
    return X_raw, X_pca, y

def feature_normalization(train, validate, test):
    """
    Rescale the merge_data so that each feature in the "train" set is in the interval [0,1], 
    and apply the same transformations to the "validate" and "test" set, 
    using the statistics computed on the "train" set.

    Args:
        train - train set, a DataFrame of shape (num_instances, num_features)
        validate - validate set, a DataFrame of shape (num_instances, num_features)
        test  - test set, a DataFrame of shape (num_instances, num_features)
    Returns:
        train_normalized - train set after normalization
        validate_normalized - validate set after normalization
        test_normalized  - test set after normalization
    """
    feature_min = train.min(axis=0)
    feature_max = train.max(axis=0)
    train_normalized = (train - feature_min)/(feature_max - feature_min) #Shift and rescale each feature
    validate_normalized = (validate - feature_min)/(feature_max - feature_min) #Apply the same transformations to validation set
    test_normalized = (test - feature_min)/(feature_max - feature_min) #Apply the same transformations to test set
    return train_normalized, validate_normalized, test_normalized

### Prepare the merge_data sets
X_train_raw, X_train_pca, y_train = split_X_y('train')
X_validate_raw, X_validate_pca, y_validate = split_X_y('validate')
X_test_raw, X_test_pca, y_test = split_X_y('test')
X_train_raw, X_validate_raw, X_test_raw = feature_normalization(X_train_raw, X_validate_raw, X_test_raw)   #Normalize the raw features


### Compute the training, validation and test accuracy at each value of alpha (regularizaion parameter) and l1_ratio
### SVM with PCA features
alpha_values = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
len = alpha_values.shape[0]
train_acc = np.zeros(len)
validate_acc = np.zeros(len)
test_acc = np.zeros(len) #Initialize accuracy
path1 = '/Users/chenfanglin/Dropbox/Spring_2017/Machine_learning/Project/poster/'

### l1 regularization
for i in range(len):
    alpha = alpha_values[i]
    clf = SGDClassifier(loss='hinge', penalty='l1', alpha=alpha)
    clf.fit(X_train_pca, y_train)
    train_acc[i] = clf.score(X_train_pca, y_train)
    validate_acc[i] = clf.score(X_validate_pca, y_validate)
    test_acc[i] = clf.score(X_test_pca, y_test)

fig = plt.figure(figsize=(15, 9))
plt_train, = plt.plot(np.log10(alpha_values), train_acc, 'b-', label='training')
plt_validate, = plt.plot(np.log10(alpha_values), validate_acc, 'g-', label='validation')
plt_test, = plt.plot(np.log10(alpha_values), test_acc, 'r-', label='test')
plt.ylabel('Accuracy')
plt.xlabel('Regularization log10(lambda)')
plt.legend(handles=[plt_train, plt_validate, plt_test], bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0., fontsize='x-small')
plt.savefig(path1 + 'SVM_l1_PCA' + '.png')
plt.clf()

### l2 regularization
for i in range(len):
    alpha = alpha_values[i]
    clf = SGDClassifier(loss='hinge', penalty='l2', alpha=alpha)
    clf.fit(X_train_pca, y_train)
    train_acc[i] = clf.score(X_train_pca, y_train)
    validate_acc[i] = clf.score(X_validate_pca, y_validate)
    test_acc[i] = clf.score(X_test_pca, y_test)

fig = plt.figure(figsize=(15, 9))
plt_train, = plt.plot(np.log10(alpha_values), train_acc, 'g-', label='train')
plt_validate, = plt.plot(np.log10(alpha_values), validate_acc, 'b-', label='validate')
plt_test, = plt.plot(np.log10(alpha_values), test_acc, 'r-', label='test')
plt.ylabel('Accuracy')
plt.xlabel('Regularization log10(lambda)')
plt.legend(handles=[plt_train, plt_validate, plt_test], bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0., fontsize='x-small')
plt.savefig(path1 + 'SVM_l2_PCA' + '.png')
plt.clf()

### elastic net regularization
ratio_values = np.array([0.2, 0.4, 0.6, 0.8]) #0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
result_table = []
for alpha in alpha_values:
    for ratio in ratio_values:
        clf = SGDClassifier(loss='hinge', penalty='elasticnet', alpha=alpha, l1_ratio=ratio)
        clf.fit(X_train_pca, y_train)
        result_table.append([alpha, ratio, clf.score(X_train_pca, y_train), clf.score(X_train_pca, y_train), clf.score(X_test_pca, y_test)])
df = p.DataFrame(result_table, columns=['Regularization', 'l1_ratio', 'Training', 'Validation', 'Test'])
print(df)

### SVM with raw features, elastic net regularization
ratio_values = np.array([0.2, 0.4, 0.6, 0.8]) #0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
result_table = []
for alpha in alpha_values:
    for ratio in ratio_values:
        clf = SGDClassifier(loss='hinge', penalty='elasticnet', alpha=alpha, l1_ratio=ratio)
        clf.fit(X_train_raw, y_train)
        result_table.append([alpha, ratio, clf.score(X_train_raw, y_train), clf.score(X_train_raw, y_train), clf.score(X_test_raw, y_test)])
df = p.DataFrame(result_table, columns=['Regularization', 'l1_ratio', 'Training', 'Validation', 'Test'])
print(df)

### Logistic regression with PCA features
for i in range(len):
    alpha = alpha_values[i]
    clf = SGDClassifier(loss='log', penalty='l1', alpha=alpha)
    clf.fit(X_train_pca, y_train)
    y_pred = (clf.predict(X_train_pca) >= 0.5)
    correct = 0
    for j in range(y_train.shape[0]):
        if y_train[j] == y_pred[j]:
            correct +=1
    train_acc[i] = correct/y_train.shape[0]
    y_pred = (clf.predict(X_validate_pca) >= 0.5)
    correct = 0
    for j in range(y_validate.shape[0]):
        if y_validate[j] == y_pred[j]:
            correct +=1
    validate_acc[i] = correct/y_validate.shape[0]
    y_pred = (clf.predict(X_test_pca) >= 0.5)
    correct = 0
    for j in range(y_validate.shape[0]):
        if y_test[j] == y_pred[j]:
            correct +=1
    test_acc[i] = correct/y_test.shape[0]

fig = plt.figure(figsize=(15, 9))
plt_train, = plt.plot(np.log10(alpha_values), train_acc, 'b-', label='training')
plt_validate, = plt.plot(np.log10(alpha_values), validate_acc, 'g-', label='validation')
plt_test, = plt.plot(np.log10(alpha_values), test_acc, 'r-', label='test')
plt.ylabel('Accuracy')
plt.xlabel('Regularization log10(lambda)')
#plt.title('Logistic regression with l1 regularization, PCA features')
plt.legend(handles=[plt_train, plt_validate, plt_test], bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0., fontsize='x-small')
plt.savefig(path1 + 'LR_PCA' + '.png')
plt.clf()

### Logistic regression with raw features
for i in range(len):
    alpha = alpha_values[i]
    clf = SGDClassifier(loss='log', penalty='l1', alpha=alpha)
    clf.fit(X_train_raw, y_train)
    y_pred = (clf.predict(X_train_raw) >= 0.5)
    correct = 0
    for j in range(y_train.shape[0]):
        if y_train[j] == y_pred[j]:
            correct +=1
    train_acc[i] = correct/y_train.shape[0]
    y_pred = (clf.predict(X_validate_raw) >= 0.5)
    correct = 0
    for j in range(y_validate.shape[0]):
        if y_validate[j] == y_pred[j]:
            correct +=1
    validate_acc[i] = correct/y_validate.shape[0]
    y_pred = (clf.predict(X_test_raw) >= 0.5)
    correct = 0
    for j in range(y_validate.shape[0]):
        if y_test[j] == y_pred[j]:
            correct +=1
    test_acc[i] = correct/y_test.shape[0]

fig = plt.figure(figsize=(15, 9))
plt_train, = plt.plot(np.log10(alpha_values), train_acc, 'b-', label='training')
plt_validate, = plt.plot(np.log10(alpha_values), validate_acc, 'g-', label='validation')
plt_test, = plt.plot(np.log10(alpha_values), test_acc, 'r-', label='test')
plt.ylabel('Accuracy')
plt.xlabel('Regularization log10(lambda)')
#plt.title('Logistic regression with l1 regularization, raw features')
plt.legend(handles=[plt_train, plt_validate, plt_test], bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0., fontsize='x-small')
plt.savefig(path1 + 'LR_raw' + '.png')
plt.clf()


### Error analysis
clf = SGDClassifier(loss='hinge', penalty='elasticnet', alpha=0.001, l1_ratio=0.2)
clf.fit(X_train_pca, y_train)
y_pred = clf.predict(X_test_pca)
print(y_pred)