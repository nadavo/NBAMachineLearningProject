from sklearn.model_selection import train_test_split, learning_curve, KFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from time import time
import sys


def createMatrix(datafile):
    columns = ['Season','TeamID','E/W','Conference Finalist','W/L','FG','FGA','3P','3PA','FT','FTA','ORB','DRB','AST','STL','BLK','TOV','PF','PTS','Age','MOV','Pace','eFG%','Attendance','Standings_Bucket']
    df = pd.read_csv(datafile, header=0, sep=',', usecols=columns)
    df.to_csv('reduced_columns.csv', index = False)
    X, Y = df.iloc[:,:-1], df.iloc[:,-1]
    print("Number of samples: " + str(len(X)))
    print("Number of labels: " + str(len(Y)))
    print("Values distribution in dataset per feature:")
    for feature in columns:
        print(feature,df[feature].value_counts(normalize=True, sort=False, dropna=False))
    return X, Y


def evaluateModel(clf, data, labels, cv_flag=False):
    if cv_flag:
        kfold = KFold(n_splits=3, random_state=1)
        cv_start = time()
        predictions = cross_val_predict(clf, data, labels, cv=kfold, n_jobs=-1)
        cv_end = time()
        print("Cross-Validation took " + str((cv_end - cv_start) / 60) + " minutes to complete\n")
    else:
        prediction_start = time()
        predictions = clf.predict(data)
        prediction_end = time()
        print("Prediction took " + str((prediction_end - prediction_start) / 60) + " minutes to complete\n")
    error = 1.0 - float(accuracy_score(labels, predictions))
    print("Accuracy: " + str(accuracy_score(labels, predictions)))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, predictions))
    print("Classification Report:")
    print(classification_report(labels, predictions))
    return error


def createModel(data, labels, cv_flag=False):
    errors = list()
    clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=2,
                                 criterion='entropy')
    print("Random Forest")
    if cv_flag:
        print("Cross-Validation")
        errors.append(evaluateModel(clf, data, labels, True))
        #plot_learning_curve(clf, "Learning Curves (Random Forest)", data, labels, (0.7, 1.01),cv=KFold(n_splits=3, random_state=1), n_jobs=-1)
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.3, random_state=1)
        train_start = time()
        clf = clf.fit(X_train, Y_train)
        train_end = time()
        print("Training took " + str((train_end - train_start) / 60) + " minutes to complete\n")
        print("Results\n")
        print("Train")
        errors.append(evaluateModel(clf, X_train, Y_train))
        print("Test")
        errors.append(evaluateModel(clf, X_test, Y_test))
        #plot_learning_curve(clf, "Learning Curves (Random Forest)", X_train, Y_train, (0.7, 1.01),cv=KFold(n_splits=3, random_state=1), n_jobs=-1)
    #plt.show()
    return errors


# def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
#     plt.figure()
#     plt.title(title)
#     if ylim is not None:
#         plt.ylim(*ylim)
#     plt.xlabel("Training examples")
#     plt.ylabel("Score")
#     train_sizes, train_scores, test_scores = learning_curve(
#         estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)
#     plt.grid()
#
#     plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                      train_scores_mean + train_scores_std, alpha=0.1,
#                      color="r")
#     plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                      test_scores_mean + test_scores_std, alpha=0.1, color="g")
#     plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
#              label="Training score")
#     plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
#              label="Cross-validation score")
#
#     plt.legend(loc="best")
#     return plt


def main():
    start = time()
    inputfile = sys.argv[1]
    print(inputfile)
    data, labels = createMatrix(inputfile)
    errors = createModel(data, labels, False)
    print("Train Error: " + str(errors[0]))
    if len(errors) > 1:
        print("Test Error: " + str(errors[1]))
    end = time()
    print("\nProcess took " + str((end - start) / 60) + " minutes to complete")

if __name__ == '__main__':
    main()
