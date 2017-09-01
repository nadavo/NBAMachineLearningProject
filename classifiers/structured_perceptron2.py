from sklearn.model_selection import train_test_split, learning_curve, KFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pystruct.models import StructuredModel, GraphCRF, ChainCRF, MultiClassClf
from seqlearn.perceptron import StructuredPerceptron
from pystruct.plot_learning import plot_learning as plt
import pandas as pd
import numpy as np
from time import time
import sys

columns_original = ['Season','TeamID','E/W','Conference Finalist','W/L','FG','FGA','3P','3PA','FT','FTA','ORB','DRB','AST','STL','BLK','TOV','PF','PTS','Pace','Attendance', 'Standings_Bucket',"(0, 0)","(0, 1)","(0, 2)","(1, 0)","(1, 1)","(1, 2)","(2, 0)","(2, 1)","(2, 2)","(0, 0, 0)","(0, 0, 1)","(0, 0, 2)","(0, 1, 0)","(0, 1, 1)","(0, 1, 2)","(0, 2, 0)","(0, 2, 1)","(0, 2, 2)","(1, 0, 0)","(1, 0, 1)","(1, 0, 2)","(1, 1, 0)","(1, 1, 1)","(1, 1, 2)","(1, 2, 0)","(1, 2, 1)","(1, 2, 2)","(2, 0, 0)","(2, 0, 1)","(2, 0, 2)","(2, 1, 0)","(2, 1, 1)","(2, 1, 2)","(2, 2, 0)","(2, 2, 1)","(2, 2, 2)", 'Standings_Bucket_Next']

columns_reduced = ['TeamID', 'E/W', 'Conference Finalist', 'FG', 'FGA', '3P', '3PA', 'FT', 'FTA', 'ORB', 'DRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'Pace', 'Attendance', 'Standings_Bucket', 'Standings_Bucket_Next']

columns = columns_original

def createMatrix(datafile):
    df = pd.read_csv(datafile, header=0, sep=',', usecols=columns)
    df.to_csv('reduced.csv', index = False)
    X, Y = df.iloc[:,:-1], df.iloc[:,-1]
    print("Number of samples: " + str(len(X)))
    print("Number of labels: " + str(len(Y)))
    return X.as_matrix(), Y.as_matrix()


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
    clf = StructuredPerceptron(verbose=1,max_iter=1000,random_state=1)
    print("Structured Perceptron 2")
    if cv_flag:
        print("Cross-Validation")
        errors.append(evaluateModel(clf, data, labels, True))
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.3, random_state=1)
        train_start = time()
        clf = clf.fit(X_train, Y_train,lengths=(30,))
        train_end = time()
        print("Training took " + str((train_end - train_start) / 60) + " minutes to complete\n")
        print("Results\n")
        print("Train")
        errors.append(evaluateModel(clf, X_train, Y_train))
        print("Test")
        errors.append(evaluateModel(clf, X_test, Y_test))
    return errors

def main():
    start = time()
    inputfile = sys.argv[1]
    print(inputfile)
    data, labels = createMatrix(inputfile)
    errors = createModel(data, labels, False)
    print("\nTrain Error: " + str(errors[0]))
    if len(errors) > 1:
        print("Test Error: " + str(errors[1]))
    end = time()
    print("\nProcess took " + str((end - start) / 60) + " minutes to complete")

if __name__ == '__main__':
    main()
