from sklearn.model_selection import train_test_split, learning_curve, KFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pystruct.models import StructuredModel, GraphCRF, ChainCRF, MultiClassClf
from pystruct.learners import StructuredPerceptron
from pystruct.plot_learning import plot_learning as plt
import pandas as pd
import numpy as np
from time import time
import sys

columns_original = ['Season','TeamID','E/W','Conference Finalist','W/L','FG','FGA','3P','3PA','FT','FTA','ORB','DRB','AST','STL','BLK','TOV','PF','PTS','Pace','Attendance', 'Standings_Bucket', "(0, 0)", "(0, 1)", "(0, 2)","(1, 0)","(1, 1)","(1, 2)","(2, 0)","(2, 1)","(2, 2)","(0, 0, 0)","(0, 0, 1)","(0, 0, 2)","(0, 1, 0)","(0, 1, 1)","(0, 1, 2)","(0, 2, 0)","(0, 2, 1)","(0, 2, 2)","(1, 0, 0)","(1, 0, 1)","(1, 0, 2)","(1, 1, 0)","(1, 1, 1)","(1, 1, 2)","(1, 2, 0)","(1, 2, 1)","(1, 2, 2)","(2, 0, 0)","(2, 0, 1)","(2, 0, 2)","(2, 1, 0)","(2, 1, 1)","(2, 1, 2)","(2, 2, 0)","(2, 2, 1)","(2, 2, 2)", 'Standings_Bucket_Next']

#0.6 with 18 seasons
columns_reduced = ['Season','TeamID', 'E/W', 'Conference Finalist', 'W/L', 'SRS', 'Standings_Bucket', 'Standings_Bucket_Next']

columns = columns_reduced

def createMatrix(datafile):
    df = pd.read_csv(datafile, header=0, sep=',', usecols=columns)
    df.to_csv('reduced.csv', index = False)
    teamIDs = df['TeamID'].unique()
    X_chained = list()
    Y_chained = list()
    for team in teamIDs:
        df_team = df[(df['TeamID']==team)&(df['Season']>2009)]
        X, Y = df_team.iloc[:,3:-1], pd.to_numeric(df_team.iloc[:,-1],downcast='unsigned')
        X_chain = X.as_matrix()
        X_chained.append(X_chain)
        Y_chain = Y.as_matrix()
        Y_chained.append(Y_chain)
    print("Number of sample chains: " + str(len(X_chained)))
    print("Number of label chains: " + str(len(Y_chained)))
    return X_chained, Y_chained


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
        score = clf.score(data,labels)
        prediction_end = time()
        print("Prediction took " + str((prediction_end - prediction_start) / 60) + " minutes to complete\n")
    print(score)
    error = 1.0 - float(score)
    # error = 1.0 - float(accuracy_score(labels, predictions))
    # print("Accuracy: " + str(accuracy_score(labels, predictions)))
    # print("Confusion Matrix:")
    # print(confusion_matrix(labels, predictions))
    # print("Classification Report:")
    # print(classification_report(labels, predictions))
    return error


def createModel(data, labels, cv_flag=False):
    errors = list()
    model = ChainCRF(n_states=3,n_features=int(len(columns)-4),directed=True)
    clf = StructuredPerceptron(model=model,max_iter=10,verbose=False,batch=False,average=False)
    print("Structured Perceptron + Chain CRF")
    if cv_flag:
        print("Cross-Validation")
        errors.append(evaluateModel(clf, data, labels, True))
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.3, random_state=1)
        train_start = time()
        clf.fit(X=X_train, Y=Y_train)
        train_end = time()
        print("Training took " + str((train_end - train_start) / 60) + " minutes to complete\n")
        print("Results\n")
        print("Train")
        train_score = clf.score(X_train,Y_train)
        print(train_score)
        errors.append(1.0 - float(train_score))
        #errors.append(evaluateModel(clf, X_train, Y_train))
        print("Test")
        print(clf.predict(X_test))
        test_score = clf.score(X_test,Y_test)
        print(test_score)
        errors.append(1.0 - float(test_score))
        #errors.append(evaluateModel(clf, X_test, Y_test))
    #plt(clf)
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
