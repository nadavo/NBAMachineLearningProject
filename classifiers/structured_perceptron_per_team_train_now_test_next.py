from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, make_scorer
from pystruct.models import StructuredModel, GraphCRF, ChainCRF, MultiClassClf
from pystruct.learners import StructuredPerceptron
from pystruct.plot_learning import plot_learning as plt
import pandas as pd
import numpy as np
from time import time
from random import shuffle, sample
from math import floor
import sys

columns_original = ['Season','TeamID','E/W','Conference Finalist','W/L','FG','FGA','3P','3PA','FT','FTA','ORB','DRB','AST','STL','BLK','TOV','PF','PTS','Pace','Attendance', 'Standings_Bucket', "(0, 0)", "(0, 1)", "(0, 2)","(1, 0)","(1, 1)","(1, 2)","(2, 0)","(2, 1)","(2, 2)","(0, 0, 0)","(0, 0, 1)","(0, 0, 2)","(0, 1, 0)","(0, 1, 1)","(0, 1, 2)","(0, 2, 0)","(0, 2, 1)","(0, 2, 2)","(1, 0, 0)","(1, 0, 1)","(1, 0, 2)","(1, 1, 0)","(1, 1, 1)","(1, 1, 2)","(1, 2, 0)","(1, 2, 1)","(1, 2, 2)","(2, 0, 0)","(2, 0, 1)","(2, 0, 2)","(2, 1, 0)","(2, 1, 1)","(2, 1, 2)","(2, 2, 0)","(2, 2, 1)","(2, 2, 2)", 'Standings_Bucket_Next']

#0.6 with 18 seasons regular chain
columns_reduced = ['Season','TeamID', 'Team', 'E/W', 'Conference Finalist', 'W/L', 'SRS', '2PA', '3PA', 'DRB', 'ORB', 'Standings_Bucket', 'Standings_Bucket_Next']

columns = columns_reduced

def createMatrix(datafile,seq_length,test_size):
    df_team = datafile
    test_size = test_size * (-1)
    X_train = list()
    Y_train = list()
    first_season = int(df_team['Season'].min())
    num_seasons = df_team['Season'].nunique()
    for i in range(0,num_seasons,seq_length):
        df_season = df_team[(df_team['Season']>=int(first_season+i))&(df_team['Season']<int(first_season+i+seq_length))]
        X, Y = df_season.iloc[:,4:-1].apply(pd.to_numeric), pd.to_numeric(df_season.iloc[:,-1],downcast='integer')
        X_chain = X.as_matrix()
        X_train.append(X_chain)
        Y_chain = Y.as_matrix()
        Y_train.append(Y_chain)
    X_test = X_train[test_size:]
    Y_test = Y_train[test_size:]
    X_train = X_train[:test_size]
    Y_train = Y_train[:test_size]
    print("Number of training sample chains: " + str(len(X_train)))
    print("Number of training label chains: " + str(len(Y_train)))
    print("Number of test sample chains: " + str(len(X_test)))
    print("Number of test label chains: " + str(len(Y_test)))
    return X_train, Y_train, X_test, Y_test


def evaluateModel(clf, data, labels, test_flag=False):
    prediction_start = time()
    if test_flag:
        predictions = clf.predict(data)
        print("Predictions:")
        print(predictions)
    score = clf.score(data,labels)
    print("Accuracy: ", score)
    prediction_end = time()
    print("Evaluation took " + str((prediction_end - prediction_start) / 60) + " minutes to complete\n")
    # print("Accuracy: " + str(accuracy_score(labels, predictions)))
    # print("Confusion Matrix:")
    # print(confusion_matrix(labels, predictions))
    # print("Classification Report:")
    # print(classification_report(labels, predictions))
    return score

def createModel(data, labels, num_classes=3):
    model = ChainCRF(n_states=num_classes,n_features=int(len(columns)-5),directed=True)
    clf = StructuredPerceptron(model=model,max_iter=200,verbose=False,batch=False,average=True)
    print("Structured Perceptron + Chain CRF")
    train_start = time()
    clf.fit(X=data, Y=labels)
    train_end = time()
    print("Training took " + str((train_end - train_start) / 60) + " minutes to complete\n")
    return clf

def main():
    start = time()
    inputfile = sys.argv[1]
    print(sys.argv[0],inputfile)
    df = pd.read_csv(inputfile, header=0, sep=',', usecols=columns)
    df.to_csv('reduced.csv', index = False)
    teamIDs = df['TeamID'].unique()
    models = dict()
    accuracy = dict()
    for team in teamIDs:
        df_team = df[df['TeamID']==team]
        num_classes = df_team['Standings_Bucket_Next'].nunique()
        team_name = str(team) + " -> " + str(df_team['Team'].unique()[0])
        print("\n"+team_name+"\n")
        X_train, Y_train, X_test, Y_test = createMatrix(datafile=df_team,seq_length=3,test_size=3)
        # if team==19 or team==10 or team==13 or team==4 or team==7:
        #     print(num_classes)
        #     print(X_train)
        #     print(Y_train)
        #     print(X_test)
        #     print(Y_test)
        accuracy[team_name] = dict()
        num_classes_list = list()
        for chain in Y_train:
            for num in chain:
                num_classes_list.append(num)
        num_classes = len(np.unique(num_classes_list))
        models[team_name] = createModel(X_train, Y_train, num_classes)
        print("\nTrain "+team_name+"\n")
        accuracy[team_name]['Train'] = evaluateModel(models[team_name], X_train, Y_train)
        print("\nTest "+team_name+"\n")
        accuracy[team_name]['Test'] = evaluateModel(models[team_name], X_test, Y_test, test_flag=True)
    avg_train = list()
    avg_test = list()
    print("\nAccuracy Results\n")
    for team in accuracy.keys():
        print("\n"+team)
        avg_train.append(accuracy[team]['Train'])
        avg_test.append(accuracy[team]['Test'])
        print("Train: ",accuracy[team]['Train'])
        print("Test: ",accuracy[team]['Test'])
    print("\nAverage Train Accuracy: " + str(np.mean(avg_train)))
    print("Average Test Accuracy: " + str(np.mean(avg_test)))
    end = time()
    print("\nProcess took " + str((end - start) / 60) + " minutes to complete")

if __name__ == '__main__':
    main()
