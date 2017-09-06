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

columns_reduced_next = ['Season','TeamID','Team','E/W','Conference Finalist','W/L','3PA','2PA','ORB','DRB','AST','STL','BLK','PTS' ,'Pace' ,'Standings_Bucket','Standings_Bucket_Next']

columns_reduced = ['Season','TeamID','Team','E/W','Conference Finalist','W/L','3PA','2PA','ORB','DRB','AST','STL','BLK','PTS','Pace','Standings_Bucket']

columns_best = ['Season','TeamID','Team','E/W','Conference Finalist','W/L','3PA','2PA','ORB','DRB','BLK','PTS','Standings_Bucket']

columns = columns_reduced_next
selected_label = columns[-1]

columns_reduced_print = columns[4:-1]
labels_print = ['0','1']

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


def evaluateModel(clf, data, labels, test_flag=False, score_override=False):
    prediction_start = time()
    if test_flag or score_override:
        predictions = clf.predict(data)
        print("Predictions:")
        print(predictions)
    if score_override:
        score = 0
        num_test = 0
        for seq in range(len(labels)):
            diff = list()
            for num in range(len(labels[seq])):
                diff.append(np.abs(predictions[seq][num] - labels[seq][num]))
                num_test += 1
            score += int(sum(diff))
        score = float(score)/num_test
        print("Averaged Test Score: ", str(score))
    else:
        score = clf.score(data,labels)
        print("Accuracy: ", score)
    prediction_end = time()
    print("Prediction took " + str((prediction_end - prediction_start) / 60) + " minutes to complete\n")
    # print("Accuracy: " + str(accuracy_score(labels, predictions)))
    # print("Confusion Matrix:")
    # print(confusion_matrix(labels, predictions))
    # print("Classification Report:")
    # print(classification_report(labels, predictions))
    return score

def printWeights(weights):
    i=0
    for label in labels_print:
        for feature in columns_reduced_print:
            print(label,feature,str(weights[i]))
            i+=1
    for label_i in labels_print:
        for label_j in labels_print:
            print(label_i,label_j,str(weights[i]))
            i+=1
    print(str(i),"Feautre Weights")

def createModel(data, labels, num_classes=2):
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
        num_classes = df_team[selected_label].nunique()
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
        model = createModel(X_train, Y_train, num_classes)
        print("\nTrain "+team_name+"\n")
        accuracy[team_name]['Train'] = evaluateModel(model, X_train, Y_train)
        print("\nTest "+team_name+"\n")
        accuracy[team_name]['Test'] = evaluateModel(model, X_test, Y_test, test_flag=True)
        print("\nScore "+team_name+" \n")
        accuracy[team_name]['Score'] = evaluateModel(model, X_test, Y_test, test_flag=True, score_override=True)
        models[team_name] = model
        printWeights(model.w)
    avg_train = list()
    avg_test = list()
    avg_score = list()
    print("\nAccuracy and Score Results\n")
    for team in accuracy.keys():
        print("\n"+team)
        avg_train.append(accuracy[team]['Train'])
        avg_test.append(accuracy[team]['Test'])
        avg_score.append(accuracy[team]['Score'])
        print("Train: ",accuracy[team]['Train'])
        print("Test: ",accuracy[team]['Test'])
        print("Score: ",accuracy[team]['Score'])
    print("\nAverage Train Accuracy: " + str(np.mean(avg_train)))
    print("Average Test Accuracy: " + str(np.mean(avg_test)))
    print("Average Test Score: " + str(np.mean(avg_score)))
    end = time()
    print("\nProcess took " + str((end - start) / 60) + " minutes to complete")

if __name__ == '__main__':
    main()
