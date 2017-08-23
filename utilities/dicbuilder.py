import csv
import pandas as pd
with open('2017.csv', newline='') as csvfile:
    dataTable = csv.reader(csvfile, delimiter=',', quotechar='|')
    dicTeams = []
    for row in dataTable:
        if row[0] == 'Rk':
            continue
        dicTeams.append(row[1])
    dicTeamID = []
    for i in range(1, 31):
        team = (sorted(dicTeams)).pop(0)
        dicTeamID.append((team, i))
        dicTeams.remove(team)
    dicTeamID.append(('League Average', 31))
    print(dicTeamID)
for j in range(0, 18):
    fileName1 = str(2000 + j) + ".csv"
    fileName2 = str(2000 + j) + "B" + ".csv"
    ##################################################
    #### merging tables a and b ######################
    '''file1 = pd.read_csv(fileName1)
    file2 = pd.read_csv(fileName2)
    x = file1.merge(file2, on='Team')
    x.to_csv(fileName1, index = False)'''
    ##################################################
    ##################################################
    with open(fileName1, newline='') as csvfile:
        dataTable1 = csv.reader(csvfile, delimiter=',', quotechar='|')
        teams =[]
        for row in dataTable1:
            if row[0] == 'Rk':
                continue
            teams.append(row[1])
        TeamID = []
        for i in range (1,31):
            team = (sorted(teams)).pop(0)
            TeamID.append((team,i))
            teams.remove(team)
        TeamID.append(('League Average',31))
        print(TeamID)