import csv
import pandas as pd
#####################################################
### creating a dictionary for each team #############
dicTeamID = {}
with open('2017.csv', newline='') as csvfile:
    dataTable = csv.reader(csvfile, delimiter=',', quotechar='|')
    dicTeams = []
    for row in dataTable:
        if row[0] == 'Rk':
            continue
        dicTeams.append(row[1])
    for i in range(1, 31):
        team = (sorted(dicTeams)).pop(0)
        dicTeamID[team] = i
        dicTeams.remove(team)
    dicTeamID['League Average'] = 31
print(dicTeamID)
#######################################################
######## merging tables a and b and adding key colomn #
for j in range(0, 18):
    fileName1 = str(2000 + j) + ".csv"
    fileName2 = str(2000 + j) + "B" + ".csv"
    ######################################################
    ### creating a key for each team per season ##########
    ### accourding to prime dictionary (2017) ############
    with open(fileName1, newline='') as csvfile:
        dataTable1 = csv.reader(csvfile, delimiter=',', quotechar='|')
        teams =[]
        for row in dataTable1:
            if row[0] == 'Rk':
                continue
            teams.append(row[1])
    ######################################################
    ######## merging tables a and b and adding key to ####
    ######## TeamID colomn ###############################
    file1 = pd.read_csv(fileName1)
    file2 = pd.read_csv(fileName2)
    x = file1.merge(file2, on='Team')
    x['TeamID'] = 100
    for row1 in x:
        if row1[0] == 'Rk':
            continue
        if row1[1] in dicTeamID.keys():
            row1[53] = dicTeamID[row1[1]]
    x.to_csv(fileName1, index = False)


