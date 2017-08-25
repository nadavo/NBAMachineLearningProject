import csv
import pandas as pd
#####################################################
### creating a dictionary for each team #############
dicTeamID = {}
with open('2017.csv', newline='') as csvfile:
    dataTable = csv.reader(csvfile, delimiter=',', quotechar='|')
    dicTeams = []
    for row in dataTable:
        if row[0] == 'Rk' or row[1] == 'League Average':
            continue
        dicTeams.append(row[1])
    for i in range(1, 31):
        team = (sorted(dicTeams)).pop(0)
        dicTeamID[team] = i
        dicTeams.remove(team)
    dicTeamID['League Average'] = 31
print(dicTeamID)
#######################################################
### merging tables a and b and adding data to table ###
for j in range(0, 18):
    fileName1 = str(2000 + j) + ".csv"
    fileName2 = str(2000 + j) + "B" + ".csv"
######################################################
### creating a key for each team per season ##########
### accourding to prime dictionary (2017) ############
    teams = []
    with open(fileName1, newline='') as csvfile:
        dataTable1 = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in dataTable1:
            if row[0] == 'Rk':
                continue
            if row[1] in dicTeamID.keys():
                teams.append((row[1], dicTeamID[row[1]]))
                if 2000 + j < 2003 and row[1] == 'Charlotte Hornets':
                    teams.remove((row[1], dicTeamID[row[1]]))
                    teams.append(('Charlotte Hornets', dicTeamID['New Orleans Pelicans']))
            else:
                if row[1] == 'Seattle SuperSonics':
                    teams.append((row[1], dicTeamID['Oklahoma City Thunder']))
                if row[1] == 'New Jersey Nets':
                    teams.append((row[1], dicTeamID['Brooklyn Nets']))
                if row[1] == 'Vancouver Grizzlies':
                    teams.append((row[1], dicTeamID['Memphis Grizzlies']))
                if row[1] == 'New Orleans Hornets':
                    teams.append((row[1], dicTeamID['New Orleans Pelicans']))
                if row[1] == 'New Orleans/Oklahoma City Hornets':
                    teams.append((row[1], dicTeamID['New Orleans Pelicans']))
                if row[1] == 'Charlotte Bobcats':
                    teams.append((row[1], dicTeamID['Charlotte Hornets']))

######################################################
######## merging tables a and b and adding key to ####
######## TeamID colomn and adding andd eleting colomns
    file1 = pd.read_csv(fileName1)
    file1.set_index('Team')
    file2 = pd.read_csv(fileName2)
    file2.set_index('Team')
    x = file1.merge(file2, on='Team', how='outer')
    x['Season'] = 2000+j
    del x['Rk_y']
    del x['Rk_x']
    x['TeamID'] = x['Team']
    ###print(len(teams), teams, 2000+j)
    for franchise in teams:
        val = franchise[1]
        x['TeamID'].replace(franchise[0],val,inplace=True)
    x['W/L'] = x['W']/x['L']
    x['Standings_Bucket'] = x['Standings_Bucket ']
    del x['Standings_Bucket ']
    x = x[['Season','TeamID','Team','E/W','Conference Finalist','G','W','L','W/L','MP','FG','FGA','FG%','3P','3PA','3P%','2P','2PA','2P%','FT','FTA','FT%','ORB','DRB','TRB','AST','STL','BLK','TOV','PF','PTS','Age','PW','PL','MOV','SOS','SRS','ORtg','DRtg','Pace','FTr','3PAr','TS%','eFG%','TOV%','ORB%','FT/FGA','eFG%','TOV%','DRB%','FT/FGA','Arena','Attendance','Standings_Bucket']]
    x.to_csv(fileName1, index = False)





