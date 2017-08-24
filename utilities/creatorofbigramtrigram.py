import csv
import pandas as pd
datafiles = ['12seasons.csv', 'allseasons.csv']
ewdict = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:1, 8:1, 9:0, 10:1, 11:1, 12:0, 13:1, 14:1, 15:1, 16:0, 17:0, 18:1, 19:1, 20:0, 21:1, 22:0, 23:0, 24:1, 25:1, 26:1, 27:1, 28:0, 29:1, 30:0}

for q in range (0,2):
    granddict = {}
#####################################################################
### creating the dictionary for each team ###########################
    with open(datafiles[q], 'r') as f:
        data = csv.reader(f, delimiter=',', quotechar='|')
        values = []
        teams = list(range(1, 31))
        years = list(range(2017, 2005, -1))
        if q == 1:
            years = list(range(2017, 1999, -1))
        for row in data:
            for team in teams:
                if row[1] == 'TeamID':
                    continue
                if int(row[1]) == team:
                    values.append((team,row[0],row[53]))
        for team, year, position in values:
            granddict.setdefault(team, {})[year] = position
        print(granddict)
#####################################################################
### editing the files, adding bigrams and trigrams and E/W fix ######
    x = pd.read_csv(datafiles[q])
    x.set_index('Season','TeamsID')
    file = x
    #################################
    ### creating headilnes ##########
    headlines = []
    for i in [0,1,2]:
        for j in [0,1,2]:
            headlines.append((i,j))
    for i in [0,1,2]:
        for j in [0,1,2]:
            for k in [0,1,2]:
                headlines.append((i,j,k))

