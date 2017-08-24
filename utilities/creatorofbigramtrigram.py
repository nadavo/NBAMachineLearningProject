import csv
import pandas as pd
datafiles = ['12seasons.csv', 'allseasons.csv']
savedata = ['test1.csv','test2.csv']
ewdict = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:1, 8:1, 9:0, 10:1, 11:1, 12:0, 13:1, 14:1, 15:1, 16:0, 17:0, 18:1, 19:1, 20:0, 21:1, 22:0, 23:0, 24:1, 25:1, 26:1, 27:1, 28:0, 29:1, 30:0}

for q in range (0,2):
    granddict = {}
#####################################################################
### creating the dictionary for each team ###########################
    with open(datafiles[q], 'r') as f:
        data = csv.reader(f, delimiter=',', quotechar='|')
        values = []
        teams = list(range(1, 31))
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
    file = x
    #################################
    ### creating the headilnes ######
    headlines1 = []
    headlines2 = []
    for i in [0,1,2]:
        for j in [0,1,2]:
            headlines1.append((int(i),int(j)))
    for i in [0,1,2]:
        for j in [0,1,2]:
            for k in [0,1,2]:
                headlines2.append((i,j,k))
######################################################################
### adding the correct value to the bigram and trigram and E/W #######
    for i in range(0, len(headlines1)):
        for j, row in x.iterrows():
            team = granddict[row['TeamID']]
            #row['E/W'] = ewdict[str(row['TeamID'])]
            val = 0
            if (q == 0 and row['Season'] != 2006) or (q == 1 and row['Season'] != 2000):
                left = int(float(team[str(row['Season'])]))
                right = int(float(team[str(row['Season']-1)]))
                tupp = (right, left)
                if tupp == (headlines1[i]):
                    val = 1
            file[str(headlines1[i])] = val
    '''for i in range(0, len(headlines2)):
        for j, row in x.iterrows():
            team = granddict[row['TeamID']]
            val = 0
            if (q == 0 and row['Season'] != 2007) or (q == 1 and row['Season'] != 2001):
                left = int(float(team[str(row['Season'])]))
                middle = int(float(team[str(row['Season']-1)]))
                right = int(float(team[str(row['Season']-2)]))
                tupp = (right, middle, left)
                if tupp == (headlines2[i]):
                    val = 1
            file[str(headlines2[i])] = val'''
    file.to_csv(savedata[q], index=False)


