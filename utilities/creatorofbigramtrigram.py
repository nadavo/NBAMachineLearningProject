import csv
import pandas as pd
datafiles = ['24unstructured.csv', '31unstructured.csv']
savedata = ['24structured.csv','31structured.csv']
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
            years = list(range(2017, 1986, -1))
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
    labels = x['Standings_Bucket']
    del x['Standings_Bucket']
    #################################
    ### creating the headilnes ######
    headlines1 = []
    headlines2 = []
    for i in [0,1,2]:
        for j in [0,1,2]:
            headlines1.append((int(i),int(j)))
            file[str((i,j))] = 0
    for i in [0,1,2]:
        for j in [0,1,2]:
            for k in [0,1,2]:
                headlines2.append((i,j,k))
                file[str((i,j,k))] = 0
######################################################################
### adding the correct value to the bigram and trigram and E/W #######
    for j, row in x.iterrows():
        if row['E/W'] != ewdict.get(row['TeamID']):
            print(row['Season'],row['Team'])
        for i in range(0, len(headlines1)):
            team = granddict[row['TeamID']]
            val = 0
            if (q == 0 and row['Season'] > 2006) or (q == 1 and row['Season'] > 2000):
                left = int(float(team[str(row['Season'])]))
                right = int(float(team.get(str(row['Season']-1),2)))
                tupp = (right, left)
                if tupp == (headlines1[i]):
                    val = 1
            file.set_value(j, str(headlines1[i]), val)
    for i in range(0, len(headlines2)):
        for j, row in x.iterrows():
            team = granddict[row['TeamID']]
            val = 0
            if (q == 0 and row['Season'] > 2007) or (q == 1 and row['Season'] > 2001):
                left = int(float(team[str(row['Season'])]))
                middle = int(float(team.get(str(row['Season']-1),2)))
                right = int(float(team.get(str(row['Season']-2),2)))
                tupp = (right, middle, left)
                if tupp == (headlines2[i]):
                    val = 1
            file.set_value(j, str(headlines2[i]), val)
    file['Standings_Bucket'] = labels
    file.to_csv(savedata[q], index=False)


