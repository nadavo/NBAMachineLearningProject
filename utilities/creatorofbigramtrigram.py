import csv
import pandas as pd
datafiles = ['18unstructured.csv','24unstructured.csv', '31unstructured.csv']
savedata = ['18structured_next.csv','24structured_next.csv','31structured_next.csv']
ewdict = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:1, 8:1, 9:0, 10:1, 11:1, 12:0, 13:1, 14:1, 15:1, 16:0, 17:0, 18:1, 19:1, 20:0, 21:1, 22:0, 23:0, 24:1, 25:1, 26:1, 27:1, 28:0, 29:1, 30:0}

for q in range (0,3):
    granddict = {}
    uberdict = {}
#####################################################################
### creating the dictionary for each team ###########################
    with open(datafiles[q], 'r') as f:
        data = csv.reader(f, delimiter=',', quotechar='|')
        values = []
        values2 = []
        teams = list(range(1, 31))
        # if q == 1:
        #     years = list(range(2017, 1986, -1))
        for row in data:
            for team in teams:
                if row[1] == 'TeamID':
                    continue
                if int(row[1]) == team:
                    values.append((team,row[0],row[53]))
        for team, year, position in values:
            granddict.setdefault(team, {})[year] = position
            uberdict.setdefault(team, {})[str(int(year)-1)] = position
        #print(granddict[4])
        #print(uberdict[4])
#####################################################################
### editing the files, adding bigrams and trigrams dictionaries and E/W fix ######
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
        # if row['E/W'] != ewdict.get(row['TeamID']):
        #     print(row['Season'],row['Team'])
        for i in range(0, len(headlines1)):
            team = uberdict[row['TeamID']]
            val = 0
            if (q == 0 and row['Season'] > 2000) or (q == 1 and row['Season'] > 1994) or (q == 2 and row['Season'] > 1987):
                left = int(float(team.get(str(row['Season']), 2)))
                right = int(float(team.get(str(row['Season']-1),2)))
                tupp = (right, left)
                if tupp == (headlines1[i]):
                    val = 1
            file.set_value(j, str(headlines1[i]), val)
    for i in range(0, len(headlines2)):
        for j, row in x.iterrows():
            team = uberdict[row['TeamID']]
            val = 0
            if (q == 0 and row['Season'] > 2001) or (q == 1 and row['Season'] > 1995) or (q == 2 and row['Season'] > 1988):
                left = int(float(team.get(str(row['Season']), 2)))
                middle = int(float(team.get(str(row['Season']-1),2)))
                right = int(float(team.get(str(row['Season']-2),2)))
                tupp = (right, middle, left)
                if tupp == (headlines2[i]):
                    val = 1
            file.set_value(j, str(headlines2[i]), val)
    file['Standings_Bucket'] = labels
    file = file.sort_values(by='Season')
    for j, row in file.iterrows():
        # if row['TeamID'] == 4 and row['Season'] > 2001 and row['Season'] < 2004:
        #     continue
        team = uberdict[row['TeamID']]
        year = team.get(str(row['Season']), 2)
        if row['Season'] == 2017:
            year = -1
        file.set_value(j, 'Standings_Bucket_Next', year)
    file = file.sort_values(by=['TeamID','Season'])
    file.to_csv(savedata[q], index=False)


