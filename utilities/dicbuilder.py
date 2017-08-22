import csv
with open('2017.csv', newline='') as csvfile:
    dataTable1 = csv.reader(csvfile, delimiter=',', quotechar='|')
    teams =[]
    for row in dataTable1:
        if row[0] == 'Rk' or row[1] == 'League Average':
            continue
        teams.append(row[1])
    TeamID = []
    for i in range (1,31):
        team = (sorted(teams)).pop(0)
        TeamID.append((i,team))
        teams.remove(team)
    TeamID.append((31,'League Average'))
    for i in range(0, 18):
        fileName1 = str(2000 + i) + ".csv"
        fileName2 = str(2000 + i)+ "B" + ".csv"
        with open(fileName1, newline='') as csvfile:
            dataTable1 = csv.reader(csvfile, delimiter=',', quotechar='|')
        with open(fileName2, newline='') as csvfile:
            dataTable2 = csv.reader(csvfile, delimiter=',', quotechar='|')
        with open(fileName2, newline='') as csvfile:
            dataWriter = csv.writer(csvfile, delimiter=',', quotechar='|')

