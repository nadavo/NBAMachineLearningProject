import csv
with open('2017.csv', newline='') as csvfile:
    dataTable = csv.reader(csvfile, delimiter=',', quotechar='|')
    teams =[]
    for row in dataTable:
        if row[0] == 'Rk' or row[1] == 'League Average':
            continue
        teams.append(row[1])
    sortedTeams = []
    for i in range (1,31):
        team = (sorted(teams)).pop(0)
        sortedTeams.append((i,team))
        teams.remove(team)
    print(sortedTeams)
