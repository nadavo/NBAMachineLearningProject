import csv
for i in range (0,18):
    fileName = str(2000+i)+".csv"
    conAdd = 0
    finAdd = 0
    zeroCount = 0
    oneCount = 0
    twoCount = 0
    with open(fileName, newline='') as csvfile:
        dataTable = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in dataTable:
            if row[0] == 'Rk' or row[1] == 'League Average':
                if row[0] == 'Rk':
                    if row[26] != 'Standings_Bucket ':
                        print(2000 + i, row[26])
                continue
            conAdd = conAdd + int(row[27])
            finAdd = finAdd + int(row[25])
            if row[26] == str(0):
                zeroCount = zeroCount + 1
            elif row[26] == str(1):
                oneCount = oneCount + 1
            else:
                twoCount = twoCount + 1
#############################################################################
### miami entered the league in 1989 (25 teams) #############################
### minnesota and orlando entered the league in 1990 (27 teams) #############
### vancouver and toronto entered the league in 1996 (29 teams) #############
        if i+2000 < 2005:
            if conAdd != 14 or finAdd != 4 or zeroCount != 8 or oneCount != 8 or twoCount != 13:
                print(fileName, conAdd, finAdd, zeroCount, oneCount, twoCount)
        else:
            if conAdd != 15 or finAdd != 4 or zeroCount != 8 or oneCount != 8 or twoCount != 14:
                print(fileName, conAdd, finAdd, zeroCount, oneCount, twoCount)





