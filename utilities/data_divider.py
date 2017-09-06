import csv
import pandas as pd
years = [1987,1997,2007]
times = [0,1,2]


for time in times:
    file = pd.read_csv('31structured_next.csv')
    #############################################
    ### divide by conference ####################
    east = file
    west = file
    if time == 0:
        for j, row in file.iterrows():
            if row['E/W']==0:
                west = west.drop(file.index[j])
            else:
                east = east.drop(file.index[j])
        east.to_csv('East.csv', index=False)
        west.to_csv('West.csv', index=False)

    ##########################################
    ### divide by seasons ####################

    season = file
    for j, row in file.iterrows():
        if row['Season'] > years[time]+10 or row['Season'] < years[time]:
            season = season.drop(file.index[j])

    season.to_csv(str(years[time])+".csv", index=False)

    ########################################################
    ### create binary data files for 18, 24, 31 seasons ####
    q = [1987, 1994, 2000]
    binary = file
    for j, row in file.iterrows():
        if row['Season'] < q[time]:
            binary = binary.drop(file.index[j])
            continue
        if row['Standings_Bucket']==1:
            binary.set_value(index=j, col='Standings_Bucket', value=0)
        if row['Standings_Bucket']==2:
            binary.set_value(index=j, col='Standings_Bucket', value=1)
        if row['Standings_Bucket_Next']==1:
            binary.set_value(index=j, col='Standings_Bucket_Next', value=0)
        if row['Standings_Bucket_Next']==2:
            binary = binary.set_value(index=j, col='Standings_Bucket_Next', value=1)

    name = int(2018-q[time])

    binary.to_csv(str(name)+'Binary.csv')