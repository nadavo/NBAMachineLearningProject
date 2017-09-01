import csv
import pandas as pd
years = [1987,1994,2002]
times = [0,1,2]


file = pd.read_csv('31structured_next.csv')
for time in times:
    x = file
    y = file
    for j, row in file.iterrows():
        if row['Season'] > years[time]+16:
            y = y.drop(file.index[j])
            x = x.drop(file.index[j])
            continue
        else:
            if row['E/W'] == 0:
                y = y.drop(file.index[j])
                continue
            else:
                x = x.drop(file.index[j])
                continue
    x.to_csv(str(years[time])+'East.csv', index=False)
    y.to_csv(str(years[time])+'west.csv', index=False)
