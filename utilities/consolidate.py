import sys

input_dir = sys.argv[1]
output_filename = sys.argv[2]
num_seasons = sys.argv[3]

output_lines = list()
averages = list()
header = str()

for i in range(num_seasons):
    filename = input_dir + "\\" + str(2017-i) + ".csv"
    with open(filename, 'r') as f:
        for line in f:
            if 'TeamID' in line:
                header = line
            elif 'League Average' in line:
                averages.append(line)
            else:
                output_lines.append(line)

with open(output_filename+".csv", 'w') as f:
        f.write(header)
        for line in output_lines:
            f.write(line)

output_filename = output_filename + "_averages.csv"

with open(output_filename, 'w') as f:
        f.write(header)
        for line in averages:
            f.write(line)
