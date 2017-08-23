import sys

input_dir = sys.argv[1]
output_filename = sys.argv[2]
num_seasons = int(sys.argv[3])

output_lines = list()
header = str()

for i in range(num_seasons):
    filename = input_dir + "/" + str(2017-i) + ".csv"
    with open(filename, 'r') as f:
        for line in f:
            if 'TeamID' in line:
                header = line
                continue
            output_lines.append(line)

with open(output_filename, 'w') as f:
        f.write(header)
        for line in output_lines:
            f.write(line)
