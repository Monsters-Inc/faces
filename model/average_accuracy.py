import sys
preprocessing = 'No Preprocessing'
runs = 1

if len(sys.argv) > 1:
    preprocessing = sys.argv[1].lower()

    if len(sys.argv) > 2:
        runs = sys.argv[2].lower()

with open('results.txt') as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content]

sum = 0
for i in content:
    sum += float(i)

accuracy = sum / len(content)

print(accuracy)

# This appends the accuracy to results.txt
f = open("results_dataset.txt", "a")
f.write(preprocessing+' ('+str(runs)+'): '+str(accuracy)+'\n')
f.close()

# Erase results file
file = open("results.txt", "r+")
file.truncate(0)
file.close()
