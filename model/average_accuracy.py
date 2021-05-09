with open('results.txt') as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content] 

sum = 0
for i in content:
    sum += float(i)

accuracy = sum / len(content)

print(accuracy)

# Erase results file
file = open("results.txt","r+")
file.truncate(0)
file.close()