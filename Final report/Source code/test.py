# Read in the file
text_file = open("Output.txt", "r")
with open('Output2.txt', 'w') as file:
    for line in text_file:
        line = line.replace("_", "")
        file.write(line)
# Write the file out again

