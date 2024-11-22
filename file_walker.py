import os
import random
import csv

# This script takes in a given input folder and walks through taking every single .dicom file and a given label
# and adding them to a .csv file with the assigned label
# If number to keep is not None then it will randomly select that number of .dicom files (use case is
# to trim the amount of training data to avoid too many of one kind of label)
# TODO Potential problem with this is it's not guaranteed to include all patients
path = '/Users/omar/Downloads/second-annual-data-science-bowl/train/train'
#path = '/Users/omar/Downloads/second-annual-data-science-bowl/test/test'
label = 0
number_to_keep = None
csv_name = 'train.txt'

############################################################################

# This chunk was taken from ChatGPT (with modification to only include sax folders)

matching_files = []
for root, dirs, files in os.walk(path):

    # Only include short axis views (by excluding folders starting with 2ch or 4ch)
    dirs[:] = [d for d in dirs if not (d.startswith('2ch') or d.startswith('4ch'))]

    for file in files:
        if file.lower().endswith('.dcm'.lower()):
            matching_files.append(os.path.join(root, file))

# Choose a random subset of our data (to avoid biasing towards one label)
if number_to_keep is not None:
    matching_files = random.sample(matching_files, number_to_keep)

print(matching_files)
print(len(matching_files))

# Now save as csv
# This part had help from https://www.geeksforgeeks.org/writing-csv-files-in-python/
rows = []
for i in range(0, len(matching_files)):
    row = []
    row.append((matching_files[i]))
    #row.append(random.choice([0, 1]))
    row.append(label)
    rows.append(row)
with open(csv_name, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(rows)