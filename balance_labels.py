import os
import pandas as pd
import numpy as np
import random
# This function takes in a data frame of image, labels with predominantly one label
# and replicates the other labeled data to equal the length of the former
def balance_labels(labels_df):

    # Split this into the positive and negative labels
    positive_labels = []
    negative_labels = []
    for i in range(0, len(labels_df)):

        file = labels_df.iloc[i, 0]
        label = labels_df.iloc[i, 1]

        # New row to be added to negative or positive labels depending on label
        row = []
        row.append(file)
        row.append(label)

        if label == 0:
            negative_labels.append(row)
        else:
            positive_labels.append(row)

    # If either no positive or negative labels then data is not balanceable
    if len(negative_labels) == 0 or len(positive_labels) == 0:
        raise Exception("Unable to balance labels since either positive or negative labels do not exist")

    # Perform the balancing depending on which has more labels
    if len(positive_labels) <= len(negative_labels):

        # The basic idea will be to generate repeats in positive labels to match the length of
        # negative labels
        # To avoid biasing this process we first scramble the positive labels
        random.shuffle(positive_labels)

        # Repeat the elements of the positive labels
        # Note this may cause the length to exceed that of negative labels which is why we need the last line
        num_repetitions = int(np.ceil(len(negative_labels) / len(positive_labels)))
        positive_labels_extended = []
        for i in range(0, num_repetitions):
            for j in range(0, len(positive_labels)):
                row = []
                row.append(positive_labels[j][0])
                row.append(positive_labels[j][1])
                positive_labels_extended.append(row)
        positive_labels = positive_labels_extended[0:len(negative_labels)]

    else:

        # The basic idea will be to generate repeats in negative labels to match the length of
        # positive labels
        # To avoid biasing this process we first scramble the negative labels
        random.shuffle(negative_labels)

        # Repeat the elements of the negative labels
        # Note this may cause the length to exceed that of positive labels which is why we need the last line
        num_repetitions = int(np.ceil(len(positive_labels) / len(negative_labels)))
        negative_labels_extended = []
        for i in range(0, num_repetitions):
            for j in range(0, len(negative_labels)):
                row = []
                row.append(negative_labels[j][0])
                row.append(negative_labels[j][1])
                negative_labels_extended.append(row)
        negative_labels = negative_labels_extended[0:len(positive_labels)]

    # Now merge everything back into one list and convert to data frame
    labels = []
    for i in range(0, len(positive_labels)):
        labels.append(positive_labels[i])
    for i in range(0, len(negative_labels)):
        labels.append(negative_labels[i])

    # Do one final shuffle to avoid bias
    random.shuffle(labels)

    return pd.DataFrame(labels)


# Test code
#balance_labels(pd.read_csv('train.txt', header=None))