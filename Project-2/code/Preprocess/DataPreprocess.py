import csv
import pandas as pd
import math

# Preproccing all the data and creating the daataset for seen, unseen, and
# shuffled data pairs.

def processDiffPairs(readFile, writeFile):
    iterator = 0
    img1 = []
    img2 = []
    target = []
    i_update = 301
    j_update = 301
    count = 0
    flag = 0
    reader = csv.reader(open(readFile), delimiter=',')
    for row in reader:
        iterator += 1
        if (iterator > 1):
            i = int(row[0][:4])
            j = int(row[1][:4])
            if ((i > i_update and j < j_update) or (i == i_update and j > j_update)):
                if(i > i_update):
                    flag = 0
                if(flag ==0):
                    img1.append(row[0])
                    img2.append(row[1])
                    target.append(row[2])
                if(i_update == i and flag == 0):
                    count += 1
                if(count == 1):
                    count = 0
                    flag = 1
                i_update = i
                j_update = j
    dataset = {}
    dataset["img_id_A"] = img1
    dataset["img_id_B"] = img2
    dataset["target"] = target

    # Writing to csv
    pd.DataFrame(dataset).to_csv(writeFile, index=False)
    print("Different pairs file truncated.")


def completeData(firstFile, secondFile, writeFile):
    a = pd.read_csv(firstFile)
    b = pd.read_csv(secondFile)
    merged = a.append(b)
    ds = merged.sample(frac=1)
    ds.to_csv(writeFile, index=False)
    print("The different pairs and same pairs mixed and shuffled.")

def combining_feature_data(pairFile, featureFile, operation, outFile):
    iterator = 0
    reader = csv.reader(open(pairFile), delimiter=',')
    row_count = sum(1 for row in reader)
    final_row = [0] * (row_count-1)
    reader = csv.reader(open(pairFile), delimiter=',')
    for row in reader:
        iterator += 1
        if (iterator > 1):
            reader1 = csv.reader(open(featureFile), delimiter=',')
            for row1 in reader1:
                if(row1[0] == row[0]):
                    a = row1[1:]
                if(row1[0] == row[1]):
                    b = row1[1:]
            if(operation == "concat"):
                list = a + b
            elif(operation == "sub"):
                list = [0] * len(a)
                for i in range(len(a)):
                    list[i] = str(int(a[i]) - int(b[i]))
            final_row[iterator-2] = [row[0]] + [row[1]] + list + [row[2]]
    field_names = [0] * len(final_row[0])
    for i in range(len(field_names)):
        field_names[i] = "f" + str(i-1)
    field_names[0] = "img_id_A"
    field_names[1] = "img_id_B"
    field_names[-1] = "target"
    final_row = pd.DataFrame(final_row, columns = field_names)
    final_row.to_csv(outFile, index=False)
    print("Features also added to total data.")


def dataset_creator_unseen(fileName, setName, start, end):
    iterator = 0
    img1 = []

    reader = csv.reader(open(fileName), delimiter=',')
    for row in reader:
        iterator += 1
        if (iterator > 1):
            i = int(row[0][:4])
            j = int(row[1][:4])
            if (i < end and j < end and i > start and j > start):
                img1.append(row)
    field_names = [0] * len(img1[0])
    for i in range(len(img1[0])):
        field_names[i] = "f" + str(i-1)
    field_names[0] = "img_id_A"
    field_names[1] = "img_id_B"
    field_names[-1] = "target"
    final_row = pd.DataFrame(img1, columns = field_names)
    final_row.to_csv(setName, index=False)
    print("Unseen data created for" + setName)

def image_counter(fileName, startpercentage, endpercentage):
    iterator = 0
    imgcount = [0] * 1569
    start = [0] * 1569
    end = [0] * 1569
    reader = csv.reader(open(fileName), delimiter=',')
    for row in reader:
        iterator += 1
        if (iterator > 1):
            i = int(row[0][:4])
            j = int(row[1][:4])
            imgcount[i] += 1
            imgcount[j] += 1
    for i in range(len(imgcount)):
        start[i] = math.floor(imgcount[i]*startpercentage*0.01)
        end [i] =  math.floor(imgcount[i]*endpercentage*0.01)
    return start, end

def dataset_creator_seen(fileName, setName, startpercentage, endpercentage):
    iterator = 0
    img1 = []
    imgcount = [0] * 1569
    start, end = image_counter(fileName, startpercentage, endpercentage)
    reader = csv.reader(open(fileName), delimiter=',')
    for row in reader:
        iterator += 1
        if (iterator > 1):
            i = int(row[0][:4])
            j = int(row[1][:4])
            imgcount[i] += 1
            imgcount[j] += 1
            if(imgcount[i] > start[i] and imgcount[j] > start[j] and
            imgcount[i] <= end[i] and imgcount[j] <= end[j]):
                img1.append(row)
    field_names = [0] * len(img1[0])
    for i in range(len(img1[0])):
        field_names[i] = "f" + str(i-1)
    field_names[0] = "img_id_A"
    field_names[1] = "img_id_B"
    field_names[-1] = "target"
    final_row = pd.DataFrame(img1, columns=field_names)
    final_row.to_csv(setName, index=False)
    print("Seen data created for" + setName)


def dataset_creator_shuffled(fileName, setName, startpercentage, endpercentage):
    iterator = 0
    img1 = []
    reader = csv.reader(open(fileName), delimiter=',')
    row_count = sum(1 for row in reader)
    reader = csv.reader(open(fileName), delimiter=',')
    for row in reader:
        iterator += 1
        if (iterator > 1 and iterator > startpercentage*0.01*row_count and
            iterator < endpercentage*0.01*row_count):
                img1.append(row)
    field_names = [0] * len(img1[0])

    for i in range(len(img1[0])):
        field_names[i] = "f" + str(i-1)
    field_names[0] = "img_id_A"
    field_names[1] = "img_id_B"
    field_names[-1] = "target"
    final_row = pd.DataFrame(img1, columns=field_names)
    final_row.to_csv(setName, index=False)
    print("Shuffled data created for" + setName)


processDiffPairs("diffn_pairs.csv", "diff_pairs_edited.csv")
completeData("same_pairs.csv", "diff_pairs_edited.csv", "total_dataset.csv")
combining_feature_data("total_dataset.csv", "HumanObserved-Features-Data.csv", "sub", "total_dataset_features_sub.csv")
dataset_creator_unseen("total_dataset_features_sub.csv", "unseen_training_sub.csv", 1, 1375)
dataset_creator_unseen("total_dataset_features_sub.csv", "unseen_testing_sub.csv", 1376, 1499)
dataset_creator_unseen("total_dataset_features_sub.csv", "unseen_validation_sub.csv", 1500, 1569)
dataset_creator_seen("total_dataset_features_sub.csv", "seen_training_sub.csv", 0, 80)
dataset_creator_seen("total_dataset_features_sub.csv", "seen_testing_sub.csv", 80, 90)
dataset_creator_seen("total_dataset_features_sub.csv", "seen_validation_sub.csv", 90, 100)
dataset_creator_shuffled("total_dataset_features_sub.csv", "shuffled_training_sub.csv", 0, 80)
dataset_creator_shuffled("total_dataset_features_sub.csv", "shuffled_testing_sub.csv", 80, 90)
dataset_creator_shuffled("total_dataset_features_sub.csv", "shuffled_validation_sub.csv", 90, 100)
combining_feature_data("total_dataset.csv", "HumanObserved-Features-Data.csv", "concat", "total_dataset_features_concat.csv")
dataset_creator_unseen("total_dataset_features_concat.csv", "unseen_training_concat.csv", 1, 1375)
dataset_creator_unseen("total_dataset_features_concat.csv", "unseen_testing_concat.csv", 1376, 1499)
dataset_creator_unseen("total_dataset_features_concat.csv", "unseen_validation_concat.csv", 1500, 1569)
dataset_creator_seen("total_dataset_features_concat.csv", "seen_training_concat.csv", 0, 80)
dataset_creator_seen("total_dataset_features_concat.csv", "seen_testing_concat.csv", 80, 90)
dataset_creator_seen("total_dataset_features_concat.csv", "seen_validation_concat.csv", 90, 100)
dataset_creator_shuffled("total_dataset_features_concat.csv", "shuffled_training_concat.csv", 0, 80)
dataset_creator_shuffled("total_dataset_features_concat.csv", "shuffled_testing_concat.csv", 80, 90)
dataset_creator_shuffled("total_dataset_features_concat.csv", "shuffled_validation_concat.csv", 90, 100)

