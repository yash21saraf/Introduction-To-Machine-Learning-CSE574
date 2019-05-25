import csv
import pandas as pd
import math


# Used the file to create the same and different pair for
# the HumanObservedFeatures set as the numbers were not in
# ascending order.
# Tried to take in data in as balanced way as possible.
def processDiffPairs(readFile, writeFile):
    iterator = 0
    img1 = []
    img2 = []
    imgfin1 = []
    imgfin2 = []
    imgfin11 = []
    imgfin21= []
    tar1 = []
    tar2 = []
    reader = csv.reader(open(readFile), delimiter=',')
    for row in reader:
        iterator += 1
        if (iterator > 1):
            img1.append(row[0])
            img2.append(row[1])
    for i in range(len(img1)):
        for j in range(i, len(img2)):
            a = int(img1[i][:4])
            b = int(img2[j][:4])
            if(a == b and not(img1[i] == img2[j])):
                imgfin1.append(img1[i])
                imgfin2.append(img2[j])
                tar1.append("1")
            if(not(a == b)):
                imgfin11.append(img1[i])
                imgfin21.append(img2[j])
                tar2.append("0")

    dataset1 = {}
    dataset1["img_id_A"] = imgfin1
    dataset1["img_id_B"] = imgfin2
    dataset1["target"] = tar1

    dataset2 = {}
    dataset2["img_id_A"] = imgfin11
    dataset2["img_id_B"] = imgfin21
    dataset2["target"] = tar2

    # Writing to csv
    pd.DataFrame(dataset1).to_csv("same_pairs.csv", index=False)
    # Writing to csv
    pd.DataFrame(dataset2).to_csv(writeFile, index=False)
    print("Different pairs file truncated.")

processDiffPairs("AllPairs.csv", "diff_pairs_edited.csv")