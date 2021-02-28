# -------------------------------------------------------------------------
# AUTHOR: Siwen Wang
# FILENAME: knn.py
# SPECIFICATION: Read the file binary_points.csv and output the LOO-CV error rate for 1NN
# FOR: CS 4200- Assignment #2
# TIME SPENT: 1 hour
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

# importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

# reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            db.append(row)


class_dict = {"-": 1, "+": 2}
size = len(db)

# 9 instance for each iteration, the rest will be the test case
counter = size-1
wrongPrediction = 0

# loop your data to allow each instance to be your test set
for i, instance in enumerate(db):
    # add the training features to the 2D array X removing the instance that will be used for testing in this iteration. For instance, X = [[1, 3], [2, 1,], ...]]
    # transform the original training classes to numbers and add to the vector Y removing the instance that will be used for testing in this iteration. For instance, Y = [1, 2, ,...]
    # --> add your Python code here
    X = []
    Y = []
    count = i
    for _ in range(counter):
        temp = list(map(int, db[count % size][:2]))
        X.append(temp)
        Y.append(class_dict.get(db[count % size][2]))
        count += 1

    # store the test sample of this iteration in the vector testSample
    # --> add your Python code here
    if i == 0:
        testCounter = size - 1
    else:
        testCounter = i - 1
    testSample = db[testCounter][0:2]
    testSample = list(map(int, testSample))
    testSample.append(class_dict.get(db[testCounter][2]))

    # fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    # use your test sample in this iteration to make the class prediction. For instance:
    # class_predicted = clf.predict([[1, 2]])[0]

    class_predicted = clf.predict([[testSample[0], testSample[1]]])[0]

    if class_predicted != testSample[2]:
        wrongPrediction += 1

    # compare the prediction with the true label of the test instance to start calculating the error rate.
    # --> add your Python code here

print("The error rate is: " + str(wrongPrediction/size))
# print the error rate
# --> add your Python code here
