# -------------------------------------------------------------------------
# AUTHOR: Siwen Wang
# FILENAME: decision_tree.py
# SPECIFICATION: Read the files contact_lens_training_1.csv, contact_lens_training_2.csv, and contact_lens_training_3.csv,
#                choosing the lowest accuracy as the classification performance of each model.
# FOR: CS 4200- Assignment #2
# TIME SPENT: 1 hour and 30 minutes
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

# importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']
for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    # reading the training data in a csv file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0:  # skipping the header
                dbTraining.append(row)

    # transform the original training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    # --> add your Python code here
    # X =
    dict = {"Young": 1, "Presbyopic": 2, "Prepresbyopic": 3,
            "Myope": 1, "Hypermetrope": 2,
            "No": 1, "Yes": 2,
            "Reduced": 1, "Normal": 2}
    for row in dbTraining:
        temp = []
        for i in range(len(row) - 1):  # skipping the last column
            temp.append(dict.get(row[i]))
        X.append(temp)

    # transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    # --> add your Python code here
    # Y =
    dict2 = {"No": 1, "Yes": 2}
    for row in dbTraining:
        Y.append(dict2.get(row[len(row) - 1]))

    accuracy = []
    totalAccuracy = []
    # loop your training and test tasks 10 times here
    for i in range(10):

        # fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
        clf = clf.fit(X, Y)

        # read the test data and add this data to dbTest
        # --> add your Python code here
        # dbTest =
        testData = []
        dbTest = []
        with open('contact_lens_test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i > 0:  # skipping the header
                    testData.append(row)

        for row in testData:
            temp = []
            for j in range(len(row)):  # skipping the last column
                temp.append(dict.get(row[j]))
            dbTest.append(temp)

        # transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
        # class_predicted = clf.predict([[3, 1, 2, 1]])[0]           -> [0] is used to get an integer as the predicted class label so that you can compare it with the true label
        # --> add your Python code here
        for_prediction = []
        correct = 0
        counter = 0
        for data in dbTest:
            temp = data[0:4]
            for_prediction.append(temp)
            class_predicted = clf.predict(for_prediction)
            if class_predicted[counter] == data[4]:
                correct += 1
            counter += 1

        # compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
        accuracy.append(correct/counter)

        # find the lowest accuracy of this model during the 10 runs (training and test set)
        # --> add your Python code here
        minimal = min(accuracy)
        totalAccuracy.append(minimal)

    # print the lowest accuracy of this model during the 10 runs (training and test set).
    # your output should be something like that:
    # final accuracy when training on contact_lens_training_1.csv: 0.2
    # final accuracy when training on contact_lens_training_2.csv: 0.3
    # final accuracy when training on contact_lens_training_3.csv: 0.4
    print("final accuracy when training on" + ds + ":" + str(min(totalAccuracy)))
