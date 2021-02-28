#-------------------------------------------------------------------------
# AUTHOR: Siwen Wang
# FILENAME: naive_bayes.py
# SPECIFICATION: Read the file weather_training.csv (training set) and output the classification of each test instance
#                from the file weather_test (test set) if the classification confidence is >= 0.75
# FOR: CS 4200- Assignment #2
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
import csv
from sklearn.naive_bayes import GaussianNB

#reading the training data
db = []
with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            db.append(row[1:])

#transform the original training features to numbers and add to the 4D array X. For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
db_dict = {"Sunny": 1, "Overcast": 2, "Rain":3,
           "Hot": 1, "Mild": 2, "Cool": 3,
           "High": 1, "Normal": 2,
           "Weak": 1, "Strong": 2,
           "No": 1, "Yes": 2}

X = []
for row in db:
    temp = []
    for i in range(len(row) - 1):  # skipping the last column
        temp.append(db_dict.get(row[i]))
    X.append(temp)

#transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
Y = []
for row in db:
    Y.append(db_dict.get(row[len(row) - 1]))


#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the data in a csv file
dbTest = []  # String version
with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            dbTest.append(row[1:len(row)-1])  # Skipping the last column -- question mark

dbTestNum = []  # int version
for row in dbTest:
    temp = []
    for col in row:
        temp.append(db_dict.get(col))
    dbTestNum.append(temp)


result = clf.predict_proba(dbTestNum)

#printing the header os the solution
print("Day".ljust(15) + "Outlook".ljust(15) + "Temperature".ljust(15) + "Humidity".ljust(15) + "Wind".ljust(15) + "PlayTennis".ljust(15) + "Confidence".ljust(15))

#use your test samples to make probabilistic predictions.
#--> add your Python code here
#-->predicted = clf.predict_proba([[3, 1, 2, 1]])[0]
with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:
            for col in range(len(row)-1):
                print(row[col].ljust(15), end="")
            if result[i-1][0] > result[i-1][1]:
                print("No".ljust(15) + str(format(result[i-1][0], '.2f')).ljust(15))
            else:
                print("Yes".ljust(15) + str(format(result[i-1][1], '.2f')).ljust(15))
