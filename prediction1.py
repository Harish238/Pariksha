import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

#reading the files
train = pd.read_csv("/home/harish/Downloads/train.csv")
test = pd.read_csv("/home/harish/Downloads/test.csv")
total_scores = pd.read_csv("/home/harish/Downloads/test_totals.csv")
train.replace(to_replace = "?", value = 0, inplace = True) # replacing all "?"s with 0 because "?" is not recognized as a missing value

# arranging the test number and their total scores in a dictionary which will be used later to calculate percentage
marks = total_scores["Max Marks"]
totals = []
for i in range(0, 27, 3) :
    totals.append(marks[i] + marks[i + 1] + marks[i + 2])
totals.append(marks[27] + marks[28] + marks[29])
x = list(total_scores["Test Id"])
test_no = []
for i in range(0, len(x), 3):
    test_no.append(x[i])
test_total_scores = {}
for i in range(len(totals)):
    test_total_scores[test_no[i]] = totals[i]

# to calculate percentages for each test
def percentage_calc(testnum, set):
    str0 = "Test_" + str(testnum)
    str1 = "test_" + str(testnum) + "_sub_score_54498"
    str2 = "test_" + str(testnum) + "_sub_score_54499"
    str3 = "test_" + str(testnum) + "_sub_score_54500"
    sub1 = list(set[str1])
    sub2 = list(set[str2])
    sub3 = list(set[str3])
    perc = []
    for i in range(len(sub1)):
        perc.append((int(sub1[i]) + int(sub2[i]) + int(sub3[i])) / int(test_total_scores[str0]))
    return perc
percentages1 = []
for i in range(1,11):
    percentages1.append(percentage_calc(i, train))

# initializing dependant and independant variables
X = np.column_stack((percentages1[0], percentages1[1], percentages1[2], percentages1[3], percentages1[4], percentages1[5], percentages1[6], percentages1[7], percentages1[8]))
y = np.asarray(percentages1[9])
y = y * test_total_scores["Test_10"]

#training phase
lm = LinearRegression()
lm.fit(X, y)

#testing phase
test.replace(to_replace = "?", value = 0, inplace = True) # replacing all "?"s with 0 because "?" is not recognized as a missing value
percentages2 = []
for i in range(1,10):
    percentages2.append(percentage_calc(i, test))
X_predict = np.column_stack((percentages2[0], percentages2[1], percentages2[2], percentages2[3], percentages2[4], percentages2[5], percentages2[6], percentages2[7], percentages2[8]))

#prediction
y_predict = lm.predict(X_predict)
np.savetxt("prediction1.csv", y_predict, delimiter=",")