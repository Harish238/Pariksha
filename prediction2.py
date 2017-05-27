import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Imputer

#reading the files
train = pd.read_csv("/home/harish/Downloads/train.csv")
test = pd.read_csv("/home/harish/Downloads/test.csv")
total_scores = pd.read_csv("/home/harish/Downloads/test_totals.csv")
train.replace(to_replace = "?", value = np.nan, inplace = True) # replacing all "?"s with NaN so we can impute the missing values with an appropriate value
train_subset = train[["test_1_sub_score_54498", "test_1_sub_score_54499", "test_1_sub_score_54500", "test_2_sub_score_54498", "test_2_sub_score_54499", "test_2_sub_score_54500", "test_3_sub_score_54498", "test_3_sub_score_54499", "test_3_sub_score_54500", "test_4_sub_score_54498", "test_4_sub_score_54499", "test_4_sub_score_54500", "test_5_sub_score_54498", "test_5_sub_score_54499", "test_5_sub_score_54500", "test_6_sub_score_54498", "test_6_sub_score_54499", "test_6_sub_score_54500", "test_7_sub_score_54498", "test_7_sub_score_54499", "test_7_sub_score_54500", "test_8_sub_score_54498", "test_8_sub_score_54499", "test_8_sub_score_54500", "test_9_sub_score_54498", "test_9_sub_score_54499", "test_9_sub_score_54500", "test_10_sub_score_54498", "test_10_sub_score_54499", "test_10_sub_score_54500"]]
imr1 = Imputer(missing_values = "NaN", strategy = "mean", axis = 1)
imr1.fit(train_subset)
train_new = imr1.transform(train_subset)
train_imputed = pd.DataFrame(data = train_new, columns = ["test_1_sub_score_54498", "test_1_sub_score_54499", "test_1_sub_score_54500", "test_2_sub_score_54498", "test_2_sub_score_54499", "test_2_sub_score_54500", "test_3_sub_score_54498", "test_3_sub_score_54499", "test_3_sub_score_54500", "test_4_sub_score_54498", "test_4_sub_score_54499", "test_4_sub_score_54500", "test_5_sub_score_54498", "test_5_sub_score_54499", "test_5_sub_score_54500", "test_6_sub_score_54498", "test_6_sub_score_54499", "test_6_sub_score_54500", "test_7_sub_score_54498", "test_7_sub_score_54499", "test_7_sub_score_54500", "test_8_sub_score_54498", "test_8_sub_score_54499", "test_8_sub_score_54500", "test_9_sub_score_54498", "test_9_sub_score_54499", "test_9_sub_score_54500", "test_10_sub_score_54498", "test_10_sub_score_54499", "test_10_sub_score_54500"])

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
    percentages1.append(percentage_calc(i, train_imputed))

# initializing dependant and independant variables
X = np.column_stack((percentages1[0], percentages1[1], percentages1[2], percentages1[3], percentages1[4], percentages1[5], percentages1[6], percentages1[7], percentages1[8]))
y = np.asarray(percentages1[9])
y = y * test_total_scores["Test_10"]

# training phase
lm = LinearRegression()
lm.fit(X, y)

# testing phase
test.replace(to_replace = "?", value = np.nan, inplace = True) # replacing all "?"s with NaN so we can impute the missing values with an appropriate value

# imputing missing values for the test data
test_subset = test[["test_1_sub_score_54498", "test_1_sub_score_54499", "test_1_sub_score_54500", "test_2_sub_score_54498", "test_2_sub_score_54499", "test_2_sub_score_54500", "test_3_sub_score_54498", "test_3_sub_score_54499", "test_3_sub_score_54500", "test_4_sub_score_54498", "test_4_sub_score_54499", "test_4_sub_score_54500", "test_5_sub_score_54498", "test_5_sub_score_54499", "test_5_sub_score_54500", "test_6_sub_score_54498", "test_6_sub_score_54499", "test_6_sub_score_54500", "test_7_sub_score_54498", "test_7_sub_score_54499", "test_7_sub_score_54500", "test_8_sub_score_54498", "test_8_sub_score_54499", "test_8_sub_score_54500", "test_9_sub_score_54498", "test_9_sub_score_54499", "test_9_sub_score_54500", "test_10_sub_score_54498", "test_10_sub_score_54499", "test_10_sub_score_54500"]]
imr2 = Imputer(missing_values = "NaN", strategy = "mean", axis = 1)
imr2.fit(test_subset)
test_new = imr2.transform(test_subset)
test_imputed = pd.DataFrame(data = test_new, columns = ["test_1_sub_score_54498", "test_1_sub_score_54499", "test_1_sub_score_54500", "test_2_sub_score_54498", "test_2_sub_score_54499", "test_2_sub_score_54500", "test_3_sub_score_54498", "test_3_sub_score_54499", "test_3_sub_score_54500", "test_4_sub_score_54498", "test_4_sub_score_54499", "test_4_sub_score_54500", "test_5_sub_score_54498", "test_5_sub_score_54499", "test_5_sub_score_54500", "test_6_sub_score_54498", "test_6_sub_score_54499", "test_6_sub_score_54500", "test_7_sub_score_54498", "test_7_sub_score_54499", "test_7_sub_score_54500", "test_8_sub_score_54498", "test_8_sub_score_54499", "test_8_sub_score_54500", "test_9_sub_score_54498", "test_9_sub_score_54499", "test_9_sub_score_54500", "test_10_sub_score_54498", "test_10_sub_score_54499", "test_10_sub_score_54500"])
percentages2 = []
for i in range(1,10):
    percentages2.append(percentage_calc(i, test_imputed))
X_predict = np.column_stack((percentages2[0], percentages2[1], percentages2[2], percentages2[3], percentages2[4], percentages2[5], percentages2[6], percentages2[7], percentages2[8]))

#prediction
y_predict = lm.predict(X_predict)
np.savetxt("prediction2.csv", y_predict, delimiter=",")