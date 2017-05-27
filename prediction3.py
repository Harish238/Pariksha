import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Imputer

# reading the files
train = pd.read_csv("/home/harish/Downloads/train.csv")
test = pd.read_csv("/home/harish/Downloads/test.csv")
total_scores = pd.read_csv("/home/harish/Downloads/test_totals.csv")

# imputing missing values
train.replace(to_replace = "?", value = np.nan, inplace = True) # replacing all "?"s with NaN so we can impute the missing values with an appropriate value
train_subset1 = train[["test_1_sub_score_54498", "test_1_sub_score_54499", "test_1_sub_score_54500", "test_2_sub_score_54498", "test_2_sub_score_54499", "test_2_sub_score_54500", "test_3_sub_score_54498", "test_3_sub_score_54499", "test_3_sub_score_54500", "test_4_sub_score_54498", "test_4_sub_score_54499", "test_4_sub_score_54500", "test_5_sub_score_54498", "test_5_sub_score_54499", "test_5_sub_score_54500", "test_6_sub_score_54498", "test_6_sub_score_54499", "test_6_sub_score_54500", "test_7_sub_score_54498", "test_7_sub_score_54499", "test_7_sub_score_54500", "test_8_sub_score_54498", "test_8_sub_score_54499", "test_8_sub_score_54500", "test_9_sub_score_54498", "test_9_sub_score_54499", "test_9_sub_score_54500", "test_10_sub_score_54498", "test_10_sub_score_54499", "test_10_sub_score_54500"]]
train_subset2 = train[["test_1_correct_q_54498", "test_1_correct_q_54499", "test_1_correct_q_54500", "test_2_correct_q_54498", "test_2_correct_q_54499", "test_2_correct_q_54500", "test_3_correct_q_54498", "test_3_correct_q_54499", "test_3_correct_q_54500", "test_4_correct_q_54498", "test_4_correct_q_54499", "test_4_correct_q_54500", "test_5_correct_q_54498", "test_5_correct_q_54499", "test_5_correct_q_54500", "test_6_correct_q_54498", "test_6_correct_q_54499", "test_6_correct_q_54500", "test_7_correct_q_54498", "test_7_correct_q_54499", "test_7_correct_q_54500", "test_8_correct_q_54498", "test_8_correct_q_54499", "test_8_correct_q_54500", "test_9_correct_q_54498", "test_9_correct_q_54499", "test_9_correct_q_54500", "test_10_correct_q_54498", "test_10_correct_q_54499", "test_10_correct_q_54500"]]
train_subset3 = train[["test_1_incorrect_q_54498", "test_1_incorrect_q_54499", "test_1_incorrect_q_54500", "test_2_incorrect_q_54498", "test_2_incorrect_q_54499", "test_2_incorrect_q_54500", "test_3_incorrect_q_54498", "test_3_incorrect_q_54499", "test_3_incorrect_q_54500", "test_4_incorrect_q_54498", "test_4_incorrect_q_54499", "test_4_incorrect_q_54500", "test_5_incorrect_q_54498", "test_5_incorrect_q_54499", "test_5_incorrect_q_54500", "test_6_incorrect_q_54498", "test_6_incorrect_q_54499", "test_6_incorrect_q_54500", "test_7_incorrect_q_54498", "test_7_incorrect_q_54499", "test_7_incorrect_q_54500", "test_8_incorrect_q_54498", "test_8_incorrect_q_54499", "test_8_incorrect_q_54500", "test_9_incorrect_q_54498", "test_9_incorrect_q_54499", "test_9_incorrect_q_54500", "test_10_incorrect_q_54498", "test_10_incorrect_q_54499", "test_10_incorrect_q_54500"]]
imr1 = Imputer(missing_values = "NaN", strategy = "mean", axis = 1)
imr1.fit(train_subset1)
train_new1 = imr1.transform(train_subset1)
train_imputed1 = pd.DataFrame(data = train_new1, columns = ["test_1_sub_score_54498", "test_1_sub_score_54499", "test_1_sub_score_54500", "test_2_sub_score_54498", "test_2_sub_score_54499", "test_2_sub_score_54500", "test_3_sub_score_54498", "test_3_sub_score_54499", "test_3_sub_score_54500", "test_4_sub_score_54498", "test_4_sub_score_54499", "test_4_sub_score_54500", "test_5_sub_score_54498", "test_5_sub_score_54499", "test_5_sub_score_54500", "test_6_sub_score_54498", "test_6_sub_score_54499", "test_6_sub_score_54500", "test_7_sub_score_54498", "test_7_sub_score_54499", "test_7_sub_score_54500", "test_8_sub_score_54498", "test_8_sub_score_54499", "test_8_sub_score_54500", "test_9_sub_score_54498", "test_9_sub_score_54499", "test_9_sub_score_54500", "test_10_sub_score_54498", "test_10_sub_score_54499", "test_10_sub_score_54500"])
imr2 = Imputer(missing_values = "NaN", strategy = "mean", axis = 1)
imr2.fit(train_subset2)
train_new2 = imr2.transform(train_subset2)
train_imputed2 = pd.DataFrame(data = train_new2, columns = ["test_1_correct_q_54498", "test_1_correct_q_54499", "test_1_correct_q_54500", "test_2_correct_q_54498", "test_2_correct_q_54499", "test_2_correct_q_54500", "test_3_correct_q_54498", "test_3_correct_q_54499", "test_3_correct_q_54500", "test_4_correct_q_54498", "test_4_correct_q_54499", "test_4_correct_q_54500", "test_5_correct_q_54498", "test_5_correct_q_54499", "test_5_correct_q_54500", "test_6_correct_q_54498", "test_6_correct_q_54499", "test_6_correct_q_54500", "test_7_correct_q_54498", "test_7_correct_q_54499", "test_7_correct_q_54500", "test_8_correct_q_54498", "test_8_correct_q_54499", "test_8_correct_q_54500", "test_9_correct_q_54498", "test_9_correct_q_54499", "test_9_correct_q_54500", "test_10_correct_q_54498", "test_10_correct_q_54499", "test_10_correct_q_54500"])
imr3 = Imputer(missing_values = "NaN", strategy = "mean", axis = 1)
imr3.fit(train_subset3)
train_new3 = imr3.transform(train_subset3)
train_imputed3 = pd.DataFrame(data = train_new3, columns = ["test_1_incorrect_q_54498", "test_1_incorrect_q_54499", "test_1_incorrect_q_54500", "test_2_incorrect_q_54498", "test_2_incorrect_q_54499", "test_2_incorrect_q_54500", "test_3_incorrect_q_54498", "test_3_incorrect_q_54499", "test_3_incorrect_q_54500", "test_4_incorrect_q_54498", "test_4_incorrect_q_54499", "test_4_incorrect_q_54500", "test_5_incorrect_q_54498", "test_5_incorrect_q_54499", "test_5_incorrect_q_54500", "test_6_incorrect_q_54498", "test_6_incorrect_q_54499", "test_6_incorrect_q_54500", "test_7_incorrect_q_54498", "test_7_incorrect_q_54499", "test_7_incorrect_q_54500", "test_8_incorrect_q_54498", "test_8_incorrect_q_54499", "test_8_incorrect_q_54500", "test_9_incorrect_q_54498", "test_9_incorrect_q_54499", "test_9_incorrect_q_54500", "test_10_incorrect_q_54498", "test_10_incorrect_q_54499", "test_10_incorrect_q_54500"])

# combining the dataframes
train_imputed = pd.concat([train_imputed1, train_imputed2, train_imputed3], axis = 1)

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

# to calculate total correct questions
def correct_calc(testnum, set):
    str1 = "test_" + str(testnum) + "_correct_q_54498"
    str2 = "test_" + str(testnum) + "_correct_q_54499"
    str3 = "test_" + str(testnum) + "_correct_q_54500"
    sub1 = list(set[str1])
    sub2 = list(set[str2])
    sub3 = list(set[str3])
    sum = []
    for i in range(len(sub1)):
        sum.append((int(sub1[i]) + int(sub2[i]) + int(sub3[i])))
    return sum

# to calculate total incorrect questions
def incorrect_calc(testnum, set):
    str1 = "test_" + str(testnum) + "_incorrect_q_54498"
    str2 = "test_" + str(testnum) + "_incorrect_q_54499"
    str3 = "test_" + str(testnum) + "_incorrect_q_54500"
    sub1 = list(set[str1])
    sub2 = list(set[str2])
    sub3 = list(set[str3])
    sum = []
    for i in range(len(sub1)):
        sum.append((int(sub1[i]) + int(sub2[i]) + int(sub3[i])))
    return sum

percentages1 = []
for i in range(1,11):
    percentages1.append(percentage_calc(i, train_imputed))
correct1 = []
for i in range(1,10):
    correct1.append(correct_calc(i, train_imputed))
incorrect1 = []
for i in range(1,10):
    incorrect1.append(incorrect_calc(i, train_imputed))

# initializing dependant and independant variables
X1 = (np.array(percentages1[0:9])).transpose()
X2 = (np.array(correct1[0:9])).transpose()
X3 = (np.array(incorrect1[0:9])).transpose()
X = np.column_stack((X1, X2, X3))
y = np.asarray(percentages1[9])
y = y * test_total_scores["Test_10"]

# training phase
lm = LinearRegression()
lm.fit(X, y)

# testing phase
test.replace(to_replace = "?", value = np.nan, inplace = True) # replacing all "?"s with NaN so we can impute the missing values with an appropriate value

# imputing missing values for the test data
test.replace(to_replace = "?", value = np.nan, inplace = True) # replacing all "?"s with NaN so we can impute the missing values with an appropriate value
test_subset1 = test[["test_1_sub_score_54498", "test_1_sub_score_54499", "test_1_sub_score_54500", "test_2_sub_score_54498", "test_2_sub_score_54499", "test_2_sub_score_54500", "test_3_sub_score_54498", "test_3_sub_score_54499", "test_3_sub_score_54500", "test_4_sub_score_54498", "test_4_sub_score_54499", "test_4_sub_score_54500", "test_5_sub_score_54498", "test_5_sub_score_54499", "test_5_sub_score_54500", "test_6_sub_score_54498", "test_6_sub_score_54499", "test_6_sub_score_54500", "test_7_sub_score_54498", "test_7_sub_score_54499", "test_7_sub_score_54500", "test_8_sub_score_54498", "test_8_sub_score_54499", "test_8_sub_score_54500", "test_9_sub_score_54498", "test_9_sub_score_54499", "test_9_sub_score_54500", "test_10_sub_score_54498", "test_10_sub_score_54499", "test_10_sub_score_54500"]]
test_subset2 = test[["test_1_correct_q_54498", "test_1_correct_q_54499", "test_1_correct_q_54500", "test_2_correct_q_54498", "test_2_correct_q_54499", "test_2_correct_q_54500", "test_3_correct_q_54498", "test_3_correct_q_54499", "test_3_correct_q_54500", "test_4_correct_q_54498", "test_4_correct_q_54499", "test_4_correct_q_54500", "test_5_correct_q_54498", "test_5_correct_q_54499", "test_5_correct_q_54500", "test_6_correct_q_54498", "test_6_correct_q_54499", "test_6_correct_q_54500", "test_7_correct_q_54498", "test_7_correct_q_54499", "test_7_correct_q_54500", "test_8_correct_q_54498", "test_8_correct_q_54499", "test_8_correct_q_54500", "test_9_correct_q_54498", "test_9_correct_q_54499", "test_9_correct_q_54500", "test_10_correct_q_54498", "test_10_correct_q_54499", "test_10_correct_q_54500"]]
test_subset3 = test[["test_1_incorrect_q_54498", "test_1_incorrect_q_54499", "test_1_incorrect_q_54500", "test_2_incorrect_q_54498", "test_2_incorrect_q_54499", "test_2_incorrect_q_54500", "test_3_incorrect_q_54498", "test_3_incorrect_q_54499", "test_3_incorrect_q_54500", "test_4_incorrect_q_54498", "test_4_incorrect_q_54499", "test_4_incorrect_q_54500", "test_5_incorrect_q_54498", "test_5_incorrect_q_54499", "test_5_incorrect_q_54500", "test_6_incorrect_q_54498", "test_6_incorrect_q_54499", "test_6_incorrect_q_54500", "test_7_incorrect_q_54498", "test_7_incorrect_q_54499", "test_7_incorrect_q_54500", "test_8_incorrect_q_54498", "test_8_incorrect_q_54499", "test_8_incorrect_q_54500", "test_9_incorrect_q_54498", "test_9_incorrect_q_54499", "test_9_incorrect_q_54500", "test_10_incorrect_q_54498", "test_10_incorrect_q_54499", "test_10_incorrect_q_54500"]]
imr10 = Imputer(missing_values = "NaN", strategy = "mean", axis = 1)
imr10.fit(test_subset1)
test_new1 = imr10.transform(test_subset1)
test_imputed1 = pd.DataFrame(data = test_new1, columns = ["test_1_sub_score_54498", "test_1_sub_score_54499", "test_1_sub_score_54500", "test_2_sub_score_54498", "test_2_sub_score_54499", "test_2_sub_score_54500", "test_3_sub_score_54498", "test_3_sub_score_54499", "test_3_sub_score_54500", "test_4_sub_score_54498", "test_4_sub_score_54499", "test_4_sub_score_54500", "test_5_sub_score_54498", "test_5_sub_score_54499", "test_5_sub_score_54500", "test_6_sub_score_54498", "test_6_sub_score_54499", "test_6_sub_score_54500", "test_7_sub_score_54498", "test_7_sub_score_54499", "test_7_sub_score_54500", "test_8_sub_score_54498", "test_8_sub_score_54499", "test_8_sub_score_54500", "test_9_sub_score_54498", "test_9_sub_score_54499", "test_9_sub_score_54500", "test_10_sub_score_54498", "test_10_sub_score_54499", "test_10_sub_score_54500"])
imr20 = Imputer(missing_values = "NaN", strategy = "mean", axis = 1)
imr20.fit(test_subset2)
test_new2 = imr20.transform(test_subset2)
test_imputed2 = pd.DataFrame(data = test_new2, columns = ["test_1_correct_q_54498", "test_1_correct_q_54499", "test_1_correct_q_54500", "test_2_correct_q_54498", "test_2_correct_q_54499", "test_2_correct_q_54500", "test_3_correct_q_54498", "test_3_correct_q_54499", "test_3_correct_q_54500", "test_4_correct_q_54498", "test_4_correct_q_54499", "test_4_correct_q_54500", "test_5_correct_q_54498", "test_5_correct_q_54499", "test_5_correct_q_54500", "test_6_correct_q_54498", "test_6_correct_q_54499", "test_6_correct_q_54500", "test_7_correct_q_54498", "test_7_correct_q_54499", "test_7_correct_q_54500", "test_8_correct_q_54498", "test_8_correct_q_54499", "test_8_correct_q_54500", "test_9_correct_q_54498", "test_9_correct_q_54499", "test_9_correct_q_54500", "test_10_correct_q_54498", "test_10_correct_q_54499", "test_10_correct_q_54500"])
imr30 = Imputer(missing_values = "NaN", strategy = "mean", axis = 1)
imr30.fit(test_subset3)
test_new3 = imr30.transform(test_subset3)
test_imputed3 = pd.DataFrame(data = test_new3, columns = ["test_1_incorrect_q_54498", "test_1_incorrect_q_54499", "test_1_incorrect_q_54500", "test_2_incorrect_q_54498", "test_2_incorrect_q_54499", "test_2_incorrect_q_54500", "test_3_incorrect_q_54498", "test_3_incorrect_q_54499", "test_3_incorrect_q_54500", "test_4_incorrect_q_54498", "test_4_incorrect_q_54499", "test_4_incorrect_q_54500", "test_5_incorrect_q_54498", "test_5_incorrect_q_54499", "test_5_incorrect_q_54500", "test_6_incorrect_q_54498", "test_6_incorrect_q_54499", "test_6_incorrect_q_54500", "test_7_incorrect_q_54498", "test_7_incorrect_q_54499", "test_7_incorrect_q_54500", "test_8_incorrect_q_54498", "test_8_incorrect_q_54499", "test_8_incorrect_q_54500", "test_9_incorrect_q_54498", "test_9_incorrect_q_54499", "test_9_incorrect_q_54500", "test_10_incorrect_q_54498", "test_10_incorrect_q_54499", "test_10_incorrect_q_54500"])
test_imputed = pd.concat([test_imputed1, test_imputed2, test_imputed3], axis = 1)

percentages2 = []
for i in range(1,10):
    percentages2.append(percentage_calc(i, test_imputed))
correct2 = []
for i in range(1,10):
    correct2.append(correct_calc(i, test_imputed))
incorrect2 = []
for i in range(1,10):
    incorrect2.append(incorrect_calc(i, test_imputed))
X_predict1 = (np.array(percentages2[0:9])).transpose()
X_predict2 = (np.array(correct2[0:9])).transpose()
X_predict3 = (np.array(incorrect2[0:9])).transpose()
X_predict = np.column_stack((X_predict1, X_predict2, X_predict3))

# prediction
y_predict = lm.predict(X_predict)
np.savetxt("prediction3.csv", y_predict, delimiter=",")