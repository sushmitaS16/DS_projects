import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


#### OBTAIN DATA

train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

print(train.shape)
print(test.shape)



#### FIND MISSING DATA

## We should remove rows if there is any missing data

def find_missing(data):
	### identify missing data

	total = data.isnull().sum().sort_values(ascending = False)
	percent_1 = data.isnull().sum()/data.isnull().count()*100
	percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
	missing_data = pd.concat([total, percent_2], axis=1, keys=['Total missing', '%'])
	return missing_data

find_missing(train)
find_missing(test)


#### SEPARATING OUT THE TARGET VALUES FROM TRAINING SET

y = train.label
train = train.drop(['label'], axis=1)


y.value_counts().plot(kind='bar')

## From this bar plot, we can see that all the classes are more or less balanced.



#### MODEL EVAL

def checkAccuracy(model, X, y):
	''' select the model giving the highest accuracy '''

	print("\nUsing model : ", model, "\n")

	# 1. measuring accuracy using cross validation
	cross_val_accuracy_score = cross_val_score(model, X, y, cv=3, scoring='accuracy')
	avg_score = cross_val_accuracy_score.mean()
	std_dev_score = cross_val_accuracy_score.std()

	return avg_score, std_dev_score



def modelEvaluation(model, X, y):
	''' get the necessary model details '''

	#### ORIGINAL INPUT DATA (Note : Making predictions using the k-fold cross_val_predict() function)

	y_pred = cross_val_predict(model, X, y, cv=3)


	# 2. error analysis using confusion matrix
	print(classification_report(y, y_pred))

	confusionMatrix = confusion_matrix(y, y_pred)
	print(confusionMatrix)
	plt.matshow(confusionMatrix, cmap=plt.cm.Blues)
	plt.show()

	# error plot
	rowSums = confusionMatrix.sum(axis=1, keepdims=True)
	normalConfusionMatrix = confusionMatrix / rowSums
	# filling all diagonals with zero to keep the error terms only
	np.fill_diagonal(normalConfusionMatrix, 0)
	plt.matshow(normalConfusionMatrix, cmap=plt.cm.gray)
	plt.show()





#### COMPARE MULTIPLE MODELS


# 1. Stochastic Gradient Descent CLassifier
sgd = SGDClassifier()
sgd.fit(train, y)
print("Mean accuracy score and its stadard deviation based on k-fold cross validation : ", checkAccuracy(sgd, train, y))

# 2. K-Nearest Neighbors CLassifier
kn = KNeighborsClassifier()
kn.fit(train, y)
print("Mean accuracy score and its standard deviation based on k-fold cross validation : ", checkAccuracy(kn, train, y))

# 3. Random Forest Classifier
rf = RandomForestClassifier()
rf.fit(train, y)
print("Mean accuracy score and its standard deviation based on k-fold cross validation : ", checkAccuracy(rf, train, y))

# 4. Support Vector Machine
svm = SVC()
svm.fit(train, y)
print("Mean accuracy score and its standard deviation based on k-fold cross validation : ", checkAccuracy(svm, train, y))


## Now, focusing on the model with the best accuracy score...

# Check if the accuracy increases with scaled data for the most accurate model
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train.astype(np.float64))

svm_scaled = SVC()
svm_scaled.fit(train_scaled, y)
print("Mean accuracy score and its standard deviation based on k-fold cross validation : ", checkAccuracy(svm_scaled, train_scaled, y))


# Based on which model has the highest accuracy above, further error evaluation is performed
modelEvaluation(svm, train, y)


#### NOTE: For Classification problems, mae, mse, rmse evaluation is not required, but classification accuracy is what matters 

### In the error plot, the rows represent actual classes, whereas the columns represent the predicted classes.
## To explain, the column for class 1 is quite bright, which tells you that many images get missclassified as 1. However, row for class 1 is good and tells that actual 1s are getting classified correctly.
# Similarly, one can say that the brighter cells are getting misclassified as one-another.


#### PREDICTING LABELS FOR TEST DATA

# csv that contains the predicted label for each observation in the test.csv dataset
submission = pd.DataFrame()

submission['ImageId'] = test.index
print(submission)

submission.drop(submission.iloc[0], inplace=True)
submission = submission.reset_index(drop=True)

additionalRow = {'ImageId' : [28000]}
additionalRow = pd.DataFrame(additionalRow)

submission = submission.append(additionalRow, ignore_index=True)
print(submission)

predictions = svm.predict(test)

submission['label'] = predictions
submission.head()

submission.to_csv('submission_mnistNumbers.csv', index=False)