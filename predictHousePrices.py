import numpy as np
import pandas as pd

# additional packages
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import BayesianRidge


###### 0. DEFINING NECESSARY CUSTOM CLASSES

class CustomAttrAdder(BaseEstimator, TransformerMixin):
	''' class definition to apply the feature engineering i.e. modified columns to the data'''

	def __init__(self):
		super(CustomAttrAdder, self).__init__()


	def addNewFeatures(self, X):
		''' new feature addition done '''

		X['TotalLivingArea'] = X['GrLivArea']+X['1stFlrSF']+X['2ndFlrSF']+X['LowQualFinSF']
		X['BedroomPerRoom'] = X['BedroomAbvGr'] / X['TotRmsAbvGrd']
		X['ExtraArea'] = X['GarageArea']+X['TotalBsmtSF']+X['MasVnrArea']+X['WoodDeckSF']+X['OpenPorchSF']+X['EnclosedPorch']+X['3SsnPorch']+X['ScreenPorch']+X['LotFrontage']+X['LotArea']+X['PoolArea']
		X['TotalBaths'] = X['FullBath']+X['HalfBath']+X['BsmtFullBath']+X['BsmtHalfBath']
		X['HouseStrength'] = X['YearRemodAdd'] - X['YearBuilt']
		X['HousePurchaseCond'] = X['YrSold'] - X['YearRemodAdd']

		return X


	def newFeatures(self, X):
		''' check the modified features for the data and verify their correlation '''

		X = self.addNewFeatures(X)
		correlation = X.select_dtypes(include=[np.number]).corr()
		print("The correlation values of the 'SalePrice' with every other column :\n", correlation['SalePrice'].sort_values(ascending=False))

		sns.heatmap(correlation)


	def fit(self, X, y=None):
		return self

	def transform(self, X, y=None):
		''' actually add the new features '''

		X = self.addNewFeatures(X)

		if y != None:
			y = self.addNewFeatures(y)
			return X, y

		return X




class MissingDataHandler(BaseEstimator, TransformerMixin):
	''' deals with all the missing values in the dataset '''

	def __init__(self):
		super(MissingDataHandler, self).__init__()


	def find_missing(self, X):
		''' finds missing data / NA values in columns '''

		total = X.isnull().sum().sort_values(ascending = False)
		percent_1 = X.isnull().sum()/X.isnull().count()*100
		percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
		missing_data = pd.concat([total, percent_2], axis=1, keys=['Total missing', 'Percentage'])
		return missing_data

	def delete_missing(self, X, column):
		''' deletes columns having more than 50% of their data missing '''

		X = X.drop(column, axis = 1)
		print("Deleting the columns with a lot of noise ...")
		return X


	def fit(self, X, y = None):
		return self

	def transform(self, X, y = None):

		# identify missing data in X/y
		missing_X = self.find_missing(X)

		# delete cols with more than 50% missing data
		# NOTE : .head(4) has been chosen to select only similar columns from multiple X's and y's
		cols = list(missing_X[missing_X.Percentage > 50].index)
		print("Columns with more than 50% of their data missing : ", cols)
		print("But 'MiscFeature' is the only one with actual missing information. Hence, dropping this column...\n")
		X = self.delete_missing(X, ['MiscFeature'])
		
		return X


		# similarly, for y
		if y != None:
			missing_y = self.find_missing(y)

			cols = list(missing_y[missing_y.Percentage > 50].index)
			print("Columns with more than 50% of their data missing : ", cols)
			print("But 'MiscFeatures' is the only one with actual missing information. Hence, dropping this column...\n")
			y = self.delete_missing(y, ['MiscFeature'])
			return X, y



class NAFiller(BaseEstimator, TransformerMixin):
	""" fills the NA values still in the data """

	def __init__(self):
		super(NAFiller, self).__init__()


	def numericalNA(self, X, y=None):
		''' modify the numerical columns of the dataset '''

		print("Dealing with NULLs in the numerical columns ...")
		num = X.select_dtypes(include = [np.number])
		data = MissingDataHandler.find_missing(self, num)
		print(data.head(10))
		missing_valued_cols = list(data[data['Total missing'] != 0].index)
		print("Columns with data missing : ", missing_valued_cols)

		# columns having continuous data with missing data (should be put in by the user at runtime)
		# NOTE: try to get the column name spelling correct
		continuous_valued_cols = []
		print("Enter the column names with continuous data values (not discrete) : \n")
		while True:
			entry = input()
			continuous_valued_cols.append(entry)
			if entry == ' ':
				continuous_valued_cols.pop()
				break

		for col in continuous_valued_cols:
			num[col].fillna(num[col].median(), inplace=True)


		# columns having discrete value with missing data (should be put in by the user at runtime)
		discrete_valued_cols = list(set(missing_valued_cols) - set(continuous_valued_cols))

		for col in discrete_valued_cols:
			num[col].fillna(num[col].mode()[0], inplace=True)

		# check if any NaN present
		print("Are there any NULLs remaining in the numerical columns --> ", num.isnull().values.any())

		return num


	def categoricalNA(self, X, y=None):
		''' modify the categorical columns of the dataset '''

		print("\nDealing with NULLs in the categorical columns ...")
		cat = X.select_dtypes(exclude = [np.number])
		data = MissingDataHandler.find_missing(self, cat)
		print(data.head(10))
		# group together all the columns with missing data
		missing_valued_cols = list(data[data['Total missing'] != 0].index)
		print("Columns with data missing : ", missing_valued_cols)

		# cols with NA meaning None
		entityNA_missing_cols = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "Alley", "PoolQC", "Fence"]

		for col in entityNA_missing_cols:
			cat[col].fillna('None', inplace=True)

		# for the rest of the cols, we can replace the NaN with their mode() value
		randomNA_missing_cols = list(set(missing_valued_cols) - set(entityNA_missing_cols))
		for col in randomNA_missing_cols:
			cat[col].fillna(cat[col].describe().top, inplace=True)


		# check if any NaN present
		print("Are there any NULLs remaining in the categorical columns --> ", cat.isnull().values.any())

		return cat



	def fit(self, X, y=None):
		return self

	def transform(self, X, y=None):
		''' fill the NA values separately for the categorical and numerical cols '''

		# numerical attributes
		X_num = self.numericalNA(X)
		# categorical attributes
		X_cat = self.categoricalNA(X)

		# concatenate both kinds
		X_naFilled = pd.concat([X_num, X_cat], axis=1)

		if y != None:
			# numerical attributes
			y_num = self.numericalNA(y)
			# categorical attributes
			y_cat = self.categoricalNA(y)

			# concatenate both kinds
			y_naFilled = pd.concat([y_num, y_cat], axis=1)

			return X_naFilled, y_naFilled

		return X_naFilled



class CategoricalAttrManager(BaseEstimator, TransformerMixin):
	''' deals with the multiple categorical columns encoding aspects '''

	def __init__(self):
		super(CategoricalAttrManager, self).__init__()


	def get_ordinal_cols(self, X, y):
		''' identify columns that can be label encoded '''

		cat_cols = list(X.columns)

		# choose orderly rankable/encodable columns
		orderly_arrangeable_cols = ["Street", "LotShape", "LandSlope", "ExterQual", "ExterCond","BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "HeatingQC", "CentralAir", "KitchenQual", "FireplaceQu", "GarageFinish", "GarageQual", "GarageCond","PavedDrive", "Alley", "PoolQC", "Fence"]

		ordinal_cols = []
		for col in orderly_arrangeable_cols:
			#check if that column is present in cat_cols and both sets havecommon elements
			if (set(X[col]) == set(y[col])) and (col in cat_cols):
				ordinal_cols.append(col)


		# identify other columns
		bad_cols = list(set(cat_cols) - set(ordinal_cols))
		print("Ordinally encodable columns are thus, : ", ordinal_cols, "\n")

		return ordinal_cols, bad_cols



	def perform_ordinal_enc(self, X, y):
		''' perform ordinal encoding on specific columns '''

		# apply ordinal encoding
		ordinal_encoder = OrdinalEncoder()

		# create ordinally encoded dataframes
		new_X = pd.DataFrame(ordinal_encoder.fit_transform(X.astype(str)), index = X.index, columns = X.columns)
		new_y = pd.DataFrame(ordinal_encoder.transform(y.astype(str)), index = y.index, columns = y.columns)
		print("\nCategories for ordinal encoding : \n", ordinal_encoder.categories_)

		return new_X, new_y



	def get_onehot_cols(self, X, bad_cols):
		''' using bad_cols for OneHotEncoding and looking at their cardinality '''
		
		low_cardinality = [col for col in bad_cols if X[col].nunique() < 10]
		cols_to_remove = list(set(bad_cols)-set(low_cardinality))
		print("Categorical columns to be dropped : ", cols_to_remove)

		return low_cardinality, cols_to_remove


	def perform_onehot_enc(self, X, y):
		''' perform one hot encoding on specific columns '''

		# apply one-hot encoding
		OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

		# create one-hot encoded dataframes
		new_X = pd.DataFrame(OH_encoder.fit_transform(X.astype(str)), index = X.index, columns = OH_encoder.get_feature_names())
		new_y = pd.DataFrame(OH_encoder.transform(y.astype(str)), index = y.index, columns = OH_encoder.get_feature_names())
		print("\nCategories for one-hot encoding : \n", OH_encoder.categories_)

		return new_X, new_y


	def fit(self, X, y):
		return self

	def transform(self, X, y):

		
		# obtain the ordinally encodable columns
		ordinal_cols, not_ordinal_cols = self.get_ordinal_cols(X, y)

		ordinal_X = X[ordinal_cols].copy()
		ordinal_y = y[ordinal_cols].copy()

		# create ordinally encoded dataframes
		ordinal_X, ordinal_y = self.perform_ordinal_enc(ordinal_X, ordinal_y)


		# obtain onehot encodable columns
		onehot_cols, cols_to_drop = self.get_onehot_cols(X, not_ordinal_cols)

		# create one-hot encoded dataframes
		OH_X, OH_y = self.perform_onehot_enc(X[onehot_cols], y[onehot_cols])
		

		#  combining both categorical sets of columns
		X_cat = pd.concat([ordinal_X, OH_X], axis=1)
		y_cat = pd.concat([ordinal_y, OH_y], axis=1)


		# check if any categorical column has 'object' dtype
		obj_cols = list(X_cat.select_dtypes(exclude=[np.number]).columns)

		for col in obj_cols:
			X_cat[col] = X_cat[col].astype(float)
			y_cat[col] = y_cat[col].astype(float)

		print("\nOrdinally encoded columns are thus : ", ordinal_cols)
		print("\nOneHot encoded columns being, : ", onehot_cols)


		return X_cat, y_cat




###### 1. OBTAIN DATA

train_input = input("Input train data : ")
test_input = input("Input test data : ")

train_full = pd.read_csv(train_input, index_col='Id')
test_full = pd.read_csv(test_input, index_col='Id')

print(train_full.info())
print("Shape training data : {}".format(train_full.shape))
print("Shape training data : {}".format(test_full.shape))


# Understanding the target variable

# checking the skewness
print("Skew : \n", train_full.SalePrice.skew())
plt.hist(train_full.SalePrice)
plt.show()

# log transformed values
print("Skew : \n", np.log(train_full.SalePrice).skew())
plt.hist(np.log(train_full.SalePrice))
plt.show()

# As is evident from the plots, transformed target values are more data-friendly



###### 2. DATA EXPLORATION / ANALYSIS

# 1. Looking carefully at the numerical cols with continuous distribution (mostly the ones involving surface areas)
print(train_full.select_dtypes(include=[np.number]).columns)

print(list(train_full.filter(regex='SF').columns), '\n')
print(list(train_full.filter(regex='Area').columns))

# list of columns with continuous data
train_continuous_num_cols = list(train_full.filter(regex='SF').columns) + list(train_full.filter(regex='Area').columns)
test_continuous_num_cols = list(test_full.filter(regex='SF').columns) + list(test_full.filter(regex='Area').columns)

# checking for the skew
print([train_full[col].skew() for col in train_continuous_num_cols])
train_full[train_continuous_num_cols].hist(bins=50, figsize = (30,20))
plt.show()

print([test_full[col].skew() for col in test_continuous_num_cols])
test_full[test_continuous_num_cols].hist(bins=50, figsize = (30,20))
plt.show()


# tranformed cols
print([np.log(train_full[col]).skew() for col in train_continuous_num_cols])
print([np.log(test_full[col]).skew() for col in test_continuous_num_cols])
# almost all these cols require log transformed values; to check after missing values are filled 


# 2. Identifying correlation in the train set
correlation = train_full.select_dtypes(include = [np.number]).corr()
print(correlation['SalePrice'].sort_values(ascending=False))
sns.heatmap(correlation)


# looking at few scatter plots to identify outliers
fig, axs = plt.subplots(2,2)
axs[0, 0].scatter(x = train_full['GrLivArea'], y = train_full['SalePrice'])
axs[0, 0].set_title('Above ground living area')
axs[0, 1].scatter(x = train_full['GarageArea'], y = train_full['SalePrice'])
axs[0, 1].set_title('Garage area in square feet')
axs[1, 0].scatter(x = train_full['TotalBsmtSF'], y = train_full['SalePrice'])
axs[1, 0].set_title('Total basement area')
axs[1, 1].scatter(x = train_full['1stFlrSF'], y = train_full['SalePrice'])
axs[1, 1].set_title('1st floor area')

fig.tight_layout()


## we see some outliers to be removed
train_full = train_full[train_full['1stFlrSF'] < 4000]
train_full = train_full[train_full['GrLivArea'] < 4000]
train_full = train_full[train_full['TotalBsmtSF'] < 5000]

print(train_full.shape)



# 3. Trying out possible feature combinations and then checking their correlation with the target
custom_attr = CustomAttrAdder()

# visualizing the correlation after the new features are added (if they are any good)
train_temp = train_full.copy()
custom_attr.newFeatures(train_temp)

# NOTE : correlation of the new attributes with the SalePrice looks pretty good


####### 3. DATA DISTINCTION

# Drop rows with missing target value
train_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
train_full.reset_index(drop=True, inplace=True)
y = np.log(train_full.SalePrice)
train = train_full.drop(['SalePrice'], axis=1)



######## 4. DATA CLEANING

# 1. Dropping columns with lots of missing values (using the MissingDataHandler() class defined above)

missingData = MissingDataHandler()

print(missingData.find_missing(train))
print(missingData.find_missing(test_full))

# dropping relevant columns
train = missingData.fit_transform(train)
test = missingData.transform(test_full)

# 2. Filling NaN spaces

na_filler = NAFiller()

# fill the remaining NA values accordingly
train = na_filler.fit_transform(train)
test = na_filler.transform(test)


######### 5. FEATURE ENGINEERING

# 1. Dealing with the numerical columns

# train set
train_numerical = train.select_dtypes(include = [np.number])
train_num = custom_attr.fit_transform(train_numerical)

# test set
test_numerical = test.select_dtypes(include = [np.number])
test_num = custom_attr.fit_transform(test_numerical)

print(train_num)
print(test_num)


# transforming the continuous valued cols

def transform_cols(listOfColumns, data):
	''' log transform the specified columns '''

	for i in range(0, len(listOfColumns)):
		if listOfColumns[i] in list(data.columns):
			data[listOfColumns[i]] = np.log(data[listOfColumns[i]])

	# all negative infinities can be replaced by zeroes
	data = data.replace(-np.Inf, 0)

	return data

# columns in the train with continuous valued data
print(train_continuous_num_cols)
train_num = transform_cols(train_continuous_num_cols, train_num)

# test data
print(test_continuous_num_cols)
test_num = transform_cols(test_continuous_num_cols, test_num)


# 2. Dealing with the categorical columns

cat_manager = CategoricalAttrManager()

train_categorical = train.select_dtypes(exclude = [np.number])
test_categorical = test.select_dtypes(exclude = [np.number])

train_cat, test_cat = cat_manager.transform(train_categorical, test_categorical)



##### 6. COMBINE BOTH NUMERICAL AND CATEGORICAL 
train_prepared = pd.concat([train_num, train_cat], axis=1)
test_prepared = pd.concat([test_num, test_cat], axis=1)


##### 7. MODEL COMPARISON

def score_displayer(scores):
	print("Scores : ", scores)
	print("Mean score : ", scores.mean())
	print("Standard deviation : ", scores.std())

# 1. LINEAR REGRESSION
lr = LinearRegression()
score_lr = cross_val_score(lr, train_prepared, y, scoring="neg_mean_squared_error", cv=10)
lr_rmse_scores = np.sqrt(-score_lr)
score_displayer(lr_rmse_scores)


# 2. DECISION TREE CLASSIFIER
dt = DecisionTreeRegressor(random_state = 0)
score_dt = cross_val_score(dt, train_prepared, y, scoring="neg_mean_squared_error", cv=10)
dt_rmse_scores = np.sqrt(-score_dt)
score_displayer(dt_rmse_scores)


# 3. GRADIENT BOOSTING
gb = XGBRegressor(random_state=0, n_estimators=500, learning_rate=0.05)
score_gb = cross_val_score(gb, train_prepared, y, scoring="neg_mean_squared_error", cv=10)
gb_rmse_scores = np.sqrt(-score_gb)
score_displayer(gb_rmse_scores)


# 4. BAYESIAN RIDGE
br = BayesianRidge()
score_br = cross_val_score(br, train_prepared, y, scoring="neg_mean_squared_error", cv=10)
br_rmse_scores = np.sqrt(-score_br)
score_displayer(br_rmse_scores)



##### 8. PREPARE FOR SUBMISSION

# csv that contains the predicted SalePrice for each observation in the test.csv dataset

submission = pd.DataFrame()
submission['Id'] = test_prepared.index


# after test preparation as train
features = test_prepared.select_dtypes(include = [np.number]).interpolate()

# since bayesian ridge performed the best
br.fit(train_prepared)
predictions_br = br.predict(features)

# Now we’ll transform the predictions_br to the correct form
final_predictions_br = np.exp(predictions_br)

submission['SalePrice'] = final_predictions_br
submission.head()

submission.to_csv('submission_predictHousingPrices.csv', index=False)