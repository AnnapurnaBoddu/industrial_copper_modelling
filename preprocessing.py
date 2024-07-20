import pandas as pd
import numpy as np
import seaborn as sns
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor




data = pd.read_excel("C:\\Users\\2anna\\Desktop\\Pyhton course\\capstoneproject5\\Copper_Set.xlsx")

print(data)

print(data.nunique())

#we have country, status, item type are catogorical columns

print(data.info())
print(data.describe())

# dealing with data in wrong format
data['item_date'] = pd.to_datetime(data['item_date'], format='%Y%m%d', errors='coerce').dt.date
data['delivery date'] = pd.to_datetime(data['delivery date'], format='%Y%m%d', errors='coerce').dt.date
data['quantity tons'] = pd.to_numeric(data['quantity tons'], errors='coerce')
data['customer'] = pd.to_numeric(data['customer'], errors='coerce')
data['country'] = pd.to_numeric(data['country'], errors='coerce')
data['application'] = pd.to_numeric(data['application'], errors='coerce')
data['thickness'] = pd.to_numeric(data['thickness'], errors='coerce')
data['width'] = pd.to_numeric(data['width'], errors='coerce')
data['product_ref'] = pd.to_numeric(data['product_ref'], errors='coerce')
data['selling_price'] = pd.to_numeric(data['selling_price'], errors='coerce')
data['material_ref'] = np.where(data['material_ref'].str.startswith('0000'), None, data['material_ref'])

print(data.isnull().sum())

# meterial ref has large no of missing values so we can drop the column
data = data.drop(columns='material_ref')

print(data.isnull().sum())

print(data.tail(15))

# quantity and selling price values are not below 0. so we convert to null for below 0 values.

data['quantity tons'] = data['quantity tons'].apply(lambda x: np.nan if x <= 0 else x)
data['selling_price'] = data['selling_price'].apply(lambda x: np.nan if x <= 0 else x)
print(data.describe().T)

print(data.isnull().sum())

print(data.dtypes)

# Handling null values using median and mode
# median - middle value in dataset (asc/desc), mode - value that appears most frequently in dataset

# object datatype using mode
data['item_date'].fillna(data['item_date'].mode().iloc[0], inplace=True)
data['status'].fillna(data['status'].mode().iloc[0], inplace=True)
data['delivery date'].fillna(data['delivery date'].mode().iloc[0], inplace=True)

# numerical datatype using median
data['quantity tons'].fillna(data['quantity tons'].median(), inplace=True)
data['customer'].fillna(data['customer'].median(), inplace=True)
data['country'].fillna(data['country'].median(), inplace=True)
data['application'].fillna(data['application'].median(), inplace=True)
data['thickness'].fillna(data['thickness'].median(), inplace=True)
data['selling_price'].fillna(data['selling_price'].median(), inplace=True)

print(data['status'].value_counts())

# convert categorical data into numerical data - using map and ordinal encoder methods


data['status'] = data['status'].map({'Lost': 0, 'Won': 1, 'Draft': 2, 'To be approved': 3, 'Not lost for AM': 4,
                                     'Wonderful': 5, 'Revised': 6, 'Offered': 7, 'Offerable': 8})
data['item type'] = OrdinalEncoder().fit_transform(data[['item type']])
print(data)

# remove id column beacuse for prediction id is not neccessary

data = data.drop(columns='id')


# find outliers - box plot & skewed data - hist plot and violin plot

def plot(df, column):
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 3, 1)
    sns.boxplot(data=df, x=column)
    plt.title(f'Box Plot for {column}')

    plt.subplot(1, 3, 2)
    sns.histplot(data=df, x=column, kde=True, bins=50)
    plt.title(f'Distribution Plot for {column}')

    plt.subplot(1, 3, 3)
    sns.violinplot(data=df, x=column)
    plt.title(f'Violin Plot for {column}')
    plt.show()


for i in ['quantity tons', 'customer', 'country', 'item type', 'application', 'thickness', 'width', 'selling_price']:
    plot(data, i)

# quantity tons, thickness and selling price data are skewd. so using the log transformation method to handle the skewness data

df1 = data.copy()
df1['quantity tons_log'] = np.log(df1['quantity tons'])
df1['thickness_log'] = np.log(df1['thickness'])
df1['selling_price_log'] = np.log(df1['selling_price'])
print(df1)

# after log transformation the data are normally distributed and reduced the skewness. [hist plot and violin plot]
for i in ['quantity tons_log', 'thickness_log', 'width', 'selling_price_log']:
    plot(df1, i)

# outliers handling IQR method
df2 = df1.copy()
print(df2)


# Using IQR and clip() methods to handle the outliers and add a new column of dataframe

def outlier(df, column):
    iqr = df[column].quantile(0.75) - df[column].quantile(0.25)
    upper_threshold = df[column].quantile(0.75) + (1.5 * iqr)
    lower_threshold = df[column].quantile(0.25) - (1.5 * iqr)
    df[column] = df[column].clip(lower_threshold, upper_threshold)


# (Ex: lower threshold = 5 and upper threshold = 20)
# above upper threshold values (>20) are converted to upper threshold value (20) in features
# below lower threshold values (<5)  are converted to lower threshold value (5)  in features

outlier(df2, 'quantity tons_log')
outlier(df2, 'thickness_log')
outlier(df2, 'selling_price_log')
outlier(df2, 'width')
print(df2)

# transform the outliers to within range using IQR and clip() methods - box plot

for i in ['quantity tons_log', 'thickness_log', 'width', 'selling_price_log']:
    plot(df2, i)

# after add the new column of 'quantity tons_log', 'thickness_log', 'selling_price_log', drop the existing columns
df3 = df2.drop(columns=['quantity tons', 'thickness', 'selling_price'])
print(df3)

# check the data types
print(df3.dtypes)

# Need to verify any columns are highly correlated using Heatmap. If any columns correalaion value >= 0.7 (absolute value), drop the columns.

col = ['quantity tons_log', 'customer', 'country', 'status', 'application', 'width', 'product_ref', 'thickness_log',
       'selling_price_log']
df_heatmap = df3[col].corr()
sns.heatmap(df_heatmap, annot=True)

# split the day, month, and year from 'delivery_date_1' column and add dataframe (This data also help us to prediction)
df_final = df3.copy()

# Convert 'delivery date' and 'item_date' columns to datetime if they are not already
df_final['delivery date'] = pd.to_datetime(df_final['delivery date'], errors='coerce')
df_final['item_date'] = pd.to_datetime(df_final['item_date'], errors='coerce')

df_final['delivery_date_day'] = df_final['delivery date'].dt.day
df_final['delivery_date_month'] = df_final['delivery date'].dt.month
df_final['delivery_date_year'] = df_final['delivery date'].dt.year
df_final['item_date_day'] = df_final['item_date'].dt.day
df_final['item_date_month'] = df_final['item_date'].dt.month
df_final['item_date_year'] = df_final['item_date'].dt.year

# finally drop the item_date, delivery_date and date_difference columns
df_final.drop(columns=['item_date', 'delivery date'], inplace=True)
print(df_final)

# classification method to predict status


print(df_final.head())

# check data types
print(df_final.dtypes)

df_c = df_final.copy()

# filter the status column values only 1 & 0 rows in a new dataframe ['Won':1 & 'Lost':0]
df_c = df_c[(df_c.status == 1) | (df_c.status == 0)]
print(df_c)

# check no of rows (records) of each 1 and 0 in dataframe
df_c['status'].value_counts()

# in status feature, the 'Won' and 'Lost' value difference is very high. So we need to oversampling to reduce the difference

x = df_c.drop('status', axis=1)
y = df_c['status']

x_new, y_new = SMOTETomek().fit_resample(x, y)

print(x.shape, y.shape, x_new.shape, y_new.shape)


# check the accuracy of training and testing using metrics
# algorithm.__name__  - it return the algorithm name

def machine_learning_classification(x_new, y_new, algorithm):
    x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, test_size=0.2, random_state=42)
    model = algorithm().fit(x_train, y_train)

    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    accuracy_train = metrics.accuracy_score(y_train, y_pred_train)
    accuracy_test = metrics.accuracy_score(y_test, y_pred_test)

    # algo = str(algorithm).split("'")[1].split(".")[-1]
    accuracy_metrics = {'algorithm': algorithm.__name__,
                        'accuracy_train': accuracy_train,
                        'accuracy_test': accuracy_test}

    return accuracy_metrics


print(machine_learning_classification(x_new, y_new, DecisionTreeClassifier))
print(machine_learning_classification(x_new, y_new, ExtraTreesClassifier))
print(machine_learning_classification(x_new, y_new, RandomForestClassifier))
print(machine_learning_classification(x_new, y_new, AdaBoostClassifier))
print(machine_learning_classification(x_new, y_new, GradientBoostingClassifier))
print(machine_learning_classification(x_new, y_new, XGBClassifier))

# we got good accuracy after oversampling
# ExtraTreesClassifier and RandomForestClassifier both have good testing accuracy, but in training accuracy is overfitting.
# RandomForestClassifier is good interpretability, so i select the algorithm

# GridsearchCV is a cross validation function.

# Hyper parameter tuning - we give parameter values manually in the algorithm to reduce the overfitting issue and get better accuracy.

# so using gridserachcv method - to pass the mulitiple values in each parameters and it try to evaluate all the combination of values and
# finally return the best accuracy parameter values based on the score.

# example: {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2}


x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, test_size=0.2, random_state=42)

param_grid = {'max_depth': [2, 5, 10, 20],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4],
              'max_features': ['sqrt', 'log2']}

grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(x_train, y_train)

# n_jobs = -1 means it uses the all processors in this process

# evaluate all the parameter combinations and return the best parameters based on score
print(grid_search.best_params_)

print(grid_search.best_score_)

# now the training accuracy overfitting reduced. so now model will predict effectively for unseen data
# predict the status and check the accuracy using metrics

x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, test_size=0.2, random_state=42)

model = RandomForestClassifier(max_depth=20, max_features='sqrt', min_samples_leaf=1, min_samples_split=2).fit(x_train,
                                                                                                               y_train)
y_pred = model.predict(x_test)

print(confusion_matrix(y_true=y_test, y_pred=y_pred))
print(classification_report(y_true=y_test, y_pred=y_pred))

# Receiver Operating Characteristic (ROC) Curve and Area Under the Curve (AUC)

FP, TP, threshold = roc_curve(y_true=y_test, y_score=y_pred)
auc_curve = auc(x=FP, y=TP)
print(auc_curve)

plt.plot(FP, TP, label=f"ROC Curve (area={round(auc_curve, 2)}) ")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.10])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# will pass the user data manually to check the prediction of status ar our model

user_data = np.array([[30153963, 30, 6, 28, 952, 628377, 5.9, -0.96, 6.46, 1, 4, 2021, 1, 1, 2021]])
y_p = model.predict(user_data)
if y_p[0] == 1:
    print('Won')
else:
    print('Lose')

user_data = np.array([[30223403, 78, 5, 10, 1500, 1668701718, 2.2, 0, 7.13, 1, 4, 2021, 1, 7, 2021]])
y_p = model.predict(user_data)
if y_p[0] == 1:
    print('Won')
else:
    print('Lose')

# Note:
# google colab sklearn version is 1.2.2
# so if you save your model in pickle in colab and download to your system.
# after you load the model in your system for prediction measns ----> your system have same 1.2.2 version of sklearn
# otherwise it cannot load in your system and it shows error


# save the classification model by using pickle


with open('classification_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Now successfully saved our model in 'classification_model.pkl' pickle file.
# Hereafter you can load and using this pickle file only to predict the status. (don't need any packages - only need numpy for pass user input)
# load pickle model to predict the status (like Trained brain).

with open('classification_model.pkl', 'rb') as f:
    model = pickle.load(f)

user_data = np.array([[30223403, 78, 5, 10, 1500, 1668701718, 2.2, 0, 7.13, 1, 4, 2021, 1, 7, 2021]])
y_p = model.predict(user_data)
if y_p[0] == 1:
    print('Won')
else:
    print('Lose')

# Regression Method - Prdict Selling Price


from sklearn.preprocessing import OrdinalEncoder


print(df_final)

print(df_final.columns)


# check the train and test accuracy using R2 (R-squared ---> coefficient of determination) to predict selling price

def machine_learning_regression(df, algorithm):
    x = df.drop(columns=['selling_price_log'], axis=1)
    y = df['selling_price_log']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = algorithm().fit(x_train, y_train)
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    # algo = str(algorithm).split("'")[1].split(".")[-1]
    accuracy_metrics = {'algorithm': algorithm.__name__,
                        'R2_train': r2_train,
                        'R2_test': r2_test}

    return accuracy_metrics


print(machine_learning_regression(df_final, DecisionTreeRegressor))
print(machine_learning_regression(df_final, ExtraTreesRegressor))
print(machine_learning_regression(df_final, RandomForestRegressor))
print(machine_learning_regression(df_final, AdaBoostRegressor))
print(machine_learning_regression(df_final, GradientBoostingRegressor))
print(machine_learning_regression(df_final, XGBRegressor))

# ExtraTreesClassifier and RandomForestClassifier both have good testing accuracy, but in training accuracy is overfitting.
# RandomForestClassifier is good interpretability, so i select the algorithm

# already i explained about GridsearchCV in classification method

x = df_final.drop(columns=['selling_price_log'], axis=1)
y = df_final['selling_price_log']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

param_grid_r = {'max_depth': [2, 5, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]}

grid_search_r = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid_r, cv=5, n_jobs=-1)
grid_search_r.fit(x_train, y_train)

# best parameters for hypertuning the random forest algorithm for better accuracy in unseen data
print(grid_search_r.best_params_, grid_search_r.best_score_)

# pass the parameters and check the accuracy for both training and testing & overfitting

x = df_final.drop(columns=['selling_price_log'], axis=1)
y = df_final['selling_price_log']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(max_depth=20, max_features=None, min_samples_leaf=1, min_samples_split=2).fit(x_train,
                                                                                                            y_train)
y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
print(r2_train, r2_test)

# predict the selling price with hypertuning parameters and calculate the accuracy using metrics

x = df_final.drop(columns=['selling_price_log'], axis=1)
y = df_final['selling_price_log']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(max_depth=20, max_features=None, min_samples_leaf=1, min_samples_split=2).fit(x_train,
                                                                                                            y_train)
y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

metrics_r = {'R2': r2,
             'Mean Absolute Error': mae,
             'Mean Squared Error': mse,
             'Root Mean Squared Error': rmse}

print(metrics_r)

# manually passed the user input and predict the selling price

user_data = np.array([[30202938, 25, 1, 5, 41, 1210, 1668701718, 6.6, -0.2, 1, 4, 2021, 1, 4, 2021]])
y_pred = model.predict(user_data)
y_pred[0]

# using Inverse Log Transformation to convert the value to original scale of the data (exp)
print(np.exp(y_pred[0]))

# save the regression model by using pickle

with open('regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# load the pickle model to predict selling price

with open('regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

y_pred = model.predict(np.array([[30202938, 25, 1, 5, 41, 1210, 1668701718, 6.6, -0.2, 1, 4, 2021, 1, 4, 2021]]))
print(np.exp(y_pred[0]))