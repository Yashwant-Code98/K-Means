# load a required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# load a dataset
df = pd.read_csv('bank.csv')
# view a first 5 records of dataset
print(df.head())
# view last 5 records of dataset
print(df.tail())
# check rows and columns
print(df.shape)
# check data type 
print(df.dtypes)
# check missing values of dataset
print(df.isnull().sum())
# check the unique values
print(df['age'].unique())
# check the duplicate value in dataset
print(df.age.duplicated().sum())
# view only numeric columns
print(df.select_dtypes(include=['int64','float64']).keys())
# for values
print(df.select_dtypes(include=['float64']).values)
# view only categorical columns
print(df.select_dtypes(include=['object']).columns)
# for values
print(df.select_dtypes(include=['object']).values)
# check indexing range of values
print(df.select_dtypes(include=['int64','float64']).index)
# EDA ON BANK DATASET
print(df['age'].value_counts())
# visualize the age column
sns.set(rc={'figure.figsize':(8,5)})
sns.histplot(df['age'],color='orange')
plt.title('Analysis On Age')
plt.legend()
sns.set()
plt.show()
print(df['job'].value_counts())
sns.set(rc=({'figure.figsize':(15,10)}))
sns.countplot(df['job'])
plt.title('Analysis On Job')
plt.show()

sns.barplot(x = 'age', y = 'job', data = df, hue='job')
plt.title('Age Vs Job')
plt.show()

# K-Means Clustering 
# split the model into training and testing
from sklearn.model_selection import train_test_split
x = df[['age']]
y = df[['marital']]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
# view training subset
print(x_train.head())
print(y_train.head())
# view testing subset
print(x_test.head())
print(y_test.head())
from sklearn.cluster import KMeans
km = KMeans()
print(km.fit(x_train,y_train))
y_pred = km.predict(x_test)
# view actual values 
print(y_test.head())
# view predicted values
print(y_pred[0:5])






