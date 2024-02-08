import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_train=pd.read_csv("train.csv")
df_test=pd.read_csv("test.csv")

#merging both the data
df=pd.concat([df_train, df_test], ignore_index=True)

df.drop(["User_ID"], axis=1, inplace=True)

#Handling categorical feature "GENDER"
#df['Gender']=pd.get_dummies(df['Gender'], drop_first=1).astype(int)
df['Gender']=df["Gender"].map({'F':0, 'M':1})

'''
df['Age'].unique()   "To know all the unique values present"
For age we can do df['Age']=pd.get_dummies(df['Age], drop_first=True) but this is not good as it 
will give 5-6 columns
'''
df['Age']=df['Age'].map({'0-17':1, '18-25':2, '26-35':3, '36-45':4, '46-50':5, '51-55':6, '55+':7})
'''
df_city=pd.get_dummies(df['City_Category'], drop_first=True).astype(int)
df=pd.concat([df, df_city], axis=1)
df.drop('City_Category', axis=1, inplace=True)
'''

df['City_Category']=df['City_Category'].map({'A':1, 'B':2, 'C':3})

#Checking NULL values
#print(df.isnull().sum())
#print(df['Product_Category_2'].value_counts())
#In discrete values, best way to fill missing values is mode
#df['Product_Category_2']=df['Product_Category_2'].mode will give values with index 
df['Product_Category_2']=df['Product_Category_2'].fillna(df['Product_Category_2'].mode()[0])
df['Product_Category_3']=df['Product_Category_3'].fillna(df['Product_Category_3'].mode()[0])

#df['Stay_In_Current_City_Years'].unique()
df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].str.replace('+','')
#print(df.info()) It is showing object so we have to change it to integer
df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].astype(int)

#Visualization
'''
sns.barplot(x='Age', y='Purchase', data=df, hue='Gender')
sns.barplot(x='Occupation', y='Purchase', data=df, hue='Gender')
sns.barplot(x='Product_Category_1', y='Purchase', data=df, hue='Gender')
sns.barplot(x='Product_Category_2', y='Purchase', data=df, hue='Gender')
sns.barplot(x='Product_Category_3', y='Purchase', data=df, hue='Gender')
plt.show()
'''
#Feature Scaling
df_test=df[df['Purchase'].isnull()]
#df_train=df[~df['Purchase'].isnull()]
df_train=df[df['Purchase'].notnull()]

x=df_train.drop(['Product_ID', 'Purchase'], axis=1)
y=df_train['Purchase']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

from sklearn.linear_model import LinearRegression
regresssor=LinearRegression()
regresssor.fit(x_train, y_train)
y_pred=regresssor.predict(x_test)

from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error
r2= r2_score(y_test, y_pred)
print(f"R2_squared : {r2}")


