from sklearn.tree import DecisionTreeClassifier
import pandas as pd 
import numpy as np
import joblib
## Preparing model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# model = LogisticRegression(random_state=42)
model = DecisionTreeClassifier(max_depth=1,random_state=42)
    # 'DecisionTreeClassifier':DecisionTreeClassifier(max_depth=1,random_state=42)

## Preparing encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
## Collecting the dataset
URL = "https://raw.githubusercontent.com/ukantjadia/Loan-Prediction/Main/Data/train_u6lujuX_CVtuZ9i.csv"
df = pd.read_csv(URL)
# df = pd.read_csv("../Data/train_u6lujuX_CVtuZ9i.csv")
## Droping the Loan id 
df.drop('Loan_ID',axis=1,inplace=True)

## Converting the data type of credit_history int to object
df['Credit_History'] = df['Credit_History'].astype('O')

# %%
## Seprating the data types
Tcat_data = []
Tnum_data = []
for name, dtype in enumerate(df.dtypes):
    if dtype == object:
        Tcat_data.append(df.iloc[:,name])
    else:
        Tnum_data.append(df.iloc[:,name])
Tcat_data = pd.DataFrame(Tcat_data).T
Tnum_data = pd.DataFrame(Tnum_data).T

## Handling the missing value in categorical and numerical data
### categorical data
Tcat_data = Tcat_data.apply(lambda x: x.fillna(x.value_counts().index[0]))
### Numerical data
Tnum_data.fillna(method='bfill',inplace=True)

## Transforming the categorical data
target_values = {'Y':0,'N':1}
target = Tcat_data['Loan_Status']
Tcat_data.drop('Loan_Status',axis=1,inplace=True)
target = target.map(target_values)

Transform_cat_data = pd.DataFrame()
for data in Tcat_data:
    Transform_cat_data[data] = le.fit_transform(Tcat_data[data])
X = pd.concat([Tnum_data,Transform_cat_data],axis=1)
y = target
## Spliting the dataset for training and testing
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
## Training and testing the dataset
model.fit(X_train.values,y_train.values)    
y_pred = model.predict(X_test.values)
## Calculating accuracy score
from sklearn.metrics import accuracy_score,f1_score
acc = accuracy_score(y_test,y_pred)
# print("Test Accuracy: ",accuracy_score(y_test,y_pred))
# print("Test F1 Score: ",f1_score(y_test,y_pred))
test = [ 4583, 1508, 128, 360, 3, 6, 0, 2, 4, 1, 5 ]
import numpy as np 
test2 = np.array(test).reshape(1,-1)
print(model.predict(test2))
## Saving moodel
joblib.dump(model,'model.sav')
