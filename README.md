# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary packages.

2. Read the given csv file and display the few contents of the data.
3. Assign the features for x and y respectively.

4. Split the x and y sets into train and test sets.

5. Convert the Alphabetical data to numeric using CountVectorizer.

6. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.

7. Find the accuracy of the model.

## Program:
```

Program to implement the SVM For Spam Mail Detection..
Developed by: ARSHATHA P
RegisterNumber:  212222230012

```
```
import chardet
file='/content/spam (1).csv'
with open(file, 'rb') as rawdata:
     print('Result output')
    result = chardet.detect(rawdata.read(10000))
result

import pandas as pd
data=pd.read_csv("/content/spam (1).csv",encoding="windows-1252")

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.feature_extraction.text import CountVectorizer 
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)
print("y_pred")
y_pred


from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print("accuracy")
accuracy
```
## Output:
Result 

![9 1](https://github.com/arshatha-palanivel/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118682484/66b4c0ad-b98a-46c1-ab3f-697783b8e1df)

data.head()

![9 2](https://github.com/arshatha-palanivel/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118682484/0283b6e0-f0b8-49d0-8d9e-0b0a6da830af)

data.info()

![9 3](https://github.com/arshatha-palanivel/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118682484/22fd6998-eb96-4925-9c57-2a17774550de)

data.isnull().sum()

![9 4](https://github.com/arshatha-palanivel/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118682484/b3634fe1-2f74-4994-9f58-9ddff6e7f407)

Y_prediction value

![9 5](https://github.com/arshatha-palanivel/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118682484/61f2218f-2b3f-4c97-8fd0-7bb06330ec25)

Accuracy value

![9 6](https://github.com/arshatha-palanivel/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118682484/9faa61e5-4c91-490b-8ea9-3e09d925c62f)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
