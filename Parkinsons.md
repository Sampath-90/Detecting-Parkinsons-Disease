#Detecting Parkinson's Disease with XGBoost

>Q. What is Parkinson's Disease?

>Ans: Parkinson’s disease is a progressive disorder of the central nervous system affecting movement and inducing tremors and stiffness. This is chronic and has no cure yet. It is a neurodegenerative disorder affecting dopamine-producing neurons in the brain.

>Q. What is XGBoost?

>Ans: XGBoost is a new Machine Learning algorithm designed with speed and performance in mind. XGBoost stands for eXtreme Gradient Boosting and is based on decision trees. In this project, we will import the XGBClassifier from the xgboost library; this is an implementation of the scikit-learn API for XGBoost classification.


```python
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

#Read the data
df=pd.read_csv('parkinsons.data')
df.head()

# Get the features and labels
# The features are all the columns except ‘status’, and the labels are those in the ‘status’ column.
features = df.loc[:, df.columns != 'status'].values[:, 1:]
labels = df.loc[:, 'status'].values

# Get the count of each label (0 and 1) in labels
print(labels[labels==1].shape[0], labels[labels==0].shape[0])
```
We have 147 ones and 48 zeros in the status column in our dataset.

Next, we initialize a MinMaxScaler and scale the features to between -1 and 1 to normalize them. The MinMaxScaler transforms features by scaling them to a given range. The fit_transform() method fits to the data and then transforms it. We don’t need to scale the labels.

```python
# Scale the features to between -1 and 1
scaler=MinMaxScaler((-1,1))
x=scaler.fit_transform(features)
y=labels
```

Now, split the dataset into training and testing sets keeping 20% of the data for testing.

```python
# Split the dataset
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=7)
```

Initialize an XGBClassifier and train the model.

```python
# Train the model
model=XGBClassifier()
model.fit(x_train,y_train)

# generate y_pred (predicted values for x_test) and calculate the accuracy for the model.
y_pred=model.predict(x_test)
print(accuracy_score(y_test, y_pred)*100)
```

#Summary
In this project, we learned to detect the presence of Parkinson’s Disease in individuals using various factors. We used an XGBClassifier for this and made use of the sklearn library to prepare the dataset. This gives us an accuracy of 94.87%.