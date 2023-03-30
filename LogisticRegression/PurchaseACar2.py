#This dataset has normalized data 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import csv
import numpy as np
import matplotlib.pyplot as plt


file_CSV = open("car_data1.csv")
data_CSV = csv.reader(file_CSV)

#Stored data from CSV file in a list and then converted the list into a numpy array
data = list(data_CSV)
data = data[1:]
data = np.array(data)

#split features and target into data/target variables
[data, target] = np.split(data,[3],axis=1)

#Stored the data inside of training and testing arrays
X_train, X_test, y_train, y_test = train_test_split( data, target , test_size=0.20, random_state=42)

#Casted the type of data to be an int
y_train = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)
X_train = X_train.astype(np.int64)
X_test = X_test.astype(np.int64)

#Normalize the data because the difference in scale among variables are large
scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.fit_transform(X_test)

#Create the model
lr_model = LogisticRegression() 

#Fitted the training data into the model
lr_model.fit(X_norm,y_train.ravel())

#Predicted the values for X_test using the model
y_pred = lr_model.predict(X_test_norm)
X_features = ["Gender","Age","Annual Salary"]

#Scored the model. 0 being innacurate 1 being accurate
print("Model score test data",lr_model.score(X_test_norm,y_test))
print("Model score train data",lr_model.score(X_norm,y_train))

fig,ax=plt.subplots(1,3,figsize=(12,3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_test[:,i],y_test, label = 'target')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_test[:,i],y_pred, label = 'predict')
ax[0].set_ylabel("Purchased a car?"); ax[0].legend();
fig.suptitle("Target versus prediction on test data (Not Normalized)")
plt.show()
