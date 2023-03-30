
#contains z-norm.. uses a smaller dataset
#This uses a smaller dataset

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import csv
import numpy as np
import matplotlib.pyplot as plt

numberOfFeatures = 3
file_CSV = open("House_Rent_Dataset_Edited2.csv")
data_CSV = csv.reader(file_CSV)

#Stored data from CSV file in a list and then converted the list into a numpy array
data = list(data_CSV)
data = data[1:]
data = np.array(data)

#split features and target into data/target variables
[data, target] = np.split(data,[numberOfFeatures],axis=1)

#Stored the data inside of training and testing data
X_train, X_test, y_train, y_test = train_test_split( data, target , test_size=0.20, random_state=42)

#Casted the type of data to be an int
y_train = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)
X_train = X_train.astype(np.int64)
X_test = X_test.astype(np.int64)

scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.fit_transform(X_test)

#Balanced the weights of the data 
lr_model = LinearRegression() 

#Fitted the training data into the model
lr_model.fit(X_norm,y_train.ravel())

#Predicted the values for X_test using the model
y_pred = lr_model.predict(X_test_norm)
X_features = ["BHK","Size","Number of Bathrooms"]

print("Model score test data",lr_model.score(X_test_norm,y_test))
print("Model score train data",lr_model.score(X_norm,y_train))

fig,ax=plt.subplots(1,numberOfFeatures,figsize=(12,3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_test_norm[:,i],y_test, label = 'target')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_test_norm[:,i],y_pred, label = 'predict')
ax[0].set_ylabel("Cost of rent"); ax[0].legend();
fig.suptitle("Target versus prediction on test data")
plt.show()


"""

Removed Posted on, Area Locality

Changes
Area type:
Built Area = 0
Super Area = 1
Carpet Area = 2

City:
Bangalore = 0
Chennai = 1
Delhi = 2
Hyderabad = 3
Kolkata = 4
Mumbai = 5

Furnished:
Furnished = 0
Semi-Furnished = 1
Unfurnished = 2

Tenant
Bachelors = 0
Bachelors/Family = 1
Family = 2

Point of contact
Contact Owner = 0
Contact Agent = 1
Contact Builder = 2


"""

