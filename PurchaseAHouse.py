#initial 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import csv
import numpy as np
import matplotlib.pyplot as plt


file_CSV = open("House_Rent_Dataset_Edited.csv")
data_CSV = csv.reader(file_CSV)

#Stored data from CSV file in a list and then converted the list into a numpy array
data = list(data_CSV)
data = data[1:]
data = np.array(data)

print(data)


#split features and target into data/target variables
[data, target] = np.split(data,[8],axis=1)

#Stored the data inside of training and testing data
X_train, X_test, y_train, y_test = train_test_split( data, target , test_size=0.20, random_state=42)

#Casted the type of data to be an int
y_train = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)
X_train = X_train.astype(np.int64)
X_test = X_test.astype(np.int64)

for d in [y_train, y_test, X_train, X_test]:
    print(d.shape)

#Balanced the weights of the data 
lr_model = LinearRegression() 

#Fitted the training data into the model
lr_model.fit(X_train,y_train.ravel())

#Predicted the values for X_test using the model
y_pred = lr_model.predict(X_test)
X_features = ["BHK","Size", "Area Type", "City", "Furnished", "Tenant Preferred", "Number of Bathrooms", "Point of Contact"]

print("Model score test data",lr_model.score(X_test,y_test))
print("Model score train data",lr_model.score(X_train,y_train))

fig,ax=plt.subplots(1,8,figsize=(12,3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_test[:,i],y_test, label = 'target')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_test[:,i],y_pred, label = 'predict')
ax[0].set_ylabel("Cost of rent"); ax[0].legend();
fig.suptitle("Target versus prediction on test data")
plt.show()



#Notes

#Left to do:
#1. increase the weights of a feature
#3 Explain the model score
#4 try to reduce the model score by removing the Gender?
#4.5 try to make the model better by increasing weight or something??
#5 Explain changes made to dataset
    #removed posted on, floor, area locality 
    #shifted rent to the end


"""
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

