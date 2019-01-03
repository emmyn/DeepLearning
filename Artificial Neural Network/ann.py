
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding of categorical data 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features= [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] # Erase first columns 

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

## ACTUAL ANN

# Importing modules
import keras 
from keras.models import Sequential
from keras.layers import Dense         # Weight close to but to equal to 0

# Avoiding overfitting
from keras.layers import Dropout

# Initialize ann
classifier = Sequential()

# Input Layer (input_dim) and hidden layers (2)
classifier.add(Dense(units = 6,
                     activation = 'relu', 
                     kernel_initializer = 'uniform',
                     input_dim = 11))

# Avoiding overfitting
classifier.add(Dropout(
        rate = 0.1,         # Percentage of neuron deactivated
        ))

classifier.add(Dense(units = 6,
                     activation = 'relu', 
                     kernel_initializer = 'uniform'))

# Avoiding overfitting
classifier.add(Dropout(
        rate = 0.1,         # Percentage of neuron deactivated
        ))

# Output Layer
classifier.add(Dense(units = 1,
                     activation = 'sigmoid', 
                     kernel_initializer = 'uniform'))

# Compilation
classifier.compile(optimizer = 'adam', 
                   loss = 'binary_crossentropy',
                   metrics = ['accuracy'])

# Training Time
classifier.fit(X_train,
               y_train,
               batch_size = 10,
               epochs = 100)
# batch_size = 25
# epochs = 500
# to optimaze


## END OF ACUAL ANN


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Prediction for a specific user
new_pred = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 0, 40, 3, 60000, 2, 1, 1, 50000]])))
new_pred = (new_pred > 0.5)


# Evaluating real accuracy using cross validation
# Uses multiple small lots and takes the mean accuracy
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6,
                     activation = 'relu', 
                     kernel_initializer = 'uniform',
                     input_dim = 11))
    classifier.add(Dense(units = 6,
                     activation = 'relu', 
                     kernel_initializer = 'uniform'))
    classifier.add(Dense(units = 1,
                     activation = 'sigmoid', 
                     kernel_initializer = 'uniform'))
    classifier.compile(optimizer = 'adam', 
                   loss = 'binary_crossentropy',
                   metrics = ['accuracy'])
    # rmstrop to optimize
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, 
                             batch_size = 10,
                             epochs = 100)

precision = cross_val_score(estimator = classifier,
                            X = X_train,
                            y = y_train,
                            cv = 10    # arbitrary
                            )
mean = precision.mean()
std = precision.std()

# Optimazing parameters to increase accuracy 
from sklearn.model_selection import GridSearchCV

def build_classifier2(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6,
                     activation = 'relu', 
                     kernel_initializer = 'uniform',
                     input_dim = 11))
    classifier.add(Dense(units = 6,
                     activation = 'relu', 
                     kernel_initializer = 'uniform'))
    classifier.add(Dense(units = 1,
                     activation = 'sigmoid', 
                     kernel_initializer = 'uniform'))
    classifier.compile(optimizer = optimizer, 
                   loss = 'binary_crossentropy',
                   metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier2)

parameters = {"batch_size": [25, 32],
              "epochs": [100, 500],
              "optimizer": ["adam", "rmsprop"]}

gridSearch = GridSearchCV(estimator = classifier,
                          param_grid = parameters,
                          scoring = "accuracy",
                          cv = 10)
gridSearch = gridSearch.fit(X_train, y_train)

best_params = gridSearch.best_params_
best_precision = gridSearch.best_score_

 