"""This program uses the data from csv to train the MLP model using Sklearn """

#importing required libraries 
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
import numpy as np
from plot_learning_curve import plot_learning_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit

#loading data from csv
data = np.genfromtxt("data_set.csv",delimiter=",")

#shuffling data for better training result
np.random.shuffle(data)
n = data.shape[1]
m = int(data.shape[0] * 0.6)

#splitting label fro mimage data
X = data[:,:n-1]/255
y = data[:,n-1:].ravel()

#splitting train, test and validation data
X_train = X[:m,:]
y_train = y[:m]

X_test = X[m:,:]
y_test = y[m:].ravel()

m_val = int(X_test.shape[0] * 0.5)

X_val = X_test[m_val:,:]
y_val = y_test[m_val:]

X_test = X_test[:m_val,:]
y_test = y_test[:m_val]

#initialising the MLP 
nn_clf = MLPClassifier(hidden_layer_sizes=(20),alpha = 0.3,activation='logistic',solver='lbfgs')

#plotting the learing curve for the model using plot_learing_curve defined in scikit documentation
plot_learning_curve(nn_clf, "NN Learning Curve", X, y)
plt.show()

#training the model
nn_clf.fit(X_train,y_train)

#validating on the validation set
acc = nn_clf.score(X_val, y_val)
print("Classifier accuracy on validation = " +str(acc * 100)+"%")

#testing on the test set
acc = nn_clf.score(X_test, y_test)
print("Classifier accuracy on test = " +str(acc * 100)+"%")

#saving the model
joblib.dump(nn_clf,"nn_clf.joblib")
print("Classifier saved")

