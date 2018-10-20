from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import numpy as np

data = np.genfromtxt("data_set.csv",delimiter=",")

np.random.shuffle(data)

m = int(data.shape[0] * 0.7) + 1
n = data.shape[1]

X_train = data[:m,:n-1]/255
y_train = data[:m,n-1:].ravel()

X_test = data[m:,:n-1]/255
y_test = data[m:,n-1:].ravel()

nn_clf = MLPClassifier(hidden_layer_sizes=(20,),activation='logistic',solver='lbfgs',verbose=True)

nn_clf.fit(X_train,y_train)

y_pred = nn_clf.predict(X_test)

acc = accuracy_score(y_test,y_pred)
print("Classifier accuracy = " +str(acc * 100)+"%")

joblib.dump(nn_clf,"nn_clf.jolib")
print("Classifier saved")
