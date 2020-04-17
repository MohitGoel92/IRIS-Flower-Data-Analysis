# Iris Dataset Analysis

# Importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset

ds = pd.read_csv('IRIS.csv')
X = ds.iloc[:,:-1].values
y = ds.iloc[:,-1].values

# There is no missing data

# Encoding categorical data

from sklearn.preprocessing import LabelEncoder
le_y = LabelEncoder()
y = le_y.fit_transform(y)

# Splitting the dataset into the training set and test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Applying the PCA to the dataset

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

# Fitting the XGBoost to the dataset

from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train, y_train)

# Predicting the test set results

y_pred = xgb.predict(X_test)

# Model Evaluation

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = xgb, X = X_train, y = y_train, cv = 10)
avg_accuracy = accuracies.mean()
std_accuracy = accuracies.std() 

# Making the confusion matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the training set

from matplotlib.colors import ListedColormap

X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 1, stop = X_set[:,0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:,1].min() - 1, stop = X_set[:,1].max() + 1, step = 0.01))
plt.contourf(X1, X2, xgb.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, 
             cmap = ListedColormap(('pink', 'purple','gold')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('pink', 'purple', 'gold'))(i), label = j)
plt.legend(loc = 'upper left', bbox_to_anchor = (1, 1.03))
plt.title('Training Set')
plt.show()

# Visualising the test set

X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 1, stop = X_set[:,0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:,1].min() - 1, stop = X_set[:,1].max() + 1, step = 0.01))
plt.contourf(X1, X2, xgb.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, 
             cmap = ListedColormap(('pink', 'purple','gold')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('pink', 'purple', 'gold'))(i), label = j)
plt.legend(loc = 'upper left', bbox_to_anchor = (1, 1.03))
plt.title('Test Set')
plt.show()