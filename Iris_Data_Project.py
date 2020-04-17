# Iris Dataset Analysis - Model Comparison

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

# Applying the KernelPCA to the dataset

from sklearn.decomposition import KernelPCA
pca = KernelPCA(kernel = 'rbf', n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# =============================================================================
# 
# =============================================================================

# Fitting Logistic Regression to the dataset

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Predicting the test set results

y_pred_LR = lr.predict(X_test)

# Making the confusion matrix

from sklearn.metrics import confusion_matrix
cm_LR = confusion_matrix(y_test, y_pred_LR)

# Model evaluation

from sklearn.model_selection import cross_val_score
accuracies_LR = cross_val_score(lr, X = X_train, y = y_train, cv = 10)
accuracies_LR_avg = accuracies_LR.mean()
accuracies_LR_std = accuracies_LR.std()

# Visualising the training set 

from matplotlib.colors import ListedColormap

X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 1, stop = X_set[:,0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:,1].min() - 1, stop = X_set[:,1].max() + 1, step = 0.01))
plt.contourf(X1, X2, lr.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, 
             cmap = ListedColormap(('pink','purple','grey')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('pink','purple','grey'))(i), label = j)
plt.legend(loc = 'upper left', bbox_to_anchor = (1,1.03))
plt.title('Training set for Logistic Regression')
plt.show()

# Visualising the test set 

X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 1, stop = X_set[:,0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:,1].min() - 1, stop = X_set[:,1].max() + 1, step = 0.01))
plt.contourf(X1, X2, lr.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, 
             cmap = ListedColormap(('pink','purple','grey')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('pink','purple','grey'))(i), label = j)
plt.legend(loc = 'upper left', bbox_to_anchor = (1,1.03))
plt.title('Test set for Logistic Regression')
plt.show()

# =============================================================================
# 
# =============================================================================

# Fitting the KNN to the dataset

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 11, algorithm = 'ball_tree')
knn.fit(X_train, y_train)

# Predicting the test set results

y_pred_knn = knn.predict(X_test)

# Making the confusion matrix

from sklearn.metrics import confusion_matrix
cm_LR = confusion_matrix(y_test, y_pred_knn)

# Model evaluation

from sklearn.model_selection import cross_val_score
accuracies_knn = cross_val_score(knn, X = X_train, y = y_train, cv = 10)
accuracies_knn_avg = accuracies_knn.mean()
accuracies_knn_std = accuracies_knn.std()

# Parameter tuning

from sklearn.model_selection import GridSearchCV
parameters_knn = [{'n_neighbors':[5,8,10,11,12,13,14,15], 'algorithm':['ball_tree','kd_tree','brute','auto']}]
grid_search_knn = GridSearchCV(knn, param_grid = parameters_knn, scoring = 'accuracy', cv=10, n_jobs = -1)
grid_search_knn.fit(X_train, y_train)
optimal_params_knn = grid_search_knn.best_params_
optimal_score_knn = grid_search_knn.best_score_

# Visualising the training set 

from matplotlib.colors import ListedColormap

X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 1, stop = X_set[:,0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:,1].min() - 1, stop = X_set[:,1].max() + 1, step = 0.01))
plt.contourf(X1, X2, knn.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, 
             cmap = ListedColormap(('pink','purple','grey')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('pink','purple','grey'))(i), label = j)
plt.legend(loc = 'upper left', bbox_to_anchor = (1,1.03))
plt.title('Training set for KNN')
plt.show()

# Visualising the test set

X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 1, stop = X_set[:,0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:,1].min() - 1, stop = X_set[:,1].max() + 1, step = 0.01))
plt.contourf(X1, X2, knn.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, 
             cmap = ListedColormap(('pink','purple','grey')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('pink','purple','grey'))(i), label = j)
plt.legend(loc = 'upper left', bbox_to_anchor = (1,1.03))
plt.title('Test set for KNN')
plt.show()

# =============================================================================
# 
# =============================================================================

# Fitting the SVM to the dataset

from sklearn.svm import SVC
svc = SVC(kernel = 'rbf', C = 100, gamma = 1)
svc.fit(X_train, y_train)

# Predicting the test set results

y_pred_svc = svc.predict(X_test)

# Making the confusion matrix

from sklearn.metrics import confusion_matrix
cm_svc = confusion_matrix(y_test, y_pred_svc)

# Model evaluation

from sklearn.model_selection import cross_val_score
accuracies_svc = cross_val_score(svc, X = X_train, y = y_train, cv = 10)
accuracies_svc_avg = accuracies_svc.mean()
accuracies_svc_std = accuracies_svc.std()

# Parameter tuning

from sklearn.model_selection import GridSearchCV
parameters_svc = [{'C':[1,10,100], 'kernel':['linear']}, 
              {'C':[1,10,100], 'kernel':['rbf'], 'gamma':[1,0.5,0.1]}, 
              {'C':[1,10,100], 'kernel':['poly'], 'degree':[2,3,4] ,'gamma':[1,0.5,0.1]},
              {'C':[1,10,100], 'kernel':['sigmoid'], 'gamma':[1,0.5,0.1]}]
grid_search_svc = GridSearchCV(svc, param_grid = parameters_svc, scoring = 'accuracy', cv=10, n_jobs = -1)
grid_search_svc.fit(X_train, y_train)
optimal_params_svc = grid_search_svc.best_params_
optimal_score_svc = grid_search_svc.best_score_

# Visualising the training set 

from matplotlib.colors import ListedColormap

X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 1, stop = X_set[:,0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:,1].min() - 1, stop = X_set[:,1].max() + 1, step = 0.01))
plt.contourf(X1, X2, svc.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, 
             cmap = ListedColormap(('pink','purple','grey')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('pink','purple','grey'))(i), label = j)
plt.legend(loc = 'upper left', bbox_to_anchor = (1,1.03))
plt.title('Training set for SVC')
plt.show()

# Visualising the test set

X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 1, stop = X_set[:,0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:,1].min() - 1, stop = X_set[:,1].max() + 1, step = 0.01))
plt.contourf(X1, X2, svc.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, 
             cmap = ListedColormap(('pink','purple','grey')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('pink','purple','grey'))(i), label = j)
plt.legend(loc = 'upper left', bbox_to_anchor = (1,1.03))
plt.title('Test set for SVC')
plt.show()

# =============================================================================
# 
# =============================================================================

# Fitting the Naive Bayes to the dataset

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)

# Predicting the test set results

y_pred_nb = svc.predict(X_test)

# Making the confusion matrix

from sklearn.metrics import confusion_matrix
cm_nb = confusion_matrix(y_test, y_pred_nb)

# Model evaluation

from sklearn.model_selection import cross_val_score
accuracies_nb = cross_val_score(nb, X = X_train, y = y_train, cv = 10)
accuracies_nb_avg = accuracies_nb.mean()
accuracies_nb_std = accuracies_nb.std()

# Visualising the training set 

from matplotlib.colors import ListedColormap

X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 1, stop = X_set[:,0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:,1].min() - 1, stop = X_set[:,1].max() + 1, step = 0.01))
plt.contourf(X1, X2, nb.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, 
             cmap = ListedColormap(('pink','purple','grey')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('pink','purple','grey'))(i), label = j)
plt.legend(loc = 'upper left', bbox_to_anchor = (1,1.03))
plt.title('Training set for Naive Bayes')
plt.show()

# Visualising the test set

X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 1, stop = X_set[:,0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:,1].min() - 1, stop = X_set[:,1].max() + 1, step = 0.01))
plt.contourf(X1, X2, nb.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, 
             cmap = ListedColormap(('pink','purple','grey')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('pink','purple','grey'))(i), label = j)
plt.legend(loc = 'upper left', bbox_to_anchor = (1,1.03))
plt.title('Test set for Naive Bayes')
plt.show()

# =============================================================================
# 
# =============================================================================

# Random Forest Classification (We do not use Feature Scaling)

# Importing the dataset

ds = pd.read_csv('IRIS.csv')
X = ds.iloc[:,:-1].values
y = ds.iloc[:,-1].values

# There is no missing data
# Encoding categorical variables

from sklearn.preprocessing import LabelEncoder
le_y = LabelEncoder()
y = le_y.fit_transform(y)

# Splitting the dataset into the training set and test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# We do not use Feature Scaling

# Applying the PCA

from sklearn.decomposition import KernelPCA
pca = KernelPCA(n_components = 2, kernel = 'rbf')
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Fitting the Random Forest Regression to the dataset

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 25, criterion = 'gini')
rfc.fit(X_train, y_train)

# Predicting the test set results

y_pred_rfc = rfc.predict(X_test)

# Making the confusion matrix

from sklearn.metrics import confusion_matrix
cm_rfc = confusion_matrix(y_test, y_pred_rfc)

# Model evaluation

from sklearn.model_selection import cross_val_score
accuracies_rfc = cross_val_score(rfc, X = X_train, y = y_train, cv = 10)
accuracies_rfc_avg = accuracies_rfc.mean()
accuracies_rfc_std = accuracies_rfc.std()

# Paramater tuning

from sklearn.model_selection import GridSearchCV
parameters_rfc = [{'n_estimators':[10,25,50,100], 'criterion':['gini','entropy']}]
grid_search_rfc = GridSearchCV(rfc, param_grid = parameters_rfc, scoring = 'accuracy', cv = 10, n_jobs = -1)
grid_search_rfc.fit(X_train, y_train)
optimal_params_rfc = grid_search_rfc.best_params_
optimal_score_rfc = grid_search_rfc.best_score_

# Visualising the training set 

from matplotlib.colors import ListedColormap

X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 1, stop = X_set[:,0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:,1].min() - 1, stop = X_set[:,1].max() + 1, step = 0.01))
plt.contourf(X1, X2, rfc.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, 
             cmap = ListedColormap(('pink','purple','grey')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('pink','purple','grey'))(i), label = j)
plt.legend(loc = 'upper left', bbox_to_anchor = (1,1.03))
plt.title('Training set for Random Forest')
plt.show()

# Visualising the test set

X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 1, stop = X_set[:,0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:,1].min() - 1, stop = X_set[:,1].max() + 1, step = 0.01))
plt.contourf(X1, X2, rfc.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, 
             cmap = ListedColormap(('pink','purple','grey')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('pink','purple','grey'))(i), label = j)
plt.legend(loc = 'upper left', bbox_to_anchor = (1,1.03))
plt.title('Test set for Random Forest')
plt.show()