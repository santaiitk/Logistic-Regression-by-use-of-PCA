# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 13:30:17 2025

@author: Santosh Chaurasia
"""

# Import all linaraies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Importing Dataset
dataset = pd.read_csv(r"E:\AIML\NareshIT\My\Adult\Adult.csv")
dataset.head()
# Encode as NaN
dataset[[dataset =='?']] = np.nan
#Impute Missing Value as Mode imputation
for i in ['workclass', 'occupation', 'native.country']:
    dataset[i].fillna(dataset[i].mode()[0], inplace = True)
#Checking Missing Value
dataset.isnull().sum()
##############################################################################################
# Set feature vector & Target Vector
'''
x = dataset.iloc[ :, :-1]
y = dataset.iloc[:, -1]

# Split data into separate training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

# Feature Engineering
# Encode categorical variables
from sklearn import preprocessing

categorical = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
for feature in categorical:
        le = preprocessing.LabelEncoder()
        X_train[feature] = le.fit_transform(X_train[feature])
        X_test[feature] = le.transform(X_test[feature])

# Feature Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = x.columns)

X_test = pd.DataFrame(scaler.transform(X_test), columns = x.columns)
        
#Logistic Regression model with all features

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('Logistic Regression accuracy score with all the features: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

## Logistic Regression with PCA
from sklearn.decomposition import PCA
pca = PCA()
X_train = pca.fit_transform(X_train)
pca.explained_variance_ratio_
'''
###############################################################################################
'''
## Logistic Regression with first 13 features(After deleting 2 columns with very low variance )

x = dataset.drop(['income','native.country'], axis=1)
y = dataset['income']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)


from sklearn import preprocessing

categorical = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex']
for feature in categorical:
        le = preprocessing.LabelEncoder()
        X_train[feature] = le.fit_transform(X_train[feature])
        X_test[feature] = le.transform(X_test[feature])

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = x.columns)

X_test = pd.DataFrame(scaler.transform(X_test), columns = x.columns)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('Logistic Regression accuracy score with the first 13 features: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
'''
############################################################################################################
"""
### Accuracy reduce to 0.8218 to 0.8213
'''Now Consider variance as approximately 7 %. Again we can drop 3 column from dataset'''

##Logistic Regression with first 12 features
x = dataset.drop(['income','native.country', 'hours.per.week'], axis=1)
y = dataset['income']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

from sklearn import preprocessing
categorical = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex']
for feature in categorical:
        le = preprocessing.LabelEncoder()
        X_train[feature] = le.fit_transform(X_train[feature])
        X_test[feature] = le.transform(X_test[feature])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = x.columns)

X_test = pd.DataFrame(scaler.transform(X_test), columns = x.columns)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Logistic Regression accuracy score with the first 12 features: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
    
'''Logistic Regression accuracy score with the first 12 features: 0.8227'''
"""
###############################################################################################################################
"""
##Logistic Regression with first 11 features
# Now Consider variance as approximately 12 %. Again we can drop 3 column from dataset

x = dataset.drop(['income','native.country', 'hours.per.week', 'capital.loss'], axis=1)
y = dataset['income']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

from sklearn import preprocessing
categorical = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex']
for feature in categorical:
        le = preprocessing.LabelEncoder()
        X_train[feature] = le.fit_transform(X_train[feature])
        X_test[feature] = le.transform(X_test[feature])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = x.columns)

X_test = pd.DataFrame(scaler.transform(X_test), columns = x.columns)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Logistic Regression accuracy score with the first 12 features: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

'''Logistic Regression accuracy score with the first 12 features: 0.8186'''
"""
####################################################################################################################################
'''We can see that accuracy has significantly decreased to 0.8187 if I drop the last three features.
Avove approach is good for small dataset dimensions.
For large data set number dimensions by use of large portion variance.
Our aim is to maximize the accuracy. We get maximum accuracy with the first 12 features and the accuracy is 0.8227.
'''
x = dataset.drop(['income'], axis=1)
y = dataset['income']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

from sklearn import preprocessing
categorical = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
for feature in categorical:
        le = preprocessing.LabelEncoder()
        X_train[feature] = le.fit_transform(X_train[feature])
        X_test[feature] = le.transform(X_test[feature])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = x.columns)
from sklearn.decomposition import PCA
pca= PCA()
pca.fit(X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
dim = np.argmax(cumsum >= 0.90) + 1
print('The number of dimensions required to preserve 90% of variance is',dim)


##The number of dimensions required to preserve 90% of variance is 12

##Plot explained variance ratio with number of dimensions
plt.figure(figsize=(8,6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()

"""Comment
The above plot shows that almost 90% of variance is explained by the first 12 components."""