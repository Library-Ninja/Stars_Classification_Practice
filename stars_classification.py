# -*- coding: utf-8 -*-
"""
Created on Thu May 14 11:46:23 2026

@author: 37cho
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM

from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


# df = pd.read_csv("C:\\Users\\37cho\\OneDrive\\Documents\\Code Projects\\cleaned_star_data.csv")
df = pd.read_csv("https://raw.githubusercontent.com/Library-Ninja/Stars_Classification_Practice/main/cleaned_star_data.csv")

df.columns = ['temperature', 'luminosity', 'radius', 'abs_magnitude', 'star_type', 'color', 'spectral_class']

#Column type identification
cat = ['color', 'spectral_class']
quant = ['temperature', 'luminosity', 'radius', 'abs_magnitude']
features = np.append(quant, cat)
target = 'star_type'

#Data Cleaning - remove records with empty values
df = df.replace(" ", np.nan)
df = df.dropna()
df[quant] = df[quant].astype(float)

#Creating training and test sets
# Note: Explore sklearn train_test_split as well!
train = df.sample(frac=.7)
test = df[~df.index.isin(train.index)]

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

#OrdinalEncoder exploration. Note: Ordinal encoding replaces each unique cat value with an integer, leaving all data in one column. It fits this situation because color and spectral class have a meaningful order (in terms of star temperature)
# print(df[cat])
# oe = OrdinalEncoder(categories=[['Red', 'Yellow-White', 'White', 'Blue-White', 'Blue'], ['M', 'K', 'G', 'F', 'A', 'B', 'O']])
# data = oe.fit_transform(df[cat])
# print(oe.categories_)
# print(pd.DataFrame(data))

#Classification
transformer = make_column_transformer(
    (OrdinalEncoder(categories=[['Red', 'Yellow-White', 'White', 'Blue-White', 'Blue'], ['M', 'K', 'G', 'F', 'A', 'B', 'O']]), cat),
    #(OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat),
    remainder = "passthrough"
    )

def create_pipeline(algorithm):
    pipeline = make_pipeline(
        transformer,
        StandardScaler(),
        algorithm # Classifier algorithm
        )
    return pipeline

'''
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

# Summary of metrics shown in classification_report (refer to this article: https://towardsdatascience.com/micro-macro-weighted-averages-of-f1-score-clearly-explained-b603420b292f/)
# accuracy = micro avg
# macro avg = average of scores for each class, unweighted (good for imbalanced datasets)
# weighted avg = average of scores for each class weighted by the support value (good for weighing common values more)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
'''

def compute_f1_score(algorithm):
    f1_scores = cross_val_score(
        create_pipeline(algorithm),
        X=X_train,
        y=y_train,
        scoring="f1_macro",
        cv=10
        )
    return f1_scores.mean()

# Find best classification algorithm
algorithms = [KNeighborsClassifier(), LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier()]
for a in algorithms:
    print(compute_f1_score(a))
#Conclusion: Random Forest produces highest f1_macro score


#Hyperparameter tuning for Random Forest Model

#n_estimators (number of trees in the forest)
# f1 = pd.Series([])
# for n in range(1, 101, 10):
#     f1[str(n)] = compute_f1_score(RandomForestClassifier(n_estimators=n))
# f1.plot()
#Conclusion: I found that at around n=30, the scores start to level off

# criterion = ['gini', 'entropy', 'log_loss']
# for c in criterion:
#     print(compute_f1_score(RandomForestClassifier(n_estimators=30, criterion=c)))
    
param_grid = {
    'n_estimators': [10, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

# grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=10)

pipeline = make_pipeline(
    transformer,
    StandardScaler()
    )
X_train_stand = pipeline.fit_transform(X_train)

# grid_search.fit(X_train_stand, y_train)

# print(grid_search.best_params_)
