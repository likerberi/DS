import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')
#train.head()
#train.info()
#train.isnull().sum()

sns.set()

def bar_chart(feature) :
    survived = train[train['Survived'] == 1][feature].value_counts()
    #feature ~ counting.
    dead = train[train['Survived'] == 0][feature].value_counts()

    df = pd.DataFrame([survived, dead])
    df.index = ['Survived', 'Dead']
    df.plot(kind='bar', stacked=True, figsize=(10, 5))

    plt.boxplot(df.values)
    plt.show()

# bar_chart('Pclass')
# bar_chart('Sex')
# # bar_chart('Age')
# # bar_chart('SibSp')
# # bar_chart('Parch')
# bar_chart('Embarked')

# print(train.head())

train_test_merge = [train, test]

# Name area, categorizing.
for dataset in train_test_merge:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.')

# print(train.head(5))
# print(pd.crosstab(train['Title'], train['Sex']))

for dataset in train_test_merge:
    dataset['Title'] = dataset['Title'].replace([
        'Capt', 'Col', 'Countess', 'Don', 'Jonkheer', 'Lady', 'Major', 'Rev', 'Sir', 'Dr'
    ], 'Other')
    # French -> English Mlle, Mme
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    # No Discrimination but pre-category...
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

# print(train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
title_masking = {"Mr":1, "Miss":2, "Mrs":3, "Master":4, "Royal":5, "Rare":6}
for dataset in train_test_merge:
    dataset['Title'] = dataset['Title'].map(title_masking)
    dataset['Title'] = dataset['Title'].fillna(0)

for dataset in train_test_merge:
    # dataset['Title'] = dataset['Title'].astype(str)
    #
    # # Sex
    # dataset['Sex'] = dataset['Sex'].astype(str)

    # Embarked (departing from)
    dataset['Embarked'] = dataset['Embarked'].fillna('S') # there are only 2 loss data.

gender_masking = {"female":1, "male":0}
for dataset in train_test_merge:
    dataset["Sex"] = dataset['Sex'].map(gender_masking)

embarked_masking = {"S":1, "C":2, "Q":3}
train['Embarked'] = train['Embarked'].map(embarked_masking)
test['Embarked'] = test['Embarked'].map(embarked_masking)

train['Age'] = train['Age'].fillna(-0.5)
test['Age'] = test['Age'].fillna(-0.5)
bins = [-1, 0, 18, 66, 80, 100, 123] # , criteria from UN
labels = ['Unknown', 'Minor', 'Youth', 'Middle', 'Old', 'Longlived']
train['AgeGroup'] = pd.cut(train['Age'], bins, labels = labels)
test['AgeGroup'] = pd.cut(train['Age'], bins, labels = labels)

age_masking = {1:"Minor", 2:"Youth", 3:"Middle", 4:"Old", 5:"Longlived"}
train["AgeGroup"] = train["AgeGroup"].map(age_masking)
test["AgeGroup"] = test["AgeGroup"].map(age_masking)

# print(train.head(5))

# print(train[['Pclass', 'Fare']].groupby(['Pclass'], as_index=False).mean())
# 13.67555

for dataset in train_test_merge:
    dataset['Fare'] = dataset['Fare'].fillna(13.6755)

# print(train[['Pclass', 'Fare']].groupby(['Fare'], as_index=False).mean())

for dataset in train_test_merge:
    dataset.loc[dataset['Fare'] <= 7.854, 'Fare'] = 0 # Pclass 0
    dataset.loc[(dataset['Fare'] > 7.854) & (dataset['Fare'] <= 10.5), 'Fare'] = 1  # Pclass 1
    dataset.loc[(dataset['Fare'] > 10.5) & (dataset['Fare'] <= 21.679), 'Fare'] = 2  # Pclass 2
    dataset.loc[(dataset['Fare'] > 21.679) & (dataset['Fare'] < 39.668), 'Fare'] = 3  # Pclass 3
    dataset.loc[(dataset['Fare'] > 39.668),'Fare'] = 4  # Pclass 4

for dataset in train_test_merge:
    dataset["Family"] = dataset["Parch"] + dataset["SibSp"]

features_drop = ['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId'], axis=1)

# print(train.head())
# print(test.head())

train = pd.get_dummies(train)
test = pd.get_dummies(test)

train_label = train['Survived']
train_data = train.drop('Survived', axis=1)
test_data = test.drop("PassengerId", axis=1).copy()


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.utils import shuffle
train_data, train_label = shuffle(train_data, train_label, random_state = 5)

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=5)

clf = RandomForestClassifier(n_estimators=13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, train_label, cv=k_fold, n_jobs=1, scoring=scoring)
# print(np.mean(score) * 100)
clf.fit(train_data, train_label)
prediction = clf.predict(test_data)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": prediction
})

submission.to_csv ('submission_result.csv', index=False)
# LOCAL :: 81%
# KAGGLE :: 75%

# def train_and_test(model):
#     model.fit(train_data, train_label)
#     prediction = model.predict(test_data)
#     accuracy = round(model.score(train_data, train_label) * 100, 2)
#     print("Pred: ", accuracy, "%_by")
#     return prediction

# print(train_data.shape)
# print(train_label.shape)
# print(train_data.shape)
# knn_pred_4 = train_and_test(KNeighborsClassifier(n_neighbors=4))
#
# rf_pred = train_and_test(RandomForestClassifier(n_estimators=100))
