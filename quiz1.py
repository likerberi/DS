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


# pre-processing
train = train.drop(['Cabin'], axis=1)
train = train.drop(['Ticket'], axis=1)

test = test.drop(['Cabin'], axis=1)
test = test.drop(['Ticket'], axis=1)

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

print(train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

for dataset in train_test_merge:
    dataset['Title'] = dataset['Title'].astype(str)

    # Sex
    dataset['Sex'] = dataset['Sex'].astype(str)

    # Embarked (departing from)
    dataset['Embarked'] = dataset['Embarked'].fillna('S') # there are only 2 loss data.

train['Age'] = train['Age'].fillna(-0.5)
test['Age'] = test['Age'].fillna(-0.5)
bins = [-1, 0, 18, 66, 80, 100, 123] # , criteria from UN
labels = ['Unknown', 'Minor', 'Youth', 'Middle', 'Old', 'Longlived']
train['AgeGroup'] = pd.cut(train['Age'], bins, labels = labels)
test['AgeGroup'] = pd.cut(train['Age'], bins, labels = labels)

print(train.head(5))




