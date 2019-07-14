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

bar_chart('Pclass')
bar_chart('Sex')
# bar_chart('Age')
# bar_chart('SibSp')
# bar_chart('Parch')
bar_chart('Embarked')