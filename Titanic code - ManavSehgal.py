# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# 0.download data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]

# -------------------------------------------------------------------------------------------------------------------------
# 1.分析feature
# ---------------------------------------------------------------------------------------------------------------------------
# 1.1 把feature分为 categorical 和 numerical
# categorical feature：Categorical: 【Survived, Sex, and Embarked】. Ordinal: 【Pclass】.
# numberical feature：Continous: 【Age, Fare】. Discrete: 【SibSp, Parch】.

# 1.2 找出可能需要correcting的feature
# 1.2.1feature mixed data types： 【Ticket】is a mix of numeric and alphanumeric data types. 【Cabin】 is alphanumeric.
# 1.2.2 feature may contain errors or typos: 【name】feature may contain errors or typos as there are several ways used to describe a name including titles, round brackets, and quotes used for alternative or short names.
# 1.2.3 features contain blank, null or empty values : Cabin > Age > Embarked features contain a number of null values in that order for the training dataset.
#                                                      Cabin > Age are incomplete in case of test dataset.
print(train_df.columns.values)

# preview the data
pd.set_option('display.max_columns', None)
print(train_df.head())

print(train_df.tail())


# 1.3 确定feature会有哪几种值类型：Seven features are integer or floats. Six in case of test dataset.
#                               Five features are strings (object).
train_df.info()
print('_'*40)
test_df.info()

# ----------------------------------------------------------------------------------------------------------------------
# 2. 分析数据 （对数据有个大体的了解，然后可以进行一些假设，然后通过下一步的分析feature和survived关系进一步完善这些假设和做出使用哪个feature的决定。
# ---------------------------------------------------------------------------------------------------------------------------
#2.1 分析distribution of numerical feature value （主要用pandas.DataFrame.describe()函数 + 变化pencentiles的值）

# 先用 print(train_df.describe())看下整体的数据统计情况
print(train_df.describe())

# 然后根据数据出来的结果设计下percentiles 来进一步猜测某些feature
# 2.1.1 survived: 的值要么是1，要么是0. 这里62%的人是1，所以100%-62%=38%的人是survived的。
print(train_df.describe(percentiles=[.61, .62]))

# 2.1.2 Review Parch distribution using `percentiles=[.75, .8]` ，
# Most passengers (> 75%) did not travel with parents or children
print(train_df.describe(percentiles=[.75, .8]))

# 2.1.3 Age and Fare `[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99]`
# Few elderly passengers (<1%) within age range 65-80
# Fares varied significantly with few passengers (<1%) paying as high as $512.
print(train_df.describe(percentiles=[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99]))

# 2.2 分析distribution of categorical features
# 2.2.1 Sex variable as two possible values with 65% male (top=male, freq=577/count=891).
# 2.2.2 Embarked takes three possible values. S port used by most passengers (top=S)
# 2.2.3 Ticket feature has high ratio (22%)((count - unique)/count) of duplicate values (unique=681)
# 2.2.4 Cabin values have several dupicates across samples. Alternatively several passengers shared a cabin. (count = 204 , unique = 681)
print(train_df.describe(include=['O']))

# ---------------------------------------------------------------------------------------------------------------------------
# 3. 分析feature和survived 的关系 （得到一些假设和决定用哪些feature）
# ---------------------------------------------------------------------------------------------------------------------------
# 3.1 分析category feature 和 survived关系 - 表格
# 注意：1. only do so at this stage for features which do not have any empty values
#      2. only for features which are categorical (Sex), ordinal (Pclass) or discrete (SibSp, Parch) type

# 3.1.1 pclass , We observe significant correlation (>0.5) among Pclass=1 and Survived
print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# 3.1.2 Sex=female had very high survival rate at 74%
print(train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# 3.1.3 SibSp and Parch These features have zero correlation for certain values . We may want to create a new feature called Family based on Parch and SibSp to get total count of family members on board.
print(train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))


# 3.2 分析numerical feature和survived 的关系 - 可视化（一般对于数字类型的feature用可视化图更好分析，所以这里age用可视化来分析）

# 3.2.1 age (numerical features)
# should consider age
# should Complete the Age feature for null values
# We may want to create new feature for Age bands. This turns a continous numerical feature into an ordinal categorical feature.

g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20) # x轴的值间隔20
# plt.show() # 显示图片

# 3.3 分析numerical and ordinal features和survived 的关系 - 可视化

# 3.3.1 numerical and ordinal features （age)
# Pclass=3 had most passengers, however most did not survive
# Infant passengers in Pclass=2 and Pclass=3 mostly survived
# Most passengers in Pclass=1 survived
# Pclass varies in terms of Age distribution of passengers.
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20) #x轴是age，age间隔20岁
grid.add_legend();
# plt.show() # 显示图片

# 3.3.2  categorical features (Embarked , Pclass', 'Survived', 'Sex')
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep') # x轴是pclass ， y 轴是survived， 两条线一条表示female，一条表示male。 三个图分别代表不同embarked
grid.add_legend()
# plt.show() # 显示图片

# 3.4 分析categorical and numerical features和survived 的关系 - 可视化
# correlating Embarked (Categorical non-numeric), Sex (Categorical non-numeric), Fare (Numeric continuous), with Survived (Categorical numeric).
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
# plt.show() # 显示图片

# ----------------------------------------------------------------------------------------------------------------------------
# 4. 处理数据（Wrangle data ）
# ----------------------------------------------------------------------------------------------------------------------------
# 4.1 Correcting by dropping features
# drop the Cabin (correcting #2) and Ticket (correcting #1) features
# Note that where applicable we perform operations on both training and testing datasets together to stay consistent
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape

# 4.2 从一个feature中生成一个新的feature（Creating new feature extracting from existing）
# analyze if Name feature can be engineered to extract titles and test correlation between titles and survival, before dropping Name and PassengerId features.

# 4.2.1 把name . 符号之前的字符提取出来给title
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False) # 设置title=name里‘.’之前的字符， expand=false返回一个dataframe
print(pd.crosstab(train_df['Title'], train_df['Sex'])) #合成一个性别和title的表 , 显示出来

# 4.2.2 由于现在截取出来的title不上很make sense，用更make sense的title取代他
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

# 4.2.3 把title数字化
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

print(train_df.head()) # 数字化以后的title打印出来看下

# 现在可以drop Name feature from training and testing datasets. We also do not need the PassengerId feature in the training dataset.
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
print(train_df.shape, test_df.shape) # 打印出来看下是否真的drop了这两个feature

# 4.3 把categorical feature 数字化 【sex】
# converting Sex feature to a new feature called Gender where female=1 and male=0.
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

print(train_df.head())

# 4.4 填补缺失值 【age】
# 有3中方法可以填补缺失值
# 4.4.1 用平均值（mean）和 标准偏差（standard deviation）来填补
# 4.4.2 更准确一点的方法使用中位数，考虑和他有关的feature然后统计出来的age的中位数(median)。  median Age for Pclass=1 and Gender=0, Pclass=1 and Gender=1, and so on..
# 4.4.3 方法4.4.1和4.4.2的结合， use random numbers between mean and standard deviation, based on sets of Pclass and Gender combinations
# 方法4.4.1 和 4.4.3 可能会导致随机噪音（ introduce random noise into our models.），我们偏向于方法4.4.2

# 画出pclass，sex，age的关系图
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()

# preparing an empty array to contain guessed Age values based on Pclass x Gender combinations
guess_ages = np.zeros((2,3))
print(guess_ages)

# iterate over Sex (0 or 1) and Pclass (1, 2, 3) to calculate guessed values of Age for the six（2*3） combinations
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                               (dataset['Pclass'] == j + 1)]['Age'].dropna() # dropna函数丢弃age为空值

            # age_mean = guess_df.mean() # 方法1，用平均来填补
            # age_std = guess_df.std()  # 方法1，用标准差来填补
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std) # 方法3，用平均值和标准差范围内的随机数来填补

            age_guess = guess_df.median() # 方法2，用中位数来填补

            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5 # 把age_guess向0.5取整，比如0.3 -> 0.5 ， 1.2 -> 1.5 ， 因为后面是把age 取整处理，所以这样只要age的小数点后面>=8， 比如10.8 就会变成11

    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), \
                        'Age'] = guess_ages[i, j]  # 把 guess出来的age填进去

    dataset['Age'] = dataset['Age'].astype(int) # 把age变成int类型

print(train_df.head())


# 4.5 把连续性数值分成几个大类并且数字化

# 把age分成 分成几类。分别是青少年，青年，中年， 老年
# create Age bands and determine correlations with Survived
train_df['AgeBand'] = pd.cut(train_df['Age'], 5) #把age分成5个等间距的区间
print(train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)) #计算出每个区间ageband的survived的平均值

# 把age 字段数字化。feature用age分成的几类代替 + 每个类用一个数字代替
for dataset in combine:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
print(train_df.head())

#ageband只是个过渡状态，是为了把age 分成几类然后数字化。现在可以把ageband drop掉
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
print(train_df.head())

# 4.6 结合几个feature新建一个新的feature

# 4.6.1 new feature familysize : combines Parch and SibSp
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1 #如果是自己一个人则familysize = 1

print(train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# 4.6.2 new feature IsAlone : combines Parch and SibSp
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1 #isalone默认都是0 ，也就是不是alone，如果familysize==1说明只有自己一个人，所以设为1

print(train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())

# 我们已经用parch和sibsp生成了familysize和isalone这两个feature，所以我们可以抛弃parch和sibsp
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]
print(train_df.head())

# 4.6.3  new feature Age*Class --create an artificial feature combining Pclass and Age.
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

print(train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10))

# 4.7 处理categorical feature 【embarked】

# 填补缺失值， embarked缺失2个值，就用embarked最经常出现的值替代
# 查看embarked最经常出现的值
freq_port = train_df.Embarked.dropna().mode()[0] # 返回embarked 众数种的第一个，也就是最频繁出现的一个
print(freq_port)

#填补
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived',
                                                                                            ascending=False)
# 把embarked转为数字类型
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

print(train_df.head())


#4.8 处理numeric feature 【fare】

# 用中位数填补fare缺失的值
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True) #先用dropna过滤空值，然后计算出中位数，然后用中位数填补空值
test_df.head()

# 把fare分成几类
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4) #把fare分成4个等区间
print(train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True))

# fare字段数字化。feature用age分成的几类代替,同时每个类用一个数字代替
for dataset in combine:
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

print(train_df.head(10))

#-----------------------------------------------------------------------------------------------------------------------------
# 5.选择模型
#-----------------------------------------------------------------------------------------------------------------------------

# 5.0 数据正则化处理，这里面的例子里没有数据差别特别的例子，比如一个是三位数的数值，另一个feature是1位数的数值，这样就需要先正则化处理

# 5.1 数据处理，得到 X_train, Y_train, X_test.

# 把训练集里的survived 分到y，除了survived以外的分到x
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
# 把测试集里面的passengerid去掉
X_test  = test_df.drop("PassengerId", axis=1).copy()

print(X_train.shape, Y_train.shape, X_test.shape)

# 5.2 logistic regression
# predict
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print(acc_log)

# 看每个feature和survived相关性系数
# 结果看出 sex , title 是排名1，2的正相关性， Age*Class ， Pclass是排名1，2的负相关性
coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

print(coeff_df.sort_values(by='Correlation', ascending=False))

# 5.3 Support Vector Machines
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

# 5.4  k-Nearest Neighbors algorithm (or k-NN for short)
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn

# 5.5 Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian

# 5.6 Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron

# 5.7 Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc

# 5.8 Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd

# 5.9 Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree

# 5.10 Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


#-----------------------------------------------------------------------------------------------------------------------
# 模型评估
#------------------------------------------------------------------------------------------------------------------------
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest', 'Naive Bayes', 'Perceptron',
              'Stochastic Gradient Decent', 'Linear SVC',
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log,
              acc_random_forest, acc_gaussian, acc_perceptron,
              acc_sgd, acc_linear_svc, acc_decision_tree]})
print(models.sort_values(by='Score', ascending=False))


#--------------------------------------------------------------------------------------------------------------------------
# 提交
#--------------------------------------------------------------------------------------------------------------------------
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
# submission.to_csv('../output/submission.csv', index=False)