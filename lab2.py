import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import random

columns = ['age', 'workclass', 'fnlwgt', 'education', 'educationNum', 'maritalStatus', 'occupation', 'relationship', 'race', 'sex',
           'capitalGain', 'capitalLoss', 'hoursPerWeek', 'nativeCountry', 'income']
df_train_set = pd.read_csv('./adult.data', names=columns)
df_test_set = pd.read_csv(
    './adult.test', names=columns, skiprows=1)  # 第一行是非法数据

# print(df_train_set.head())
# print(df_test_set.head())
df_train_set.to_csv('./train_adult.csv', index=False)
df_test_set.to_csv('./test_adult.csv', index=False)

df_train_set = pd.read_csv('./train_adult.csv')

# fnlwgt列用处不大，educationNum与education类似
df_train_set.drop(['fnlwgt', 'educationNum'], axis=1, inplace=True)

df_train_set.drop_duplicates(inplace=True)  # 去除重复行

df_train_set[df_train_set.isna().values == True]  # 输出有缺失值的数据行

df_train_set.dropna(inplace=True)  # 去除空行

df_train_set[df_train_set['workclass'].str.contains(
    r'\?', regex=True)]  # 查找异常值, 避免与正则表达式的?冲突需要转义

df_train_set = df_train_set[~df_train_set['workclass'].str.contains(
    r'\?', regex=True)]

# 删除有异常值的行
new_columns = ['workclass', 'education', 'maritalStatus', 'occupation', 'relationship', 'race', 'sex',
               'nativeCountry', 'income']
for col in new_columns:
    df_train_set = df_train_set[~df_train_set[col].str.contains(
        r'\?', regex=True)]

# 处理连续型变量
continuous_column = ['age', 'capitalGain', 'capitalLoss', 'hoursPerWeek']
bins = [0, 25, 50, 75, 100]  # 分箱区间左开右闭 (0, 25], (25, 50], ...
bins1 = [-1, 1, 1000000]
df_train_set['age'] = pd.cut(df_train_set['age'], bins, labels=False)
df_train_set['capitalGain'] = pd.cut(
    df_train_set['capitalGain'], bins1, labels=False)
df_train_set['capitalLoss'] = pd.cut(
    df_train_set['capitalLoss'], bins1, labels=False)
df_train_set['hoursPerWeek'] = pd.cut(
    df_train_set['hoursPerWeek'], bins, labels=False)

# 处理离散型变量
discrete_column = ['workclass', 'education', 'maritalStatus',
                   'occupation', 'relationship', 'race', 'sex', 'nativeCountry', 'income']
# df_train_set['workclass'].value_counts().keys()
# df_train_set['education'].value_counts().keys()
# df_train_set['maritalStatus'].value_counts().keys()
# df_train_set['occupation'].value_counts().keys()
# df_train_set['relationship'].value_counts().keys()
# df_train_set['race'].value_counts().keys()
# df_train_set['sex'].value_counts().keys()
# df_train_set['nativeCountry'].value_counts().keys()
# df_train_set['income'].value_counts().keys()
workclass_mapping = {' Private': 0, ' Self-emp-not-inc': 1, ' Self-emp-inc': 1, ' Local-gov': 2,
                     ' State-gov': 2, ' Federal-gov': 2, ' Without-pay': 3, ' Never-worked': 3}
df_train_set['workclass'] = df_train_set['workclass'].map(workclass_mapping)

education_mapping = {' HS-grad': 0, ' Some-college': 0, ' Bachelors': 0, ' Masters': 1, ' Assoc-voc': 2,
                     ' Assoc-acdm': 2, ' 11th': 3, ' 10th': 3, ' 7th-8th': 3, ' Prof-school': 2, ' 9th': 3,
                     ' Doctorate': 2, ' 12th': 3, ' 5th-6th': 3, ' 1st-4th': 3, ' Preschool': 3}
df_train_set['education'] = df_train_set['education'].map(education_mapping)

maritalStatus_mapping = {' Married-civ-spouse': 0, ' Never-married': 1, ' Divorced': 1, ' Separated': 2,
                         ' Widowed': 2, ' Married-spouse-absent': 2, ' Married-AF-spouse': 2}
df_train_set['maritalStatus'] = df_train_set['maritalStatus'].map(
    maritalStatus_mapping)

occupation_mapping = {' Prof-specialty': 0, ' Exec-managerial': 0, ' Adm-clerical': 2, ' Craft-repair': 1,
                      ' Sales': 1, ' Other-service': 2, ' Machine-op-inspct': 2, ' Transport-moving': 2,
                      ' Handlers-cleaners': 2, ' Farming-fishing': 2, ' Tech-support': 2,
                      ' Protective-serv': 2, ' Priv-house-serv': 3, ' Armed-Forces': 3}
df_train_set['occupation'] = df_train_set['occupation'].map(occupation_mapping)

relationship_mapping = {' Husband': 0, ' Not-in-family': 1, ' Own-child': 2, ' Unmarried': 1, ' Wife': 1,
                        ' Other-relative': 2}
df_train_set['relationship'] = df_train_set['relationship'].map(
    relationship_mapping)

race_mapping = {' White': 0, ' Black': 1, ' Asian-Pac-Islander': 1, ' Amer-Indian-Eskimo': 2,
                ' Other': 2}
df_train_set['race'] = df_train_set['race'].map(race_mapping)

sex_mapping = {' Male': 0, ' Female': 1}
df_train_set['sex'] = df_train_set['sex'].map(sex_mapping)

nativeCountry_mapping = {' United-States': 0, ' Mexico': 1, ' Philippines': 1, ' Germany': 1, ' Puerto-Rico': 1,
                         ' Canada': 1, ' India': 1, ' El-Salvador': 1, ' Cuba': 1, ' England': 1, ' Jamaica': 1,
                         ' South': 1, ' China': 1, ' Italy': 1, ' Dominican-Republic': 1, ' Vietnam': 1,
                         ' Guatemala': 1, ' Japan': 1, ' Poland': 1, ' Columbia': 1, ' Iran': 1, ' Taiwan': 1,
                         ' Haiti': 1, ' Portugal': 1, ' Nicaragua': 1, ' Peru': 1, ' Greece': 1, ' France': 1,
                         ' Ecuador': 1, ' Ireland': 1, ' Hong': 1, ' Cambodia': 1, ' Trinadad&Tobago': 1,
                         ' Thailand': 1, ' Laos': 1, ' Yugoslavia': 1, ' Outlying-US(Guam-USVI-etc)': 1,
                         ' Hungary': 1, ' Honduras': 1, ' Scotland': 1, ' Holand-Netherlands': 1}
df_train_set['nativeCountry'] = df_train_set['nativeCountry'].map(
    nativeCountry_mapping)

income_mapping = {' <=50K': 0, ' >50K': 1}
df_train_set['income'] = df_train_set['income'].map(income_mapping)

df_train_set.to_csv('./after_train_adult.csv', index=False)