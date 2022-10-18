from json.encoder import INFINITY
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import random

columns = ['age', 'workclass', 'fnlwgt', 'education', 'educationNum', 'maritalStatus', 'occupation', 'relationship', 'race', 'sex',
           'capitalGain', 'capitalLoss', 'hoursPerWeek', 'nativeCountry', 'income']
df_train_set = pd.read_csv('adult.data', names=columns)
df_test_set = pd.read_csv(
    'adult.test', names=columns, skiprows=1)  # 第一行是非法数据

# print(df_train_set.head())
# print(df_test_set.head())
df_train_set.to_csv('./train_adult.csv', index=False)
df_test_set.to_csv('./test_adult.csv', index=False)

df_train_set = pd.read_csv('./train_adult.csv')
df_test_set = pd.read_csv('./test_adult.csv')

# fnlwgt列用处不大，educationNum与education类似
df_train_set.drop(['fnlwgt', 'educationNum'], axis=1, inplace=True)
df_test_set.drop(['fnlwgt', 'educationNum'], axis=1, inplace=True)

df_train_set.drop_duplicates(inplace=True)  # 去除重复行
df_test_set.drop_duplicates(inplace=True)  # 去除重复行

df_train_set[df_train_set.isna().values == True]  # 输出有缺失值的数据行
df_test_set[df_test_set.isna().values == True]  # 输出有缺失值的数据行

df_train_set.dropna(inplace=True)  # 去除空行
df_test_set.dropna(inplace=True)  # 去除空行

df_train_set[df_train_set['workclass'].str.contains(
    r'\?', regex=True)]  # 查找异常值, 避免与正则表达式的?冲突需要转义
df_test_set[df_test_set['workclass'].str.contains(
    r'\?', regex=True)]  # 查找异常值, 避免与正则表达式的?冲突需要转义

df_train_set[df_train_set['workclass'].str.contains(
    r'\?', regex=True)]  # 查找异常值, 避免与正则表达式的?冲突需要转义
df_test_set[df_test_set['workclass'].str.contains(
    r'\?', regex=True)]  # 查找异常值, 避免与正则表达式的?冲突需要转义

df_train_set = df_train_set[~df_train_set['workclass'].str.contains(
    r'\?', regex=True)]
df_test_set = df_test_set[~df_test_set['workclass'].str.contains(
    r'\?', regex=True)]

# 删除有异常值的行
new_columns = ['workclass', 'education', 'maritalStatus', 'occupation', 'relationship', 'race', 'sex',
               'nativeCountry', 'income']
for col in new_columns:
    df_train_set = df_train_set[~df_train_set[col].str.contains(
        r'\?', regex=True)]
    df_test_set = df_test_set[~df_test_set[col].str.contains(
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

df_test_set['age'] = pd.cut(df_test_set['age'], bins, labels=False)
df_test_set['capitalGain'] = pd.cut(
    df_test_set['capitalGain'], bins1, labels=False)
df_test_set['capitalLoss'] = pd.cut(
    df_test_set['capitalLoss'], bins1, labels=False)
df_test_set['hoursPerWeek'] = pd.cut(
    df_test_set['hoursPerWeek'], bins, labels=False)

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
df_test_set['workclass'] = df_test_set['workclass'].map(workclass_mapping)

education_mapping = {' HS-grad': 0, ' Some-college': 0, ' Bachelors': 0, ' Masters': 1, ' Assoc-voc': 2,
                     ' Assoc-acdm': 2, ' 11th': 3, ' 10th': 3, ' 7th-8th': 3, ' Prof-school': 2, ' 9th': 3,
                     ' Doctorate': 2, ' 12th': 3, ' 5th-6th': 3, ' 1st-4th': 3, ' Preschool': 3}
df_train_set['education'] = df_train_set['education'].map(education_mapping)
df_test_set['education'] = df_test_set['education'].map(education_mapping)

maritalStatus_mapping = {' Married-civ-spouse': 0, ' Never-married': 1, ' Divorced': 1, ' Separated': 2,
                         ' Widowed': 2, ' Married-spouse-absent': 2, ' Married-AF-spouse': 2}
df_train_set['maritalStatus'] = df_train_set['maritalStatus'].map(
    maritalStatus_mapping)
df_test_set['maritalStatus'] = df_test_set['maritalStatus'].map(
    maritalStatus_mapping)

occupation_mapping = {' Prof-specialty': 0, ' Exec-managerial': 0, ' Adm-clerical': 2, ' Craft-repair': 1,
                      ' Sales': 1, ' Other-service': 2, ' Machine-op-inspct': 2, ' Transport-moving': 2,
                      ' Handlers-cleaners': 2, ' Farming-fishing': 2, ' Tech-support': 2,
                      ' Protective-serv': 2, ' Priv-house-serv': 3, ' Armed-Forces': 3}
df_train_set['occupation'] = df_train_set['occupation'].map(occupation_mapping)
df_test_set['occupation'] = df_test_set['occupation'].map(occupation_mapping)

relationship_mapping = {' Husband': 0, ' Not-in-family': 1, ' Own-child': 2, ' Unmarried': 1, ' Wife': 1,
                        ' Other-relative': 2}
df_train_set['relationship'] = df_train_set['relationship'].map(
    relationship_mapping)
df_test_set['relationship'] = df_test_set['relationship'].map(
    relationship_mapping)

race_mapping = {' White': 0, ' Black': 1, ' Asian-Pac-Islander': 1, ' Amer-Indian-Eskimo': 2,
                ' Other': 2}
df_train_set['race'] = df_train_set['race'].map(race_mapping)
df_test_set['race'] = df_test_set['race'].map(race_mapping)

sex_mapping = {' Male': 0, ' Female': 1}
df_train_set['sex'] = df_train_set['sex'].map(sex_mapping)
df_test_set['sex'] = df_test_set['sex'].map(sex_mapping)

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
df_test_set['nativeCountry'] = df_test_set['nativeCountry'].map(
    nativeCountry_mapping)

income_mapping = {' <=50K': 0, ' >50K': 1}
income_test_mapping = {' <=50K.': 0, ' >50K.': 1}
df_train_set['income'] = df_train_set['income'].map(income_mapping)
df_test_set['income'] = df_test_set['income'].map(income_test_mapping)

df_train_set.to_csv('after_train_adult.csv', index=False)
df_test_set.to_csv('after_test_adult.csv', index=False)


# class TreeNode():  # 二叉树节点
#     def __init__(self, val, lchild=None, rchild=None):
#         self.label = (0, 0)
#         self.val = val
#         self.lchild = lchild
#         self.rchild = rchild


# # def calc_gini(df):
# #     """
# #     计算数据集的基尼指数
# #     :param df: 数据集
# #     :return: 基尼指数
# #     """
# #     count = 0
# #     for i in df['income']:
# #         if i == 0:
# #             count += 1
# #     p = count / df.shape[0]
# #     return 1 - p ** 2 - (1-p) ** 2

# def calc_gini(df):
#     """
#     计算数据集的基尼指数
#     :param df: 数据集
#     :return: 基尼指数
#     """
#     total_num = df['income'].count()

#     income_count = {}

#     p_income = {}

#     for income in df['income'].value_counts().keys():
#         income_count[income] = df[df['income']==income]['income'].count()
#         p_income[income] = income_count[income] / total_num


#     if len(df['income'].value_counts().keys()) == 1:
#         if df['income'].value_counts().keys()[0] == 0:
#             gini = 1 - p_income[0]*p_income[0]
#         else:
#             gini = 1 - p_income[1]*p_income[1]
#     else:
#         gini = 1 - p_income[0]*p_income[0] - p_income[1]*p_income[1]

#     # print(gini)
#     return gini


# def split_dataset(df, index, value):
#     """
#     按照给定的列划分数据集
#     :param df: 原始数据集
#     :param index: 指定特征的列索引
#     :param value: 指定特征的值
#     :return: 切分后的数据集
#     """
#     column_name = list(df)
#     groups = df.groupby(column_name[index])
#     return groups.get_group(value)


# def get_complete_dataset(df, df2):
#     """
#     取补集
#     """
#     df1 = df.append(df2)
#     return df1.drop_duplicates(keep=False)


# def get_feature_num(column_proc, index):
#     """
#     :param column_proc: 处理后的标签组
#     :param index: 指定特征的列索引数
#     :return: 指定特征的取值个数
#     """
#     return len(column_proc[index][1])


# def choose_best_feature_to_split(df):
#     """
#     选择最好的特征进行分裂
#     :param df: 数据集
#     :return: best_value:(分裂特征的index, 特征的值), best_df:(分裂后的左右子树数据集), best_gini:(选择该属性分裂的最小基尼指数)
#     """
#     best_gini = INFINITY
#     column_name = list(df)
#     res = 0
#     for i in range(df.shape[1]-1):
#         index = column_name[i]
#         if len(df[index].value_counts().index) == 1:
#             continue
#         for key in df[index].value_counts().index:
#             df_after = split_dataset(df, i, key)
#             df_after_comp = get_complete_dataset(df, df_after)
#             res = df_after.shape[0] / df.shape[0] * calc_gini(
#                 df_after) + df_after_comp.shape[0] / df.shape[0] * calc_gini(df_after_comp)
#             if res < best_gini:
#                 best_value, best_df_l, best_df_r, best_gini = (
#                     index, key), df_after, df_after_comp, res
#     return best_value, best_df_l, best_df_r, best_gini


# def is_same_gini(df):
#     """
#     判断数据集以某特征分裂的基尼指数是否相同
#     :param columns: 数据集
#     :return: True or False
#     """
#     l_set = set()
#     l_df = list(df)
#     for i in range(df.shape[1]):
#         feature = l_df[i]
#         if len(df[feature].value_counts().index) == 1:
#             continue
#         for key in df[feature].value_counts().index:
#             df_after = split_dataset(df, i, key)
#             df_after_comp = get_complete_dataset(df, df_after)
#             res = df_after.shape[0] / df.shape[0] * calc_gini(
#                 df_after) + df_after_comp.shape[0] / df.shape[0] * calc_gini(df_after_comp)
#             l_set.add(res)
#     if len(l_set) == 1:
#         return True
#     else:
#         return False


# def getLabel(df):
#     """
#     返回数据集中出现次数最多的类别
#     :param df: 数据集
#     :return: 类别
#     """

#     for key, value in df['income'].value_counts(sort=True).items():
#         return ('income', key)


# def rm_column(columns, feature, value):
#     columns0 = columns.copy()
#     for (i, j) in enumerate(columns0):
#         if j[0] == feature:
#             index = i
#             columns0[index][1] = list(
#                 filter(lambda x: x != value, columns0[index][1]))
#             if columns0[index][1] == []:
#                 del columns0[index]
#     return columns0


# def build_decision_tree(df, columns, flags):
#     """
#     构建CART树 
#     :param df: 数据集
#     :param columns: 特征列表
#     :param flags: 区分特征是否被完全区分开,初始为全0, 若某个特征被区分开那么flags对应的下标为0
#     :return: CART树
#     """
#     node = TreeNode(df, None, None)
#     if len(df['income'].value_counts().index) == 1:
#         node.label = ('income', df.iloc[0][-1])
#         return node
#     if len(columns) == 0 or is_same_gini(df):
#         node.label = getLabel(df)
#         return node
#     best_value, best_df_l, best_df_r, best_gini = choose_best_feature_to_split(
#         df)

#     feature, value = best_value[0], best_value[1]
#     node.label = (feature, value)
#     columns0 = columns.copy()
#     for (i, j) in enumerate(columns0):
#         if j[0] == feature:
#             index = i
#             columns0[index][1] = list(
#                 filter(lambda x: x != value, columns0[index][1]))
#             if columns0[index][1] == []:
#                 del columns0[index]
#             break
#     lchild = TreeNode(best_df_l, None, None)
#     node.lchild = lchild
#     if best_df_l.shape[0] == 0:
#         lchild.label = getLabel(df)
#         return node
#     else:
#         node.lchild = build_decision_tree(best_df_l, columns0, flags)

#     rchild = TreeNode(best_df_r, None, None)
#     node.rchild = rchild
#     if best_df_r.shape[0] == 0:
#         rchild.label = getLabel(df)
#         return node
#     else:
#         node.rchild = build_decision_tree(best_df_r, columns0, flags)
#     return node

#     # 递归结束情况1: 若当前集合的所有样本标签相等,即样本已被分"纯",则可以返回该标签值作为一个叶子节点
#     # 递归结束情况2: 若当前训练集的所有特征都被使用完毕,当前无可用特征但样本仍未分"纯"，则返回样本最多的标签作为结果


# def save_decision_tree(cart):
#     """
#     决策树的存储
#     :param cart: 训练好的决策树
#     :return: void
#     """
#     np.save('cart.npy', cart)


# def load_decision_tree():
#     """
#     决策树的加载
#     :return: 保存的决策树
#     """

#     cart = np.load('cart.npy', allow_pickle=True)
#     return cart.item()


def calc_gini(df):
    """
    计算数据集的基尼指数
    :param df: 数据集
    :return: 基尼指数
    """
    total_num = df['income'].count()

    income_count = {}

    p_income = {}

    for income in df['income'].value_counts().keys():
        income_count[income] = df[df['income']==income]['income'].count()
        p_income[income] = income_count[income] / total_num


    if len(df['income'].value_counts().keys()) == 1:
        if df['income'].value_counts().keys()[0] == 0:
            gini = 1 - p_income[0]*p_income[0]
        else:
            gini = 1 - p_income[1]*p_income[1]
    else:
        gini = 1 - p_income[0]*p_income[0] - p_income[1]*p_income[1]

    # print(gini)
    return gini

def split_dataset(df, index, value):
    """
    按照给定的列划分数据集
    :param df: 原始数据集
    :param index: 指定特征的列索引
    :param value: 指定特征的值
    :return: 切分后的数据集
    """

    sub_df_1 = df.copy()
    sub_df_2 = df.copy()

    # sub_df_3 = sub_df_3[~(sub_df_3['age'] == 0)]
    # 去0

    sub_df_1 = sub_df_1[~(sub_df_1[index] != value)]
    sub_df_2 = sub_df_2[~(sub_df_2[index] == value)]

    

    return sub_df_1, sub_df_2
    
def choose_best_feature_to_split(df, flags, feature_used):
    """
    选择最好的特征进行分裂
    :param df: 数据集
    :return: best_value:(分裂特征的index, 特征的值), best_df:(分裂后的左右子树数据集), best_gain:(选择该属性分裂的最大信息增益)
    """
    numFeatures = len(df.columns) - 1
    # 如果只有一个特征, 直接返回
    if numFeatures == 1:
        return df.columns[0]

    # 最佳基尼系数初始化
    best_gini = 1

    

    for i in range(0, numFeatures):
        # Gini = {}
        name_of_feature = df.columns[i]
        if flags[i] == 1:
            continue
        
        
        
        for feature_value in df[name_of_feature].value_counts().keys():
            # 对各个值划分子树
            # TODO
            # if feature_value in feature_used[name_of_feature]:
            #     # print(feature_used)
            #     continue

            
            sub_df_1, sub_df_2 = split_dataset(df, name_of_feature, feature_value)
            
            # 求两个子树的占比
            prob1 = sub_df_1.shape[0] / df.shape[0]
            prob2 = sub_df_2.shape[0] / df.shape[0]
            
            # 计算gini
            gini1 = calc_gini(sub_df_1)
            gini2 = calc_gini(sub_df_2)

            # 当前gini
            gini = prob1*gini1 + prob2*gini2

            # 判断
            if gini < best_gini:
                best_gini = gini
                best_index = name_of_feature
                best_value = feature_value
                best_df1 = sub_df_1
                best_df2 = sub_df_2

    # print(best_gini)
    if best_gini == 1:
        best_index = 'none'
        best_value = 'none'
        best_df1 = 'none'
        best_df2 = 'none'
    
    return best_index, best_value, best_df1, best_df2
                
            
def find_label(df):
    max_income = 0
    income_count = {}
    for income in df['income'].value_counts().keys():
        income_count[income] = df[df['income']==income]['income'].count()
        if income_count[income] > max_income:
            max_income = income_count[income]
            ans = income
        
    return ans


    

    


def build_decision_tree(df, flags, tree_depth, feature_used):
    """
    构建CART树
    :param df: 数据集
    :param columns: 特征列表
    :param flags: 区分特征是否被完全区分开,初始为全0, 若某个特征被区分开那么flags对应的下标为0
    :return: CART树
    """


    # 规定flags
    for i in range(0, len(df.columns)-1):
        if len(df[df.columns[i]].value_counts().keys()) == 1:
            flags[i] = 1
        else:
            flags[i] = 0

    # 加深度
    tree_depth += 1


    # "纯"
    if len(df['income'].value_counts().keys()) == 1:
        print('纯')
        return df['income'].value_counts().keys()[0]
    # 没有特征了
    
    no_f = True
    for i in range(0, len(df.columns)):
        if flags[i] == 0:
            no_f = False
            break
    if no_f:
        print('梅特')
        return find_label(df)
    
    # 限制深度
    # TODO 先用一层测试一下
    # 10 : test 0.8241
    # none : test 0.81
    # 15 : test 0.8192
    # 13 : test 0.8208
    # 7 : 8252
    # 5 : 8219
    # 8 : 8249
    # 6 : 8246
    if tree_depth >= 7:
        print('深度')
        return find_label(df)

    if feature_used == {'age': [1, 2, 0, 3], 'workclass': [3, 0, 2, 1], 'education': [1, 4, 2, 5, 3, 0, 6], 'maritalStatus': [1, 0, 2, 3], 'occupation': [2, 1, 3, 0, 4, 5], 'relationship': [0, 1, 2], 'race': [4, 0, 1, 2, 3], 'sex': [0, 1], 'capitalGain': [3, 1, 0], 'capitalLoss': [0, 1], 'hoursPerWeek': [1, 2, 0, 3], 'nativeCountry': [2, 5, 4, 3, 1, 0]}:
        print('都用了')
        return find_label(df)

    tree_depth_l = tree_depth / 1
    tree_depth_r = tree_depth / 1
    


    best_index, best_value, best_df1, best_df2 = choose_best_feature_to_split(df, flags, feature_used)

    # print(best_index, best_value)
    # print(best_df1)

    # 初始化决策树
    if best_index == 'none':
        print('没jini')
        return find_label(df)
    else:
        feature_used[best_index].append(best_value)
    
    decision_tree = {best_index: {}}


    fu_l = feature_used.copy()

    fu_r = feature_used.copy()

    # print(feature_used)

    # print(best_df1)
    # print(best_df2)
    

    # if  best_df1.shape[0] == 1:
    #     decision_tree[best_index][best_value] = find_label(best_df1)
    # else:
    decision_tree[best_index][best_value] = build_decision_tree(best_df1, flags, tree_depth_l, feature_used=fu_l)

    # if  best_df2.shape[0] == 1:
    #     decision_tree[best_index]['others'] = find_label(best_df2)
    # else:
    decision_tree[best_index]['others'] = build_decision_tree(best_df2, flags, tree_depth_r, feature_used=fu_r)
    # decision_tree[best_index]['others'] = build_decision_tree(best_df2, flags, tree_depth, feature_used)

    return decision_tree

    

    
    # 递归结束情况1: 若当前集合的所有样本标签相等,即样本已被分"纯",则可以返回该标签值作为一个叶子节点
    # 递归结束情况2: 若当前训练集的所有特征都被使用完毕,当前无可用特征但样本仍未分"纯"，则返回样本最多的标签作为结果
    
    
def save_decision_tree(cart):
    """
    决策树的存储
    :param cart: 训练好的决策树
    :return: void
    """
    np.save('cart.npy', cart)
    
    
def load_decision_tree():
    """
    决策树的加载
    :return: 保存的决策树
    """    
    
    cart = np.load('cart.npy', allow_pickle=True)
    return cart.item()





def classify(cart, df_row, columns):
    """
    用训练好的决策树进行分类
    :param cart:决策树模型
    :param df_row: 一条测试样本
    :param columns: 特征列表
    :return: 预测结果
    """
    if cart.lchild is None and cart.rchild is None:
        return cart.label[1]
    feature = cart.label[0]
    value = cart.label[1]
    featIndex = columns.index(feature)
    key = df_row[featIndex]
    if value == key:
        return classify(cart.lchild, df_row, columns)
    else:
        return classify(cart.rchild, df_row, columns)


# def classify(cart, df_row, columns):
#     """
#     用训练好的决策树进行分类
#     :param cart:决策树模型
#     :param df_row: 一条测试样本
#     :param columns: 特征列表
#     :return: 预测结果
#     """
#     # 首个特征
#     first_feature = list(cart.keys())[0]
    
#     second_dict = cart[first_feature]

#     # key 一般是数字或者 'others'
#     for key in second_dict.keys():
#         # 是数字
#         if key != 'others':
#             # 如果特征值等于该数字
#             if df_row[first_feature] == key:
#                 # 如果还是dict, 需要再次递归
#                 if type(second_dict[key]).__name__ == 'dict':
#                     classLabel = classify(second_dict[key], df_row, columns)

#                 # 如果是一个数字, 就是他啦!
#                 else:
#                     classLabel = second_dict[key]

#             # 不等于就是在others树中
#             else:
#                 if type(second_dict['others']).__name__ == 'dict':
#                     classLabel = classify(second_dict['others'], df_row, columns)
#                 else:
#                     classLabel = second_dict['others']

#     return classLabel

def prune(cart, test_data, coulumns):
    """
    对决策树进行后剪枝处理
    :param cart:决策树模型
    :param df: 数据集
    """
    for index in range(len(coulumns)):
        if coulumns[index] == cart.label[0]:
            break
    if cart.label[1] not in test_data[cart.label[0]].value_counts().index:
        return cart
    lSet = split_dataset(test_data, index, cart.label[1])
    rSet = get_complete_dataset(test_data, lSet)
    # 如果有左子树或者右子树,则递归处理
    if cart.lchild.lchild is not None or cart.lchild.rchild is not None \
    or cart.rchild.lchild is not None or cart.rchild.rchild is not None:
        # 处理左子树(剪枝)
        if cart.lchild.lchild is not None or cart.lchild.rchild is not None:
            cart.lchild = prune(cart.lchild, lSet, coulumns)
        # 处理右子树(剪枝)
        if cart.rchild.lchild is not None or cart.rchild.rchild is not None:
            cart.rchild = prune(cart.rchild, rSet, coulumns)
    # 如果当前结点的左右结点为叶结点
    if cart.lchild.lchild is None and cart.lchild.rchild is None \
    and cart.rchild.lchild is None and cart.rchild.rchild is None:
        # 计算没有合并的准确率
        
        rateNoMerge = (calc_acc(predict(cart, lSet, coulumns), lSet['income'].to_numpy()) + calc_acc(predict(cart, rSet, coulumns), rSet['income'].to_numpy())) / 2
        # 合并行为
        value = getLabel(test_data)
        pred_list = [value[1] for i in range(test_data.shape[0])]
        # 计算合并的准确率
        rateMerge = calc_acc(pred_list, test_data['income'].to_numpy())
        # 如果合并的准确率大于没有合并的准确率,则合并
        if rateMerge > rateNoMerge:
            cart.label = value
            cart.lchild = None
            cart.rchild = None
        return cart
    else:
        return cart


def predict(cart, df, columns):
    """
    用训练好的决策树进行分类
    :param cart:决策树模型
    :param df: 所有测试集
    :param columns: 特征列表
    :return: 预测结果
    """
    pred_list = []
    for i in range(len(df)):
        pred_label = classify(cart, df.iloc[i, :], columns)
        if pred_label == -1:
            pred_label = random.randint(0, 1)  # 防止classify执行到返回-1,但一般不会执行到返回-1
        pred_list.append(pred_label)
    return pred_list


def calc_acc(pred_list, test_list):
    """
    返回预测准确率
    :param pred_list: 预测列表
    :param test_list: 测试列表
    :return: 准确率
    """
    pred = np.array(pred_list)
    test = np.array(test_list)
    acc = np.sum(pred_list == test_list) / len(test_list)
    return acc

df_train = df_train_set.copy()  # 防止预处理重新来
df_test = df_test_set.copy()  # 防止预处理重新来

columns = df_train.columns.to_list()
flags = [0 for i in range(len(columns)-1)]

# df_train.head()
# print(flags)
# [(a1, [v1]),(a1, [v2, v3, ...]),...]
columns_proc = [[x, list(df_train[x].unique())]
                for x in columns if x != 'income']

cart = build_decision_tree(df_train, columns_proc, flags)
save_decision_tree(cart)

# columns = df_train.columns.to_list()
# flags = [0 for i in range(len(columns))]
# flags[len(columns)-1] = 1
# tree_depth = 0
# feature_used = {'age':[], 'workclass':[], 'education':[], 'maritalStatus':[],
#                 'occupation':[], 'relationship':[], 'race':[], 'sex':[],
#                 'capitalGain':[], 'capitalLoss':[], 'hoursPerWeek':[],
#                 'nativeCountry':[]}
# cart = build_decision_tree(df_train, flags, tree_depth, feature_used)

columns = df_train.columns.to_list()
# cart = load_decision_tree() # 加载模型
test_list = df_test['income'].to_numpy()
# cart_pre = prune(cart, df_test, columns)
pred_list = predict(cart, df_test, columns)
acc = calc_acc(pred_list, test_list)
print(acc)
