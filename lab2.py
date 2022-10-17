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

class TreeNode():#二叉树节点
    def __init__(self,val,lchild=None,rchild=None):
        self.label = (0, 0)
        self.val = val		
        self.lchild = lchild		
        self.rchild = rchild	


from json.encoder import INFINITY


def calc_gini(df):
    """
    计算数据集的基尼指数
    :param df: 数据集
    :return: 基尼指数
    """
    count = 0
    for i in df['income']:
        if i == 0:
            count += 1
    p = count / df.index.stop
    return 1 - p ** 2 - (1-p) ** 2
    
    

def split_dataset(df, index, value):
    """
    按照给定的列划分数据集
    :param df: 原始数据集
    :param index: 指定特征的列索引
    :param value: 指定特征的值
    :return: 切分后的数据集
    """
    column_name = list(df)
    groups = df.groupby(column_name[index])
    return groups.get_group(value)

def get_complete_dataset(df, df2):
    """
    取补集
    """
    df1 = df.append(df2)
    return df1.drop_duplicates(keep=False)
    
def get_feature_num(column_proc, index):
    """
    :param column_proc: 处理后的标签组
    :param index: 指定特征的列索引数
    :return: 指定特征的取值个数
    """
    return len(column_proc[index][1])

def choose_best_feature_to_split(df):
    """
    选择最好的特征进行分裂
    :param df: 数据集
    :return: best_value:(分裂特征的index, 特征的值), best_df:(分裂后的左右子树数据集), best_gini:(选择该属性分裂的最小基尼指数)
    """
    best_gini = INFINITY
    column_name = list(df)
    res = 0
    for i in range(df.shape[1]):
        index = column_name[i]
        for j in range(get_feature_num(df, index)):
            df_after = split_dataset(df, index, j)
            df_after_comp = get_complete_dataset(df, df_after)
            res = df_after.shape[0] / df.shape[0] * calc_gini(df_after) + df_after_comp.shape[0] / df.shape[0] * calc_gini(df_after_comp)
            if res < best_gini:
                best_value, best_df_l, best_df_r, best_gini = (index, j), df_after, df_after_comp, res
    return best_value, best_df_l, best_df_r, best_gini
        
def is_same_gini(df, columns):
    """
    判断数据集以某特征分裂的基尼指数是否相同
    :param columns: 数据集
    :return: True or False
    """
    list = set()
    for i in range(len(columns)):
        for j in range(get_feature_num(columns, i)):
            df_after = split_dataset(df, i, j)
            df_after_comp = get_complete_dataset(df, df_after)
            res = df_after.shape[0] / df.shape[0] * calc_gini(df_after) + df_after_comp.shape[0] / df.shape[0] * calc_gini(df_after_comp)
            list.add(res)
    if len(list) == 1:
        return True
    else:
        return False

def getMostCa(df):
    """
    返回数据集中出现次数最多的类别
    :param df: 数据集
    :return: 类别
    """
    
    column_name = list(df)
    ca = (column_name[0], 0)
    for i in range(df.shape[1]):
        for key, value in df[column_name[i]].value_counts().items():
            if value > ca[1]:
                ca = (column_name[i], key)
    return ca

def build_decision_tree(df, columns, flags):
    """
    构建CART树 
    :param df: 数据集
    :param columns: 特征列表
    :param flags: 区分特征是否被完全区分开,初始为全0, 若某个特征被区分开那么flags对应的下标为0
    :return: CART树
    """
    for i in range(len(flags)):
        node = TreeNode(df, None, None)
        index = get_feature_num(columns, i)
        if index == 1:
            flags[i] = 0
            node.label = columns[i]
            return node
        else:
            flags[i] = 1
    if len(columns) == 0 or is_same_gini(df, columns):
        node.label = getMostCa(df)
        return node
    best_value, best_df_l, best_df_r, best_gini = choose_best_feature_to_split(df)

    feature, value = best_value[0], best_value[1]
    node.label = (feature, value)
    columns0 = columns.copy()
    for (i, j) in enumerate(columns0):
        if j[0] == feature:
            index = i
            columns0[index][1] = list(filter(lambda x: x != value, columns0[index][1]))
            if columns0[index][1] == []:
                del columns0[index]
    lchild = TreeNode(best_df_l, None, None)
    node.left = lchild
    if best_df_l.shape[0] == 0:
        lchild.label = getMostCa(df)
    else:
        lchild = build_decision_tree(best_df_l, columns0, flags)
    
    rchild = TreeNode(best_df_r, None, None)
    node.right = rchild
    if best_df_r.shape[0] == 0:
        rchild.label = getMostCa(df)
    else:
        rchild = build_decision_tree(best_df_r, columns0, flags)

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

df_train = df_train_set.copy() #防止预处理重新来

columns = df_train.columns.to_list()
flags = [0 for i in range(len(columns))]

# df_train.head()
# print(flags)
# [(a1, [v1]),(a1, [v2, v3, ...]),...]
columns_proc = [[x, l] for x in columns for l in df_train[x].unique()]

cart = build_decision_tree(df_train, columns_proc, flags)
save_decision_tree(cart)

def classify(cart, df_row, columns):
    """
    用训练好的决策树进行分类
    :param cart:决策树模型
    :param df_row: 一条测试样本
    :param columns: 特征列表
    :return: 预测结果
    """
    feature = cart.label[0]
    value = cart.label[1]
    featIndex = columns.index(feature)
    key = df_row[featIndex]
    if value == key:
        if cart.left is None:
            return cart.label[1]
        else:
            return classify(cart.left, df_row, columns)
    else:
        if cart.right is None:
            return cart.label[1]
        else:
            return classify(cart.right, df_row, columns)


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
        pred_label = classify(cart, df.iloc[i,:], columns)
        if pred_label == -1:
            pred_label = random.randint(0, 1) # 防止classify执行到返回-1,但一般不会执行到返回-1
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


columns = df_train.columns.to_list()
cart = load_decision_tree() # 加载模型
test_list = df_train['income'].to_numpy()
pred_list = predict(cart, df_train, columns)
acc = calc_acc(pred_list, test_list)