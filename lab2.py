import pandas as pd
import numpy as np
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

df_test_out = df_test_set.copy()

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
workclass_mapping = {' Private': 0, ' Self-emp-not-inc': 1, ' Self-emp-inc': 2, ' Local-gov': 3,
                     ' State-gov': 4, ' Federal-gov': 5, ' Without-pay': 6, ' Never-worked': 7}
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


class TreeNode():  # 二叉树节点
    def __init__(self, val, lchild=None, rchild=None):
        self.label = (0, 0)
        self.val = val
        self.lchild = lchild
        self.rchild = rchild


def calc_gini(df):
    """
    计算数据集的基尼指数
    :param df: 数据集
    :return: 基尼指数
    """
    try:

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
    except(KeyError):
        print(df['income'].value_counts().keys())
        # return 1


def split_dataset(df, index, value):
    """
    按照给定的列划分数据集
    :param df: 原始数据集
    :param index: 指定特征的列索引
    :param value: 指定特征的值
    :return: 切分后的数据集
    """
    lchild = df.copy()
    rchild = df.copy()

    lchild = lchild[~(lchild[index] != value)]
    rchild = rchild[~(rchild[index] == value)]

    return lchild, rchild


# def get_complete_dataset(df, df2):
#     """
#     取补集
#     """
#     df1 = df.append(df2)

#     return df1.drop_duplicates(keep=False)


def choose_best_feature_to_split(df, flags):
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
        if flags[i] == 0:
            continue

        for feature_value in df[name_of_feature].value_counts().keys():
            # 对各个值划分子树
            lchild, rchild = split_dataset(df, name_of_feature, feature_value)

            if lchild.empty or rchild.empty:
                continue
            # 当前gini
            gini = lchild.shape[0] / df.shape[0] * calc_gini(lchild) + rchild.shape[0] / df.shape[0] * calc_gini(rchild)

            # 判断
            if gini < best_gini:
                best_gini, best_feature, best_df_l, best_df_r,  = gini, (name_of_feature, feature_value), lchild, rchild
    if best_gini == 1:
        best_feature = ('none', 'none')
        best_df_l = None
        best_df_r = None
    return best_feature, best_df_l, best_df_r



def getLabel(df):
    """
    返回数据集中出现次数最多的类别
    :param df: 数据集
    :return: 类别
    """

    for key, value in df['income'].value_counts(sort=True).items():
        return ('income', key)


def build_decision_tree(df, columns, flags):
    """
    构建CART树
    :param df: 数据集
    :param columns: 特征列表
    :param flags: 区分特征是否被完全区分开,初始为全0, 若某个特征被区分开那么flags对应的下标为0
    :return: CART树
    """
    node = TreeNode(df, None, None)
    for i in range(len(df.columns) - 1):
        name_of_feature = df.columns[i]
        if len(df[name_of_feature].value_counts().index) == 1:
            flags[i] = 0
        else: flags[i] = 1

        
    # 若数据集中的类别完全相同,则返回该类别
    if len(df['income'].value_counts().index) == 1:
        node.label = ('income', df['income'].value_counts().keys()[0])
        return node
    
    # 若数据集中的特征完全相同,则返回出现次数最多的类别
    if sum(flags) == 0:
        node.label = getLabel(df)
        return node

    # 获取最优划分
    best_value, best_df_l, best_df_r = choose_best_feature_to_split(df, flags)

    # 获取进行划分的特征
    feature, value = best_value[0], best_value[1]
    node.label = (feature, value)
    
    # 递归构建左右子树
    lchild = TreeNode(best_df_l, None, None)
    node.lchild = lchild
    if best_df_l is None:
        lchild.label = getLabel(df)
        return node
    else:
        node.lchild = build_decision_tree(best_df_l, columns, flags)

    rchild = TreeNode(best_df_r, None, None)
    node.rchild = rchild
    if best_df_r is None:
        rchild.label = getLabel(df)
        return node
    else:
        node.rchild = build_decision_tree(best_df_r, columns, flags)
    return node

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
    feature = cart.label[0]
    value = cart.label[1]
    featIndex = columns.index(feature)
    key = df_row[featIndex]
    if value == key:
        if cart.lchild is None:
            return value
        else:
            return classify(cart.lchild, df_row, columns)
    else:
        if cart.rchild is None:
            return value
        else:
            return classify(cart.rchild, df_row, columns)


def prune(cart, test_data, coulumns):
    """
    对决策树进行后剪枝处理
    :param cart:决策树模型
    :param df: 数据集
    """
    if cart.label[1] not in test_data[cart.label[0]].value_counts().index:
        return cart
    lSet, rSet = split_dataset(test_data, cart.label[0], cart.label[1])
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

        rateNoMerge = (calc_acc(predict(cart, lSet, coulumns), lSet['income'].to_numpy(
        )) + calc_acc(predict(cart, rSet, coulumns), rSet['income'].to_numpy())) / 2
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
flags = [1 for i in range(len(columns)-1)]

# df_train.head()
# print(flags)
# [(a1, [v1]),(a1, [v2, v3, ...]),...]
# columns_proc = [[x, list(df_train[x].unique())]
#                 for x in columns if x != 'income']

# cart = build_decision_tree(df_train, columns, flags)
# save_decision_tree(cart)

cart = load_decision_tree() # 加载模型
test_list = df_test['income'].to_numpy()
cart = prune(cart, df_test, columns)
pred_list = predict(cart, df_test, columns)
acc = calc_acc(pred_list, test_list)
print(acc)

df_test_out['income'] = pred_list
out_income_mapping = {0:' <=50K', 1:' >50K'}

df_test_out['income'] = df_test_out['income'].map(out_income_mapping)

df_test_out.to_csv('test_adult_out.csv', index=False)