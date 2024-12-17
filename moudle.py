# 导入必要的库

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder

from ucimlrepo import fetch_ucirepo

  

# 在导入库后添加

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑字体

plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

  

# 1. 数据加载

heart_disease = fetch_ucirepo(id=45)

data = pd.DataFrame(heart_disease.data.features)

data['HeartDisease'] = heart_disease.data.targets

  

# 2. 数据概览

print("数据集列名：", data.columns.tolist())

print("\n数据预览：")

print(data.head())

print("\n数据信息：")

print(data.info())

print("\n数据描述：")

print(data.describe())

  

# 3. 特征描述与统计分析

continuous_vars = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

mean_values = data[continuous_vars].mean()

variance_values = data[continuous_vars].var()

print("均值:\n", mean_values)

print("\n方差:\n", variance_values)

  

# 绘制箱线图

plt.figure(figsize=(15, 10))

for i, var in enumerate(continuous_vars):

    plt.subplot(2, 3, i+1)

    sns.boxplot(y=data[var])

    plt.title(f'Boxplot of {var}')

plt.tight_layout()

plt.show()

  

# 正态性检验

for var in continuous_vars:

    stat, p = stats.shapiro(data[var])

    print(f'{var}: Statistics={stat:.3f}, p={p:.3f}')

    if p > 0.05:

        print(f'{var} 数据可能服从正态分布。\n')

    else:

        print(f'{var} 数据不服从正态分布。\n')

  

# t检验

group0 = data[data['HeartDisease'] == 0]

group1 = data[data['HeartDisease'] == 1]

for var in continuous_vars:

    stat, p = stats.ttest_ind(group0[var], group1[var], equal_var=False)

    print(f'{var}: t-statistic={stat:.3f}, p-value={p:.3f}')

    if p < 0.05:

        print(f'{var} 在两组之间存在显著差异。\n')

    else:

        print(f'{var} 在两组之间不存在显著差异。\n')

  

# 4. 数据预处理

  

# 检查缺失值

print("缺失值统计：")

print(data.isnull().sum())

  

# 使用SimpleImputer处理缺失值

imputer = SimpleImputer(strategy='mean')

# 对连续变量使用均值填充

data[continuous_vars] = imputer.fit_transform(data[continuous_vars])

# 对ca和thal列单独处理

data['ca'] = SimpleImputer(strategy='mean').fit_transform(data[['ca']])

data['thal'] = SimpleImputer(strategy='most_frequent').fit_transform(data[['thal']])

  

# 再次检查确保没有缺失值

print("\n处理后的缺失值统计：")

print(data.isnull().sum())

  

# 异常值处理

for var in continuous_vars:

    Q1 = data[var].quantile(0.25)

    Q3 = data[var].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR

    upper = Q3 + 1.5 * IQR

    print(f'{var}: 下限={lower}, 上限={upper}')

    data = data[(data[var] >= lower) & (data[var] <= upper)]

  

# 数据标准化

scaler = StandardScaler()

data[continuous_vars] = scaler.fit_transform(data[continuous_vars])

  

# 编码分类变量

categorical_vars = ['cp', 'restecg', 'slope', 'thal']

data = pd.get_dummies(data, columns=categorical_vars, drop_first=True)

  

print("\n处理后的数据预览：")

print(data.head())

  

# 5. 降维技术

  

# 分离特征和目标变量

X = data.drop('HeartDisease', axis=1)

y = data['HeartDisease']

  

# PCA

pca = PCA(n_components=0.90)

X_pca = pca.fit_transform(X)

print(f'原始维度: {X.shape}')

print(f'降维后维度: {X_pca.shape}')

  

# 绘制累计方差贡献率

plt.figure(figsize=(8,6))

plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')

plt.xlabel('Number of Components')

plt.ylabel('Cumulative Explained Variance Ratio')

plt.title('PCA Cumulative Explained Variance Ratio')

plt.grid(True)

plt.show()

  

# 6. 模型构建

  

# 数据集划分

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)

print(f'训练集样本数: {X_train.shape[0]}')

print(f'测试集样本数: {X_test.shape[0]}')

  

# 随机森林模型

rf_model = RandomForestClassifier(

    n_estimators=200,

    max_depth=15,

    min_samples_split=4,

    min_samples_leaf=2,

    random_state=42,

    n_jobs=-1,

    class_weight='balanced'

)

rf_model.fit(X_train, y_train)

  

# 随机森林评估

y_pred_rf = rf_model.predict(X_test)

print("随机森林混淆矩阵:")

print(confusion_matrix(y_test, y_pred_rf))

print("\n随机森林分类报告:")

print(classification_report(y_test, y_pred_rf, zero_division=0))

print(f'随机森林准确率: {accuracy_score(y_test, y_pred_rf):.2f}')

print(f'随机森林精确率(macro): {precision_score(y_test, y_pred_rf, average="macro", zero_division=0):.2f}')

print(f'随机森林召回率(macro): {recall_score(y_test, y_pred_rf, average="macro", zero_division=0):.2f}')

print(f'随机森林F1分数(macro): {f1_score(y_test, y_pred_rf, average="macro", zero_division=0):.2f}')

  

# 支持向量机模型

svm_model = SVC(

    kernel='rbf',

    C=1.0,

    gamma='scale',

    random_state=42,

    class_weight='balanced'

)

svm_model.fit(X_train, y_train)

  

# SVM评估

y_pred_svm = svm_model.predict(X_test)

print("\nSVM混淆矩阵:")

print(confusion_matrix(y_test, y_pred_svm))

print("\nSVM分类报告:")

print(classification_report(y_test, y_pred_svm, zero_division=0))

print(f'SVM准确率: {accuracy_score(y_test, y_pred_svm):.2f}')

print(f'SVM精确率(macro): {precision_score(y_test, y_pred_svm, average="macro", zero_division=0):.2f}')

print(f'SVM召回率(macro): {recall_score(y_test, y_pred_svm, average="macro", zero_division=0):.2f}')

print(f'SVM F1分数(macro): {f1_score(y_test, y_pred_svm, average="macro", zero_division=0):.2f}')