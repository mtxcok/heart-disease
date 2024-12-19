import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.impute import SimpleImputer

from ucimlrepo import fetch_ucirepo

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.metrics import precision_score, recall_score, f1_score

import numpy as np

  

# 设置绘图风格

sns.set(style="whitegrid")

plt.rcParams['figure.figsize'] = (10, 6)

  

# 获取数据集

heart_disease = fetch_ucirepo(id=45)

  

# 获取特征和目标

X = heart_disease.data.features

y = heart_disease.data.targets.values.flatten()  # 直接转换为一维numpy数组

  

# 转换为DataFrame和Series

X = pd.DataFrame(X, columns=heart_disease.feature_names)

y = pd.DataFrame(y).iloc[:, 0]  # 直接取第一列作为Series

  

# 检查数据加载是否成功

print("\n数据加载检查：")

print("X形状:", X.shape)

print("y形状:", y.shape)

print("y的前几个值:", y[:5])

  

# 查看数据集的一部分

print(X.head())

  

# 检查 y 的信息

print("\nInformation about y:")

print(y.info())

print(f"Unique values in y: {y.unique()}")

  

# 1. 数据可视化

  

## 1.1. 查看特征的基本信息

print("\nDataset Information:")

print(X.info())

  

## 1.2. 绘制特征的分布图

print("\nPlotting feature distributions...")

X.hist(bins=30, figsize=(20, 15))

plt.suptitle("Feature Distributions")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整布局以为总标题留出空间

plt.show()

  

## 1.3. 查看目标变量的分布

print("\nPlotting target variable distribution...")

sns.countplot(x=y)

plt.title("Target Variable Distribution")

plt.xlabel("Target")

plt.ylabel("Count")

plt.show()

  

## 1.4. 绘制相关性热图

print("\nPlotting correlation heatmap...")

# 合并特征和目标以计算相关性

data = X.copy()

data['target'] = y

corr_matrix = data.corr()

  

# 绘热图

plt.figure(figsize=(12, 10))

sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')

plt.title("Feature Correlation Heatmap")

plt.show()

  

## 1.5. 绘制特征与目标的关系图

print("\nPlotting relationship between features and target...")

# 选择与目标变量相关性较高的前几项特征

top_features = corr_matrix['target'].abs().sort_values(ascending=False).index[1:6]

  

for feature in top_features:

    # 检查 y 和 data[feature] 的长度是否一致

    if len(y) != len(data[feature]):

        print(f"Length mismatch: y length {len(y)}, {feature} length {len(data[feature])}")

        continue  # 跳过这个特征

  

    # 检查 y 是否为分类变量

    if y.dtype not in ['object', 'category']:

        # 将 y 转换为字符串类型

        y_plot = y.astype(str)

    else:

        y_plot = y

  

    # 检查是否存在缺失值

    if y_plot.isnull().sum() > 0 or data[feature].isnull().sum() > 0:

        # 删除 'target' 或该特征中包含缺失值的行

        data_clean = data.dropna(subset=['target', feature])

        y_clean = data_clean['target'].astype(str)

        feature_clean = data_clean[feature]

    else:

        y_clean = y_plot

        feature_clean = data[feature]

  

    plt.figure(figsize=(8, 6))

    sns.boxplot(x=y_clean, y=feature_clean)

    plt.title(f"Relationship between {feature} and Target Variable")

    plt.xlabel("Target")

    plt.ylabel(feature)

    plt.show()

  

# 2. 数据预处理

  

# 2.1. 检查缺失值情况

print("\n特征中的缺失值数量：")

print(X.isnull().sum())

print("\n目标变量中的缺失值数量：", y.isnull().sum())

  

# 2.2. 处理特征X中的缺失值

# 使用 SimpleImputer 来填充缺失值（均值）

imputer = SimpleImputer(strategy='mean')

X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

  

# 2.3. 处理目标变量y中的缺失值（如果有的话）

print("\n目标变量y的基本信息：")

print("形状:", y.shape)

print("数据类型:", y.dtype)

print("唯一值:", y.unique())

print("缺失值数量:", y.isnull().sum())

  

# 确保y是一维数组

if isinstance(y, pd.DataFrame):

    y = y.squeeze()  # 如果y是DataFrame，转换为Series

if isinstance(y, pd.Series):

    y = y.to_numpy()  # 转换为numpy数组

  

# 检查y是否为空

if len(y) == 0:

    raise ValueError("目标变量y是空的！请检查数据加载步骤。")

  

# 2.4. 对分类特征进行编码

categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

for feature in categorical_features:

    if feature in X.columns:

        label_encoder = LabelEncoder()

        X[feature] = label_encoder.fit_transform(X[feature])

  

# 2.5. 标准化数据

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

  

# 打印处理后的信息

print("\n数据预处理完成。")

print("处理后的X形状：", X.shape)

print("处理后的y形状：", y.shape)

  

# 3. 划分训练集和测试集

X_train, X_test, y_train, y_test = train_test_split(

    X_scaled, y, test_size=0.2, random_state=42

)

  

# 4. 模型搭建

  

# 4.1 随机森林模型

print("\n训练随机森林模型...")

rf_model = RandomForestClassifier(

    n_estimators=200,      # 树的数量

    max_depth=15,          # 树的最大深度

    min_samples_split=4,   # 分裂节点所需的最小样本数

    min_samples_leaf=2,    # 叶节点所需的最小样本数

    random_state=42,       # 随机种子

    n_jobs=-1,            # 使用所有CPU核心

    class_weight='balanced'  # 处理类别不平衡

)

  

# 训练模型

rf_model.fit(X_train, y_train)

  

# 预测

y_pred_rf = rf_model.predict(X_test)

  

# 评估随机森林模型

print("\n随机森林模型评估结果：")

print("混淆矩阵:")

print(confusion_matrix(y_test, y_pred_rf))

print("\n分类报告:")

print(classification_report(y_test, y_pred_rf, zero_division=0))

print(f'准确率: {accuracy_score(y_test, y_pred_rf):.2f}')

print(f'精确率(macro): {precision_score(y_test, y_pred_rf, average="macro", zero_division=0):.2f}')

print(f'召回率(macro): {recall_score(y_test, y_pred_rf, average="macro", zero_division=0):.2f}')

print(f'F1分数(macro): {f1_score(y_test, y_pred_rf, average="macro", zero_division=0):.2f}')

  

# 4.2 SVM模型

print("\n训练SVM模型...")

svm_model = SVC(

    kernel='rbf',          # 使用RBF核函数

    C=1.0,                # 正则参数

    gamma='scale',        # 核函数系数

    random_state=42,      # 随机种子

    class_weight='balanced'  # 处理类别不平衡

)

  

# 训练模型

svm_model.fit(X_train, y_train)

  

# 预测

y_pred_svm = svm_model.predict(X_test)

  

# 评估SVM模型

print("\nSVM模型评估结果：")

print("混淆矩阵:")

print(confusion_matrix(y_test, y_pred_svm))

print("\n分类报告:")

print(classification_report(y_test, y_pred_svm, zero_division=0))

print(f'准确率: {accuracy_score(y_test, y_pred_svm):.2f}')

print(f'精确率(macro): {precision_score(y_test, y_pred_svm, average="macro", zero_division=0):.2f}')

print(f'召回率(macro): {recall_score(y_test, y_pred_svm, average="macro", zero_division=0):.2f}')

print(f'F1分数(macro): {f1_score(y_test, y_pred_svm, average="macro", zero_division=0):.2f}')

  

# 4.3 特征重要性分析（随机森林）

feature_importance = pd.DataFrame({

    'feature': X.columns,

    'importance': rf_model.feature_importances_

})

feature_importance = feature_importance.sort_values('importance', ascending=False)

  

plt.figure(figsize=(10, 6))

sns.barplot(x='importance', y='feature', data=feature_importance)

plt.title('Feature Importance (Random Forest)')

plt.show()

  

print("\ny的信息：")

print("类型:", type(y))

print("形状:", y.shape if hasattr(y, 'shape') else len(y))

print("前几个值:", y[:5])
