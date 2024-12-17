import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from ucimlrepo import fetch_ucirepo

# 设置绘图风格
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# 获取数据集
heart_disease = fetch_ucirepo(id=45)

# 获取特征和目标
X = heart_disease.data.features
y = heart_disease.data.targets

# 转换为 DataFrame 以便于处理和可视化
# 假设 heart_disease.feature_names 包含特征的列名
X = pd.DataFrame(X, columns=heart_disease.feature_names)
y = pd.DataFrame(y, columns=['target'])  # 确保 y 是 DataFrame 并命名列

# 确保 y 是一维的
y = y.squeeze()  # 使用 squeeze() 将 y 转换为 Series

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
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整布局以为总标题留出空间
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

# 绘制热图
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

## 1.5. 绘制特征与目标的关系图
print("\nPlotting relationship between features and target...")
# 选择与目标变量相关性较高的前几项特征
top_features = corr_matrix['target']。abs()。sort_values(ascending=False).index[1:6]

for feature in top_features:
    # 检查 y 和 data[feature] 的长度是否一致
    if len(y) != len(data[feature]):
        print(f"Length mismatch: y length {len(y)}, {feature} length {len(data[feature])}")
        continue  # 跳过这个特征

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

# 2.1. 处理缺失值
# 使用 SimpleImputer 来填充缺失值（均值）
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# 2.2. 对分类特征进行编码（如果有的话）
# 假设 'sex' 列为分类数据
if 'sex' in X.columns:
    label_encoder = LabelEncoder()
    X['sex'] = label_encoder.fit_transform(X['sex'])

# 2.3. 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 打印一些信息
print("\nData Preprocessing Completed.")
print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)
