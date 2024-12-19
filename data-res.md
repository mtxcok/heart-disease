```
数据加载检查：
X形状: (303, 13)
y形状: (303,)
y的前几个值: 0    0
1    2
2    1
3    0
4    0
Name: 0, dtype: int64
   age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  slope   ca  thal        
0   63    1   1       145   233    1        2      150      0      2.3      3  0.0   6.0        
1   67    1   4       160   286    0        2      108      1      1.5      2  3.0   3.0        
2   67    1   4       120   229    0        2      129      1      2.6      2  2.0   7.0        
3   37    1   3       130   250    0        0      187      0      3.5      3  0.0   3.0        
4   41    0   2       130   204    0        2      172      0      1.4      1  0.0   3.0        

Information about y:
<class 'pandas.core.series.Series'>
RangeIndex: 303 entries, 0 to 302
Series name: 0
Non-Null Count  Dtype
--------------  -----
303 non-null    int64
dtypes: int64(1)
memory usage: 2.5 KB
None
Unique values in y: [0 2 1 3 4]

Dataset Information:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 303 entries, 0 to 302
Data columns (total 13 columns):
 #   Column    Non-Null Count  Dtype
---  ------    --------------  -----
 0   age       303 non-null    int64
 1   sex       303 non-null    int64
 2   cp        303 non-null    int64
 3   trestbps  303 non-null    int64
 4   chol      303 non-null    int64
 5   fbs       303 non-null    int64
 6   restecg   303 non-null    int64
 7   thalach   303 non-null    int64
 8   exang     303 non-null    int64
 9   oldpeak   303 non-null    float64
 10  slope     303 non-null    int64
 11  ca        299 non-null    float64
 12  thal      301 non-null    float64
dtypes: float64(3), int64(10)
memory usage: 30.9 KB
None

Plotting feature distributions...

Plotting target variable distribution...

Plotting correlation heatmap...

Plotting relationship between features and target...

特征中的缺失值数量：
age         0
sex         0
cp          0
trestbps    0
chol        0
fbs         0
restecg     0
thalach     0
exang       0
oldpeak     0
slope       0
ca          4
thal        2
dtype: int64

目标变量中的缺失值数量： 0

目标变量y的基本信息：
形状: (303,)
数据类型: int64
唯一值: [0 2 1 3 4]
缺失值数量: 0

数据预处理完成。
处理后的X形状： (303, 13)
处理后的y形状： (303,)

训练随机森林模型...

随机森林模型评估结果：
混淆矩阵:
[[28  0  1  0  0]
 [ 5  0  5  1  1]
 [ 0  3  3  3  0]
 [ 0  4  1  2  0]
 [ 0  2  1  1  0]]

分类报告:
              precision    recall  f1-score   support

           0       0.85      0.97      0.90        29
           1       0.00      0.00      0.00        12
           2       0.27      0.33      0.30         9
           3       0.29      0.29      0.29         7
           4       0.00      0.00      0.00         4

    accuracy                           0.54        61
   macro avg       0.28      0.32      0.30        61
weighted avg       0.48      0.54      0.51        61

准确率: 0.54
精确率(macro): 0.28
召回率(macro): 0.32
F1分数(macro): 0.30

训练SVM模型...

SVM模型评估结果：
混淆矩阵:
[[23  4  1  1  0]
 [ 2  3  5  2  0]
 [ 0  3  1  4  1]
 [ 0  4  2  1  0]
 [ 0  0  2  2  0]]

分类报告:
              precision    recall  f1-score   support

           0       0.92      0.79      0.85        29
           1       0.21      0.25      0.23        12
           2       0.09      0.11      0.10         9
           3       0.10      0.14      0.12         7
           4       0.00      0.00      0.00         4

    accuracy                           0.46        61
   macro avg       0.27      0.26      0.26        61
weighted avg       0.50      0.46      0.48        61

准确率: 0.46
精确率(macro): 0.27
召回率(macro): 0.26
F1分数(macro): 0.26

y的信息：
类型: <class 'numpy.ndarray'>
形状: (303,)
前几个值: [0 2 1 0 0]
```
