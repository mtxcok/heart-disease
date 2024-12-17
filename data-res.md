```
数据集列名： ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'HeartDisease']

数据预览：
   age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  slope   ca  thal  HeartDisease
0   63    1   1       145   233    1  2         150        0      2.3      3    0.0   6.0          0
1   67    1   4       160   286    0  2         108        1      1.5      2    3.0   3.0          2
2   67    1   4       120   229    0  2         129        1      2.6      2    2.0   7.0          1
3   37    1   3       130   250    0  2         187        0      3.5      3    0.0   3.0          0
4   41    0   2       130   204    0  2         172        0      1.4      1    0.0   3.0          0

[5 rows x 14 columns]

数据信息：
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 303 entries, 0 to 302
Data columns (total 14 columns):
 #   Column        Non-Null Count  Dtype
---  ------        --------------  -----
 0   age           303 non-null    int64
 1   sex           303 non-null    int64
 2   cp            303 non-null    int64
 3   trestbps      303 non-null    int64
 4   chol          303 non-null    int64
 5   fbs           303 non-null    int64
 6   restecg       303 non-null    int64
 7   thalach       303 non-null    int64
 8   exang         303 non-null    int64
 9   oldpeak       303 non-null    float64
 10  slope         303 non-null    int64
 11  ca            299 non-null    float64
 12  thal          301 non-null    float64
 13  HeartDisease  303 non-null    int64
dtypes: float64(3), int64(11)
memory usage: 33.3 KB
None

数据描述：
              age         sex          cp  ...          ca        thal  HeartDisease
count  303.000000  303.000000  303.000000  ...  299.000000  301.000000    303.000000     
mean    54.438944    0.679868    3.158416  ...    0.672241    4.734219      0.937294     
std      9.038662    0.467299    0.960126  ...    0.937438    1.939706      1.228536     
min     29.000000    0.000000    1.000000  ...    0.000000    3.000000      0.000000     
25%     48.000000    0.000000    3.000000  ...    0.000000    3.000000      0.000000     
50%     56.000000    1.000000    3.000000  ...    0.000000    3.000000      0.000000     
75%     61.000000    1.000000    4.000000  ...    1.000000    7.000000      2.000000     
max     77.000000    1.000000    4.000000  ...    3.000000    7.000000      4.000000     

[8 rows x 14 columns]
均值:
 age          54.438944
trestbps    131.689769
chol        246.693069
thalach     149.607261
oldpeak       1.039604
dtype: float64

方差:
 age           81.697419
trestbps     309.751120
chol        2680.849190
thalach      523.265775
oldpeak        1.348095
dtype: float64
age: Statistics=0.986, p=0.006
age 数据不服从正态分布。

trestbps: Statistics=0.967, p=0.000
trestbps 数据不服从正态分布。

chol: Statistics=0.947, p=0.000
chol 数据不服从正态分布。

thalach: Statistics=0.976, p=0.000
thalach 数据不服从正态分布。

oldpeak: Statistics=0.844, p=0.000
oldpeak 数据不服从正态分布。

age: t-statistic=-2.135, p-value=0.035
age 在两组之间存在显著差异。

trestbps: t-statistic=-1.461, p-value=0.148
trestbps 在两组之间不存在显著差异。

chol: t-statistic=-0.937, p-value=0.350
chol 在两组之间不存在显著差异。

thalach: t-statistic=3.635, p-value=0.000
thalach 在两组之间存在显著差异。

oldpeak: t-statistic=-2.846, p-value=0.006
oldpeak 在两组之间存在显著差异。

缺失值统计：
age             0
sex             0
cp              0
trestbps        0
chol            0
fbs             0
restecg         0
thalach         0
exang           0
oldpeak         0
slope           0
ca              4
thal            2
HeartDisease    0
dtype: int64

处理后的缺失值统计：
age             0
sex             0
cp              0
trestbps        0
chol            0
fbs             0
restecg         0
thalach         0
exang           0
oldpeak         0
slope           0
ca              0
thal            0
HeartDisease    0
dtype: int64
age: 下限=28.5, 上限=80.5
trestbps: 下限=90.0, 上限=170.0
chol: 下限=116.5, 上限=368.5
thalach: 下限=79.5, 上限=219.5
oldpeak: 下限=-2.4000000000000004, 上限=4.0

处理后的数据预览：
        age  sex  trestbps      chol  ...  slope_2  slope_3  thal_6.0  thal_7.0
0  0.980969    1  0.973671 -0.213548  ...    False     True      True     False
1  1.420393    1  1.946200  0.973426  ...     True    False     False     False
2  1.420393    1 -0.647211 -0.303131  ...     True    False     False      True
3 -1.875291    1  0.001141  0.167180  ...    False     True     False     False
4 -1.435866    0  0.001141 -0.863025  ...    False    False     False     False

[5 rows x 19 columns]
原始维度: (284, 18)
降维后维度: (284, 10)
训练集样本数: 227
测试集样本数: 57
随机森林混淆矩阵:
[[29  2  1  0  0]
 [ 6  2  0  3  0]
 [ 2  1  2  1  0]
 [ 1  3  1  1  0]
 [ 0  1  0  1  0]]

随机森林分类报告:
              precision    recall  f1-score   support

           0       0.76      0.91      0.83        32
           1       0.22      0.18      0.20        11
           2       0.50      0.33      0.40         6
           3       0.17      0.17      0.17         6
           4       0.00      0.00      0.00         2

    accuracy                           0.60        57
   macro avg       0.33      0.32      0.32        57
weighted avg       0.54      0.60      0.56        57

随机森林准确率: 0.60
随机森林精确率(macro): 0.33
随机森林召回率(macro): 0.32
随机森林F1分数(macro): 0.32

SVM混淆矩阵:
[[23  5  3  0  1]
 [ 3  5  1  2  0]
 [ 0  1  3  1  1]
 [ 0  1  2  2  1]
 [ 0  0  0  0  2]]

SVM分类报告:
              precision    recall  f1-score   support

           0       0.88      0.72      0.79        32
           1       0.42      0.45      0.43        11
           2       0.33      0.50      0.40         6
           3       0.40      0.33      0.36         6
           4       0.40      1.00      0.57         2

    accuracy                           0.61        57
   macro avg       0.49      0.60      0.51        57
weighted avg       0.67      0.61      0.63        57

SVM准确率: 0.61
SVM精确率(macro): 0.49
SVM召回率(macro): 0.60
SVM F1分数(macro): 0.51
```