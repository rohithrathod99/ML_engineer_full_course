# <center><u>Logistic Regression</u>

**Logsitic** word is because of logit function and **Regression** because under the hood we have LR.

**Supervised | Uses the concept of LR(Straight line equation, but modified to give results b/w 0 and 1) | Mostly used for Binary classification | Predict categorical/discrete values | Solves classification problems | Gives probabilistic value b/w 0 & 1 | Fit S shaped logistic/Sigmoid function which maps predicted value to probabalities(0 to 1) | To classify, we need to set some threshold | Types of Log reg - 1. Binomial/Multinomial(Dep var has unordered 2 or more values) 2. Ordinal(Dep var has 3 or more ordered values, for eg, low, medium, high)**

### Important pointers
1. BFL to sigmoid curve(S shape), where BFL can go from -inf to +inf but sigmoid can go from 0 to 1.
2. For multiclass, multiple S curve cut each other, each curve gives prob of one class(One vs rest - OVR)
3. Linear model is passed to the logistic function [ y = 1/(1+e^-x) ], result of which ranges from 0 to 1(also denoted as p). We should decide the cut-off/threshold.
4. Log loss/binary cross-entropy is the cost function, lower the better. README - [link]( https://www.analyticsvidhya.com/blog/2020/11/binary-cross-entropy-aka-log-loss-the-cost-function-used-in-logistic-regression/)
5. Since sigmoid function is non linear, so MSE can't be used as cost function because it's difficult to get global minima by GD in non-linear case.
6. Outlier has to be addressed because it's linear model. More the outlier, more the log loss.
7. Adv: Multiclass | also gived prob of each class | One of the best algo | quick to learn | Fast | White box | resitance to overfit | Interpret model coeff as feature importane.|
8. Disadv: Data dist should be linearly separable because it contructs linear boundaries.
9. L1 and L2 regularization technique can be used if overfit
10. README - [link](https://www.analyticsvidhya.com/blog/2021/10/building-an-end-to-end-logistic-regression-model/)

### Import Packages


```python
import pandas as pd
import numpy as np
import seaborn as sns
from tabulate import tabulate
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
import warnings
warnings.simplefilter('ignore')

# Remove max col display limit
pd.options.display.max_columns = None
# Print all cols in single line
pd.options.display.width=None
```

### Read CSV file


```python
df = pd.read_csv("diabetes.csv")
print(df.head().to_markdown()) 
```

    |    |   Pregnancies |   Glucose |   BloodPressure |   SkinThickness |   Insulin |   BMI |   DiabetesPedigreeFunction |   Age |   Outcome |
    |---:|--------------:|----------:|----------------:|----------------:|----------:|------:|---------------------------:|------:|----------:|
    |  0 |             6 |       148 |              72 |              35 |         0 |  33.6 |                      0.627 |    50 |         1 |
    |  1 |             1 |        85 |              66 |              29 |         0 |  26.6 |                      0.351 |    31 |         0 |
    |  2 |             8 |       183 |              64 |               0 |         0 |  23.3 |                      0.672 |    32 |         1 |
    |  3 |             1 |        89 |              66 |              23 |        94 |  28.1 |                      0.167 |    21 |         0 |
    |  4 |             0 |       137 |              40 |              35 |       168 |  43.1 |                      2.288 |    33 |         1 |
    

### Desciptive statistics of all the numeric columns


```python
print(df.describe().transpose().to_markdown())
```

    |                          |   count |       mean |        std |    min |      25% |      50% |       75% |    max |
    |:-------------------------|--------:|-----------:|-----------:|-------:|---------:|---------:|----------:|-------:|
    | Pregnancies              |     768 |   3.84505  |   3.36958  |  0     |  1       |   3      |   6       |  17    |
    | Glucose                  |     768 | 120.895    |  31.9726   |  0     | 99       | 117      | 140.25    | 199    |
    | BloodPressure            |     768 |  69.1055   |  19.3558   |  0     | 62       |  72      |  80       | 122    |
    | SkinThickness            |     768 |  20.5365   |  15.9522   |  0     |  0       |  23      |  32       |  99    |
    | Insulin                  |     768 |  79.7995   | 115.244    |  0     |  0       |  30.5    | 127.25    | 846    |
    | BMI                      |     768 |  31.9926   |   7.88416  |  0     | 27.3     |  32      |  36.6     |  67.1  |
    | DiabetesPedigreeFunction |     768 |   0.471876 |   0.331329 |  0.078 |  0.24375 |   0.3725 |   0.62625 |   2.42 |
    | Age                      |     768 |  33.2409   |  11.7602   | 21     | 24       |  29      |  41       |  81    |
    | Outcome                  |     768 |   0.348958 |   0.476951 |  0     |  0       |   0      |   1       |   1    |
    

### Check for missing values, datatypes of cols, memory, and NULL values


```python
display(df.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 768 entries, 0 to 767
    Data columns (total 9 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   Pregnancies               768 non-null    int64  
     1   Glucose                   768 non-null    int64  
     2   BloodPressure             768 non-null    int64  
     3   SkinThickness             768 non-null    int64  
     4   Insulin                   768 non-null    int64  
     5   BMI                       768 non-null    float64
     6   DiabetesPedigreeFunction  768 non-null    float64
     7   Age                       768 non-null    int64  
     8   Outcome                   768 non-null    int64  
    dtypes: float64(2), int64(7)
    memory usage: 54.1 KB
    


    None


### Pairplot to see the Distribution of all the vars, and scatter plot b/w 2 var


```python
sns.pairplot(df, diag_kind = "kde")         # Kernel density estimate
```




    <seaborn.axisgrid.PairGrid at 0x1e917229610>




    
![png](output_12_1.png)
    


### Pearson coeff of correlation


```python
corr = df.corr(method="pearson")
print(corr.to_markdown())
```

    |                          |   Pregnancies |   Glucose |   BloodPressure |   SkinThickness |    Insulin |       BMI |   DiabetesPedigreeFunction |        Age |   Outcome |
    |:-------------------------|--------------:|----------:|----------------:|----------------:|-----------:|----------:|---------------------------:|-----------:|----------:|
    | Pregnancies              |     1         | 0.129459  |       0.141282  |      -0.0816718 | -0.0735346 | 0.0176831 |                 -0.0335227 |  0.544341  | 0.221898  |
    | Glucose                  |     0.129459  | 1         |       0.15259   |       0.0573279 |  0.331357  | 0.221071  |                  0.137337  |  0.263514  | 0.466581  |
    | BloodPressure            |     0.141282  | 0.15259   |       1         |       0.207371  |  0.0889334 | 0.281805  |                  0.0412649 |  0.239528  | 0.0650684 |
    | SkinThickness            |    -0.0816718 | 0.0573279 |       0.207371  |       1         |  0.436783  | 0.392573  |                  0.183928  | -0.11397   | 0.0747522 |
    | Insulin                  |    -0.0735346 | 0.331357  |       0.0889334 |       0.436783  |  1         | 0.197859  |                  0.185071  | -0.042163  | 0.130548  |
    | BMI                      |     0.0176831 | 0.221071  |       0.281805  |       0.392573  |  0.197859  | 1         |                  0.140647  |  0.0362419 | 0.292695  |
    | DiabetesPedigreeFunction |    -0.0335227 | 0.137337  |       0.0412649 |       0.183928  |  0.185071  | 0.140647  |                  1         |  0.0335613 | 0.173844  |
    | Age                      |     0.544341  | 0.263514  |       0.239528  |      -0.11397   | -0.042163  | 0.0362419 |                  0.0335613 |  1         | 0.238356  |
    | Outcome                  |     0.221898  | 0.466581  |       0.0650684 |       0.0747522 |  0.130548  | 0.292695  |                  0.173844  |  0.238356  | 1         |
    

### Heatmap for coeff of correlation


```python
fig, ax = plt.subplots(figsize=(10,10))  
sns.heatmap(corr, annot = True, xticklabels= corr.columns, yticklabels= corr.columns, linewidths= 1, ax=ax)
```




    <AxesSubplot:>




    
![png](output_16_1.png)
    


### Check if the target column is balanced or not


```python
sns.countplot(data = df, x = "Outcome")
```




    <AxesSubplot:xlabel='Outcome', ylabel='count'>




    
![png](output_18_1.png)
    


## Implement Logistic Regression using Sklearn package


```python
# Conver to numpy arrays
np_array = df.values
X = np_array[:,0:7]
Y = np_array[:,8]

# Any number in random state is fine, signifies, whenever we run the code, train and test data should be same.
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# Create object and train the model
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)

# Predict the class
y_predict = log_reg.predict(x_test)
```

### Confusion matrix
   - TP, TN, FP/Type 1 error, and FN/Type 2 error
   - Accuracy: TP + TN / no. of rows
       + Not a good metric in case of imbalanced target class
   - Precision: TP / ( TP + FP )        
       + Used when FP is of higher concern, for eg, recommendation engine, suggesting irrelevant content to user
   - Recall: TP / ( TP + FN )
       + Used when FN is more concerned, for eg, medical case, predicting negative result for infected person
   - Combine both - F1 score: 
      + Harmonic mean of Precision and recall. F1-score = 2 * P * R / (P + R) 
      + Less interpretability, can't rely because it won't say which one is better, precision or recall? So, always use with other metrics.


```python
plot_confusion_matrix(log_reg, x_test, y_test)  
plt.show()
```


    
![png](output_22_0.png)
    


### Accuracy and AUC of Model


```python
accuracy = accuracy_score(y_true=y_test, y_pred=y_predict)
print("Model accuracy is: {}".format(accuracy))
roc_auc = roc_auc_score(y_test, y_predict)
print("AUC-ROC score is: {}".format(roc_auc))
```

    Model accuracy is: 0.75
    AUC-ROC score is: 0.7221633085896076
    

### Plot ROC-AUC curve
   - Probability curve and AUC represent measure of separability(how well model dintinguish b/w classes). Higher the AUC, better the model
   - TPR/Recall/Sensitivity: TP / ( TP + FN )
   - FPR: FP / ( FP + TN ). Also,  1 - Specificity 
   - AUC: 1 - Best | 0 - worst, Predicting opp classes | 0.5 - No class separation capacity
   - Specificity: TN / ( TN + FP ).
   - When AUC = 0.72(threshold as 0.5), meaning, 72% chance that the model will be able to predict correct class.
   - Sensitivity and Specificity are inversely propotional because when we increase threshold, TP increases and vice versa.
   - For multiclass, we plot AUC of One vs rest(OVR)
   - README - [Link](https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5)


```python
fpr, tpr, threshold = roc_curve(y_test, y_predict)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
```


    
![png](output_26_0.png)
    


### Summary of Classification report


```python
print(classification_report(y_true=y_test, y_pred=y_predict))
```

                  precision    recall  f1-score   support
    
             0.0       0.80      0.82      0.81       123
             1.0       0.66      0.62      0.64        69
    
        accuracy                           0.75       192
       macro avg       0.73      0.72      0.72       192
    weighted avg       0.75      0.75      0.75       192
    
    
