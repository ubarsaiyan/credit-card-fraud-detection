
# Detecting Fraudulent Credit Card Transactions Through Data Science

<h4 style="text-align: center;"> By Utkarsh Barsaiyan</h4>

### Abstract

The advent of online transactions has its downsides too. Each year, millions of people all across the globe fall victim to credit card fraud leading to losses in billions of dollars. Therefore, it is the need of the hour to create algorithms that can accurately detect such fraudulent transactions by analysing the various details such as the time of the transaction, the amount transacted, the account numbers, etc. Putting data science and machine learning to good use, this project uses XGBoost algorithm along with anomaly detection to weed out the fraudulent transactions using a highly imbalanced dataset obtained from <a href="https://www.kaggle.com/mlg-ulb/creditcardfraud">Kaggle</a>. This IPython Notebook can be found <a href="https://github.com/ubarsaiyan/credit-card-fraud-detection">here</a>. 

### SMOTE (Synthetic Minority Oversampling Technique)

Machine Learning algorithms perform poorly on highly imbalanced dataset such as the one we have for the credit card fraud. Multiple techniques have been developed to mitigate this problem. Undersampling and oversampling are two of the most popular approaches. In undersampling, we randomly sample from the majority class the number of data points equal to that of the minority class. This creates a balanced dataset but loses on vital information from the majority class data that was not selected.

Oversampling on the other hand uses methods to either resample the minority data points or generate new minority data points. SMOTE synthesises new minority instances between existing minority instances. Figuratively, it can be assumed that SMOTE draws lines between existing minority instances like the figure given below. It then interpolates the real minority instances along these lines to generate new data points.  

![smote](files/smote.png)

### XGBoost Algorithm

XGBoost is an ensemble learning method. Sometimes, it may not be sufficient to rely upon the results of just one machine learning model. Ensemble learning offers a systematic solution to combine the predictive power of multiple learners. The resultant is a single model which gives the aggregated output from several models.

![mistakes](files/mistakes.png)

Boosting is a sequential technique which works on the principle of an ensemble. It combines a set of weak learners and delivers improved prediction accuracy. At any instant t, the model outcomes are weighed based on the outcomes of previous instant t-1. The outcomes predicted correctly are given a lower weight and the ones misclassified are weighted higher. Note that a weak learner is one which is slightly better than random guessing. For example, a decision tree whose predictions are slightly better than 50%. The algorithm is represented in the figure below.

![adaboost](files/adaboost.png)

Just like the above boosting algorithm called AdaBoost, Gradient Boosting works by sequentially adding predictors to an ensemble, each one correcting its predecessor. However, instead of changing the weights for every incorrect classified observation at every iteration like AdaBoost, Gradient Boosting method tries to fit the new predictor to the residual errors made by the previous predictor.

![xgboost](files/xgboost.png)

### Code & Observations


```python
# imports
%matplotlib inline
import scipy.stats as stats
import numpy as np
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import simplefilter
plt.style.use('seaborn')
pandas.set_option('precision', 3)
simplefilter(action='ignore', category=FutureWarning)
```


```python
df = pandas.read_csv('creditcard.csv')
```


```python
# sample data
df.sample(10) 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>53914</th>
      <td>46211.0</td>
      <td>-0.529</td>
      <td>1.000</td>
      <td>-0.658</td>
      <td>-0.238</td>
      <td>2.370</td>
      <td>3.415</td>
      <td>-0.142</td>
      <td>1.259</td>
      <td>-0.959</td>
      <td>...</td>
      <td>0.154</td>
      <td>0.188</td>
      <td>-0.142</td>
      <td>1.007</td>
      <td>-0.075</td>
      <td>-0.318</td>
      <td>0.051</td>
      <td>0.085</td>
      <td>10.89</td>
      <td>0</td>
    </tr>
    <tr>
      <th>283803</th>
      <td>171889.0</td>
      <td>-1.584</td>
      <td>1.264</td>
      <td>1.646</td>
      <td>-0.752</td>
      <td>0.192</td>
      <td>0.217</td>
      <td>0.588</td>
      <td>-0.310</td>
      <td>0.964</td>
      <td>...</td>
      <td>-0.172</td>
      <td>-0.594</td>
      <td>-0.263</td>
      <td>0.524</td>
      <td>0.275</td>
      <td>-0.553</td>
      <td>-0.961</td>
      <td>-0.031</td>
      <td>14.99</td>
      <td>0</td>
    </tr>
    <tr>
      <th>91349</th>
      <td>63427.0</td>
      <td>-1.328</td>
      <td>-0.129</td>
      <td>0.734</td>
      <td>-0.342</td>
      <td>-0.766</td>
      <td>0.354</td>
      <td>1.079</td>
      <td>0.069</td>
      <td>-1.317</td>
      <td>...</td>
      <td>-0.426</td>
      <td>-0.672</td>
      <td>-0.004</td>
      <td>-0.325</td>
      <td>-0.252</td>
      <td>-0.651</td>
      <td>-0.084</td>
      <td>-0.308</td>
      <td>254.80</td>
      <td>0</td>
    </tr>
    <tr>
      <th>118688</th>
      <td>75174.0</td>
      <td>1.028</td>
      <td>-0.101</td>
      <td>0.688</td>
      <td>1.407</td>
      <td>-0.473</td>
      <td>0.204</td>
      <td>-0.254</td>
      <td>0.234</td>
      <td>0.362</td>
      <td>...</td>
      <td>0.043</td>
      <td>0.260</td>
      <td>-0.098</td>
      <td>0.237</td>
      <td>0.569</td>
      <td>-0.273</td>
      <td>0.039</td>
      <td>0.014</td>
      <td>40.00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>195020</th>
      <td>130863.0</td>
      <td>0.183</td>
      <td>1.061</td>
      <td>-1.435</td>
      <td>0.505</td>
      <td>1.302</td>
      <td>-0.622</td>
      <td>1.392</td>
      <td>-0.665</td>
      <td>-0.462</td>
      <td>...</td>
      <td>0.318</td>
      <td>1.248</td>
      <td>-0.113</td>
      <td>-0.970</td>
      <td>-0.687</td>
      <td>-0.068</td>
      <td>-0.039</td>
      <td>0.252</td>
      <td>23.59</td>
      <td>0</td>
    </tr>
    <tr>
      <th>99417</th>
      <td>67110.0</td>
      <td>1.216</td>
      <td>0.170</td>
      <td>-0.254</td>
      <td>0.841</td>
      <td>0.180</td>
      <td>0.140</td>
      <td>-0.184</td>
      <td>0.102</td>
      <td>0.670</td>
      <td>...</td>
      <td>-0.210</td>
      <td>-0.332</td>
      <td>-0.167</td>
      <td>-0.833</td>
      <td>0.580</td>
      <td>0.494</td>
      <td>0.016</td>
      <td>0.031</td>
      <td>12.31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>48269</th>
      <td>43571.0</td>
      <td>1.133</td>
      <td>-0.660</td>
      <td>0.590</td>
      <td>-0.150</td>
      <td>-0.493</td>
      <td>1.065</td>
      <td>-0.856</td>
      <td>0.445</td>
      <td>1.129</td>
      <td>...</td>
      <td>-0.216</td>
      <td>-0.480</td>
      <td>-0.072</td>
      <td>-1.093</td>
      <td>0.150</td>
      <td>1.026</td>
      <td>-0.035</td>
      <td>-0.006</td>
      <td>46.90</td>
      <td>0</td>
    </tr>
    <tr>
      <th>160275</th>
      <td>113194.0</td>
      <td>-0.932</td>
      <td>0.011</td>
      <td>2.228</td>
      <td>-2.261</td>
      <td>-0.736</td>
      <td>-0.762</td>
      <td>0.019</td>
      <td>0.188</td>
      <td>-1.062</td>
      <td>...</td>
      <td>0.006</td>
      <td>-0.149</td>
      <td>-0.119</td>
      <td>0.492</td>
      <td>0.322</td>
      <td>-0.481</td>
      <td>0.246</td>
      <td>0.114</td>
      <td>28.00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>30036</th>
      <td>35748.0</td>
      <td>1.147</td>
      <td>-0.567</td>
      <td>0.976</td>
      <td>0.086</td>
      <td>-1.103</td>
      <td>0.108</td>
      <td>-0.809</td>
      <td>0.280</td>
      <td>1.136</td>
      <td>...</td>
      <td>-0.203</td>
      <td>-0.465</td>
      <td>0.021</td>
      <td>0.060</td>
      <td>0.104</td>
      <td>0.946</td>
      <td>-0.050</td>
      <td>0.002</td>
      <td>31.00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>224070</th>
      <td>143638.0</td>
      <td>-3.235</td>
      <td>-1.619</td>
      <td>-2.405</td>
      <td>-3.072</td>
      <td>-2.313</td>
      <td>-0.935</td>
      <td>1.773</td>
      <td>0.054</td>
      <td>-0.319</td>
      <td>...</td>
      <td>-0.426</td>
      <td>0.196</td>
      <td>0.044</td>
      <td>0.166</td>
      <td>0.741</td>
      <td>-0.747</td>
      <td>-0.510</td>
      <td>-0.618</td>
      <td>442.79</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 31 columns</p>
</div>



Most of the features of the data has been anonymized (V1 to V28) except the time of transaction, its amount and whether it is fraudulent or not.


```python
# summary of the time and amount features
df.loc[:, ['Time', 'Amount']].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>Amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>284807.000</td>
      <td>284807.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>94813.860</td>
      <td>88.350</td>
    </tr>
    <tr>
      <th>std</th>
      <td>47488.146</td>
      <td>250.120</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>54201.500</td>
      <td>5.600</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>84692.000</td>
      <td>22.000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>139320.500</td>
      <td>77.165</td>
    </tr>
    <tr>
      <th>max</th>
      <td>172792.000</td>
      <td>25691.160</td>
    </tr>
  </tbody>
</table>
</div>



#### Data Visualisation


```python
# visualising the time distribution
plt.figure(figsize=(10,9))
plt.title('Distribution of the Time Feature')
sns.distplot(df['Time'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f50e11b7160>




![png](output_22_1.png)


There are significant peaks and lows in the time distribution indicating the huge difference between the number of transactions occuring during the day and night.


```python
# Visualising the amount distribution
plt.figure(figsize=(10,9))
plt.title('Distribution of the Amount Feature')
sns.distplot(df['Amount'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f50e4b76128>




![png](output_24_1.png)


While the mean of the transaction amount stands at \\$88.35, the maximum amount is \\$25691.16. But the data is heavily skewed and most of the transactions are low valued.


```python
counts = df['Class'].value_counts()
correct = counts[0]
fraudulent = counts[1]
correct_perc = (correct/(correct+fraudulent))*100
fraudulent_perc = (fraudulent/(correct+fraudulent))*100
print('There are {} ({:.3f}%) non-fraudulent transactions and {} ({:.3f}%) fraudulent transactions.'.format(correct, correct_perc, fraudulent, fraudulent_perc))
```

    There are 284315 (99.827%) non-fraudulent transactions and 492 (0.173%) fraudulent transactions.



```python
plt.figure(figsize=(8,10))
sns.barplot(x=counts.index, y=counts)
plt.title('Number of Non-Fraudulent and Fraudulent Transactions')
plt.ylabel('Number')
plt.xlabel('Class - 0:Non-Fraudulent; 1:Fraudulent')
```




    Text(0.5, 0, 'Class - 0:Non-Fraudulent; 1:Fraudulent')




![png](output_27_1.png)


More than 99% of the transactions are non-fraudulent. So, our dataset is heavily imbalanced.


```python
# heatmap of correlation between the predictor variables
corr = df.corr()
plt.figure(figsize=(12,10))
sns.heatmap(data=corr)
plt.title('Heatmap of Correlation')
```




    Text(0.5, 1.0, 'Heatmap of Correlation')




![png](output_29_1.png)


Some of the predictors do seem to be correlated with the Class variable but there seem to be relatively little significant correlations for such a large number of variables.

#### Scaling the Time and Amount features 


```python
from sklearn import preprocessing

# scale the time column
scaler_time = preprocessing.StandardScaler().fit(df[['Time']])
scaled_time = scaler_time.transform(df[['Time']])
scaled_time = scaled_time.flatten()

# scale the amount column
scaler_amount = preprocessing.StandardScaler().fit(df[['Amount']])
scaled_amount = scaler_amount.transform(df[['Amount']])
scaled_amount = scaled_amount.flatten()
```


```python
# insert the scaled time and amount columns  
df.insert(0, "scaled_amount", scaled_amount, True)
df.insert(0, "scaled_time", scaled_time, True)

# delete the old time and amount columns
df.drop(['Amount', 'Time'], axis=1, inplace=True)
```


```python
df.sample(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>scaled_time</th>
      <th>scaled_amount</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>...</th>
      <th>V20</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>245380</th>
      <td>1.220</td>
      <td>0.487</td>
      <td>-4.338</td>
      <td>-1.240</td>
      <td>-1.777e+00</td>
      <td>1.026</td>
      <td>2.253</td>
      <td>-1.008</td>
      <td>0.805</td>
      <td>-0.125</td>
      <td>...</td>
      <td>-1.018</td>
      <td>-0.649</td>
      <td>-0.483</td>
      <td>-2.707</td>
      <td>0.373</td>
      <td>1.037</td>
      <td>-0.452</td>
      <td>0.492</td>
      <td>-1.771e+00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>38171</th>
      <td>-1.169</td>
      <td>-0.241</td>
      <td>1.323</td>
      <td>-0.752</td>
      <td>8.457e-01</td>
      <td>-0.521</td>
      <td>-1.580</td>
      <td>-0.766</td>
      <td>-0.912</td>
      <td>-0.062</td>
      <td>...</td>
      <td>-0.466</td>
      <td>-0.192</td>
      <td>-0.108</td>
      <td>0.011</td>
      <td>0.414</td>
      <td>0.121</td>
      <td>1.154</td>
      <td>-0.036</td>
      <td>1.843e-02</td>
      <td>0</td>
    </tr>
    <tr>
      <th>47571</th>
      <td>-1.086</td>
      <td>-0.232</td>
      <td>1.242</td>
      <td>-0.214</td>
      <td>9.859e-04</td>
      <td>-1.404</td>
      <td>-0.432</td>
      <td>-1.006</td>
      <td>0.084</td>
      <td>-0.278</td>
      <td>...</td>
      <td>0.035</td>
      <td>0.120</td>
      <td>0.531</td>
      <td>-0.135</td>
      <td>0.194</td>
      <td>0.608</td>
      <td>0.153</td>
      <td>0.013</td>
      <td>1.372e-02</td>
      <td>0</td>
    </tr>
    <tr>
      <th>251012</th>
      <td>1.271</td>
      <td>-0.293</td>
      <td>2.388</td>
      <td>-1.284</td>
      <td>-1.845e+00</td>
      <td>-1.978</td>
      <td>-0.425</td>
      <td>-0.295</td>
      <td>-0.870</td>
      <td>-0.198</td>
      <td>...</td>
      <td>-0.405</td>
      <td>0.029</td>
      <td>0.533</td>
      <td>-0.109</td>
      <td>-1.377</td>
      <td>0.235</td>
      <td>0.137</td>
      <td>-0.021</td>
      <td>-8.543e-02</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14146</th>
      <td>-1.467</td>
      <td>-0.313</td>
      <td>1.629</td>
      <td>-0.832</td>
      <td>1.043e-02</td>
      <td>-1.454</td>
      <td>-0.755</td>
      <td>-0.168</td>
      <td>-1.012</td>
      <td>-0.261</td>
      <td>...</td>
      <td>-0.309</td>
      <td>-0.455</td>
      <td>-0.635</td>
      <td>-0.127</td>
      <td>-1.078</td>
      <td>0.565</td>
      <td>-0.159</td>
      <td>0.005</td>
      <td>6.634e-04</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>



We need to scale only the Time and Amount features as all the other features (V1 to V28) were prepared using PCA.

#### Splitting the dataset into training and test data


```python
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2)
```


```python
train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)
```

#### Using SMOTE to remove data imbalance


```python
X_train = train.loc[:,:'V28']
y_train = train.loc[:,'Class']
```


```python
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
```


```python
train_res = np.concatenate((X_train_res, y_train_res.reshape((-1,1))), axis = 1)
np.random.shuffle(train_res)
train_res = pandas.DataFrame(train_res, columns=train.columns)
```


```python
counts = train_res['Class'].value_counts()
plt.figure(figsize=(8,10))
sns.barplot(x=counts.index, y=counts)
plt.title('Number of Non-Fraudulent and Fraudulent Transactions After Oversampling')
plt.ylabel('Number')
plt.xlabel('Class - 0:Non-Fraudulent; 1:Fraudulent')
```




    Text(0.5, 0, 'Class - 0:Non-Fraudulent; 1:Fraudulent')




![png](output_43_1.png)


#### Outlier Detection and Removal


```python
corr = train_res.corr()
corr = corr[['Class']]
```


```python
# features with high negative correlations
corr[corr['Class'] < -0.5]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>V3</th>
      <td>-0.571</td>
    </tr>
    <tr>
      <th>V9</th>
      <td>-0.577</td>
    </tr>
    <tr>
      <th>V10</th>
      <td>-0.635</td>
    </tr>
    <tr>
      <th>V12</th>
      <td>-0.689</td>
    </tr>
    <tr>
      <th>V14</th>
      <td>-0.758</td>
    </tr>
    <tr>
      <th>V16</th>
      <td>-0.609</td>
    </tr>
    <tr>
      <th>V17</th>
      <td>-0.577</td>
    </tr>
  </tbody>
</table>
</div>




```python
# box plot of the features with high negative correlation
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(24, 30))
fig.suptitle('Features With High Negative Correlation', size=30)
sns.boxplot(x="Class", y="V3", data=train_res, ax=axes[0,0])
sns.boxplot(x="Class", y="V9", data=train_res, ax=axes[0,1])
sns.boxplot(x="Class", y="V10", data=train_res, ax=axes[0,2])
sns.boxplot(x="Class", y="V12", data=train_res, ax=axes[1,0])
sns.boxplot(x="Class", y="V14", data=train_res, ax=axes[1,1])
sns.boxplot(x="Class", y="V16", data=train_res, ax=axes[1,2])
sns.boxplot(x="Class", y="V17", data=train_res, ax=axes[2,0])
fig.delaxes(axes[2,1])
fig.delaxes(axes[2,2])
```


![png](output_47_0.png)



```python
# features with high positive correlations
corr[corr.Class > 0.5]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>V2</th>
      <td>0.510</td>
    </tr>
    <tr>
      <th>V4</th>
      <td>0.720</td>
    </tr>
    <tr>
      <th>V11</th>
      <td>0.697</td>
    </tr>
    <tr>
      <th>Class</th>
      <td>1.000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# box plot of the features with high positive correlation
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 12))
fig.suptitle('Features With High Positive Correlation', size=30)
sns.boxplot(x="Class", y="V4", data=train_res, ax=axes[0])
sns.boxplot(x="Class", y="V11", data=train_res, ax=axes[1])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f50e13b0c50>




![png](output_49_1.png)



```python
# removing extreme outliers
Q1 = train_res.quantile(0.25)
Q3 = train_res.quantile(0.75)
IQR = Q3 - Q1
train2 = train_res[~((train_res < (Q1-2.5*IQR)) | (train_res > (Q3+2.5*IQR))).any(axis=1)]
```


```python
len_after = len(train2)
len_before = len(train_res)
len_diff = len(train_res) - len(train2)
print('We reduced our data size from {} observations by {} observations to {} observations.'.format(len_before, len_diff, len_after))
```

    We reduced our data size from 454916 observations by 121524 observations to 333392 observations.



```python
counts = train2['Class'].value_counts()
plt.figure(figsize=(8,10))
sns.barplot(x=counts.index, y=counts)
plt.title('Number of Non-Fraudulent and Fraudulent Transactions After Oversampling')
plt.ylabel('Number')
plt.xlabel('Class - 0:Non-Fraudulent; 1:Fraudulent')
```




    Text(0.5, 0, 'Class - 0:Non-Fraudulent; 1:Fraudulent')




![png](output_52_1.png)


#### Visualising on 2D scatter plot using t-SNE


```python
from sklearn.manifold import TSNE

subtrain2 = train2.sample(1000)
X = subtrain2.loc[:,:'V28']
y = subtrain2.loc[:,'Class']
X_tsne = TSNE(n_components=2, random_state=42, verbose=5).fit_transform(X.values)
```

    [t-SNE] Computing 91 nearest neighbors...
    [t-SNE] Indexed 1000 samples in 0.001s...
    [t-SNE] Computed neighbors for 1000 samples in 0.055s...
    [t-SNE] Computed conditional probabilities for sample 1000 / 1000
    [t-SNE] Mean sigma: 2.012820
    [t-SNE] Computed conditional probabilities in 0.036s
    [t-SNE] Iteration 50: error = 65.4220886, gradient norm = 0.2819089 (50 iterations in 0.353s)
    [t-SNE] Iteration 100: error = 63.9896965, gradient norm = 0.2589646 (50 iterations in 0.349s)
    [t-SNE] Iteration 150: error = 64.3916626, gradient norm = 0.2558551 (50 iterations in 0.328s)
    [t-SNE] Iteration 200: error = 64.2604980, gradient norm = 0.2554580 (50 iterations in 0.332s)
    [t-SNE] Iteration 250: error = 64.1053238, gradient norm = 0.2419452 (50 iterations in 0.317s)
    [t-SNE] KL divergence after 250 iterations with early exaggeration: 64.105324
    [t-SNE] Iteration 300: error = 0.8575150, gradient norm = 0.0011415 (50 iterations in 0.284s)
    [t-SNE] Iteration 350: error = 0.7476299, gradient norm = 0.0004131 (50 iterations in 0.237s)
    [t-SNE] Iteration 400: error = 0.7202361, gradient norm = 0.0002191 (50 iterations in 0.245s)
    [t-SNE] Iteration 450: error = 0.7089534, gradient norm = 0.0001512 (50 iterations in 0.251s)
    [t-SNE] Iteration 500: error = 0.7019995, gradient norm = 0.0001353 (50 iterations in 0.260s)
    [t-SNE] Iteration 550: error = 0.6976496, gradient norm = 0.0001331 (50 iterations in 0.266s)
    [t-SNE] Iteration 600: error = 0.6944770, gradient norm = 0.0001098 (50 iterations in 0.263s)
    [t-SNE] Iteration 650: error = 0.6923516, gradient norm = 0.0000928 (50 iterations in 0.262s)
    [t-SNE] Iteration 700: error = 0.6904641, gradient norm = 0.0000923 (50 iterations in 0.275s)
    [t-SNE] Iteration 750: error = 0.6890587, gradient norm = 0.0000773 (50 iterations in 0.268s)
    [t-SNE] Iteration 800: error = 0.6878876, gradient norm = 0.0000744 (50 iterations in 0.269s)
    [t-SNE] Iteration 850: error = 0.6868870, gradient norm = 0.0000773 (50 iterations in 0.270s)
    [t-SNE] Iteration 900: error = 0.6858996, gradient norm = 0.0000798 (50 iterations in 0.271s)
    [t-SNE] Iteration 950: error = 0.6851627, gradient norm = 0.0000684 (50 iterations in 0.272s)
    [t-SNE] Iteration 1000: error = 0.6847250, gradient norm = 0.0000650 (50 iterations in 0.272s)
    [t-SNE] KL divergence after 1000 iterations: 0.684725



```python
# scatter plot
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(24,16))

blue_patch = mpatches.Patch(color='#0A0AFF', label='Non-Fraudulent')
red_patch = mpatches.Patch(color='#AF0000', label='Fraudulent')

ax.scatter(X_tsne[:,0], X_tsne[:,1], c=(y == 0), cmap='coolwarm', label='Non-Fraudulent', linewidths=2)
ax.scatter(X_tsne[:,0], X_tsne[:,1], c=(y == 1), cmap='coolwarm', label='Fraudulent', linewidths=2)
ax.set_title('t-SNE Scatter Plot', fontsize=14)
ax.grid(True)
ax.legend(handles=[blue_patch, red_patch])
```




    <matplotlib.legend.Legend at 0x7f50e0e4bbe0>




![png](output_55_1.png)


#### Classification using XGBoost


```python
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
```


```python
subtrain2 = train2.sample(20000)
X_train = subtrain2.loc[:,:'V28']
y_train = subtrain2.loc[:,'Class']
X_test = test.loc[:,:'V28'].values
y_test = test.loc[:,'Class'].values
```


```python
kfold = KFold(n_splits=10, random_state=42)
clf = XGBClassifier(objective ='reg:logistic')
cv_results = cross_val_score(clf, X_train, y_train, cv=kfold, scoring='roc_auc')
print('XGB ROCAUC on validation data: {0:.5f} {1:.5f}'.format(cv_results.mean(), cv_results.std()))
```

    XGB ROCAUC on validation data: 0.99900 0.00022


### Results


```python
clf.fit(X_train.as_matrix(), y_train.as_matrix(), eval_metric='auc', verbose=True)
y_pred = clf.predict(X_test)
roc_auc = roc_auc_score(y_test, y_pred)
print("XGB ROCAUC on test data: %f" % (roc_auc))
```

    XGB ROCAUC on test data: 0.941833



```python
cnf_matrix = confusion_matrix(y_test, y_pred)
```


```python
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        1#print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
```


```python
# plot confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.grid(False)
plt.show()
```


![png](output_64_0.png)


The ROCAUC metric on test data is 0.942. Out of the 56962 observations in the test dataset, the model correctly predicted 56199 non-fraudulent transactions and 94 fraudulent transactions. However, it also misclassified 658 non-fraudulent transactions as fraudulent and 11 fraudulent transactions as non-fraudulent.

### Conclusions

The model is able to achieve a good enough ROCAUC score. Though the model was able to correctly classify most of the data points, it did misclassify some of them. The 658 non-fraudulent points misclassified as fraudulent may increase the manual workload of the credit card company but is not a major issue. But the 11 fraudulent transactions that made their way undetected, do pose a serious problem. The algorithm can be further tweaked to obtain an even better precision and recall, or other classification and oversampling methods can also be employed. Random Forests may also be a suitable alternate. Even if this model cannot be completely relied upon, it can be employed as an extra measure of security to flag probable fraudulent transactions.
