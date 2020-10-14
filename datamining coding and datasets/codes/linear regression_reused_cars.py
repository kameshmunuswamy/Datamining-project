#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas_profiling as pp

# models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import sklearn.model_selection
from sklearn.model_selection import cross_val_predict as cvp
import sklearn.metrics as metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, confusion_matrix


import warnings
warnings.filterwarnings("ignore")


# # Importing the data 

# In[2]:


valid_part = 0.3
pd.set_option('max_columns',100)


# In[3]:


train0 = pd.read_csv('C:/Users/Kamesh/Desktop/Data mining project dataset/Reused cars/craigslist/vehicles_full.csv')
train0.head(5)


# In[4]:


drop_columns = ['id','url', 'region', 'region_url', 'model', 'title_status', 'vin', 'size', 'image_url', 'description','county','state', 'lat','long']
train0 = train0.drop(columns = drop_columns)


# In[5]:


train0.info()


# In[6]:


train0 = train0.dropna()
train0.head(5)


# In[7]:


#Encoding the categorical values to integer 
numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
categorical_columns = []
features = train0.columns.values.tolist()
for col in features:
    if train0[col].dtype in numerics: continue
    categorical_columns.append(col)
# Encoding categorical features
for col in categorical_columns:
    if col in train0.columns:
        le = LabelEncoder()
        le.fit(list(train0[col].astype(str).values))
        train0[col] = le.transform(list(train0[col].astype(str).values))


# In[8]:


#Converting "year" and "odometer" data to integer data type for better analysis
train0['year'] = (train0['year']-1900).astype(int)
train0['odometer'] = train0['odometer'].astype(int)


# In[9]:


train0.head(10)


# In[10]:


train0.info()


# # EDA - Exploratory data analysis

# In[11]:


train0['price'].value_counts()


# In[12]:


#Restricting the target column 'price' and 'odometer' range based on the analysis undertaken
train0 = train0[train0['price'] > 1000]
train0 = train0[train0['price'] < 40000]
# Rounded ['odometer'] to 5000
train0['odometer'] = train0['odometer'] // 5000
train0 = train0[train0['year'] > 110]


# In[13]:


train0.info()


# Correlation matrix

# In[14]:


train0.corr()


# In[15]:


import seaborn as sns
sns.distplot(train0['price']);
plt.xticks(rotation=90);
plt.title("Normal distribution of dependent variable - Price")


# In[16]:


train0.describe()


# In[17]:


pp.ProfileReport(train0)


# # Preparing to modeling 

# In[18]:


target_name = 'price'
train_target0 = train0[target_name]
train0 = train0.drop([target_name], axis=1)


# In[19]:


# Synthesis test0 from train0
train0, test0, train_target0, test_target0 = train_test_split(train0, train_target0, test_size=0.2, random_state=0)


# In[20]:


#For models from Sklearn
scaler = StandardScaler()
train0 = pd.DataFrame(scaler.fit_transform(train0), columns = train0.columns)


# In[21]:


train0.head(3)


# In[22]:


len(train0)


# In[23]:


# Synthesis valid as test for selection models
train, test, target, target_test = train_test_split(train0, train_target0, test_size=valid_part, random_state=0)


# # Building model and tuning test for all features

# In[24]:


acc_train_r2 = []
acc_test_r2 = []
acc_train_d = []
acc_test_d = []
acc_train_rmse = []
acc_test_rmse = []


# In[25]:


def acc_d(y_meas, y_pred):
    # Relative error between predicted y_pred and measured y_meas values
    return mean_absolute_error(y_meas, y_pred)*len(y_meas)/sum(abs(y_meas))

def acc_rmse(y_meas, y_pred):
    # RMSE between predicted y_pred and measured y_meas values
    return (mean_squared_error(y_meas, y_pred))**0.5


# In[26]:




def acc_model(num,model,train,test):
    # Calculation of accuracy of model Sklearn by different metrics   
  
    global acc_train_r2, acc_test_r2, acc_train_d, acc_test_d, acc_train_rmse, acc_test_rmse
    
    ytrain = model.predict(train)  
    ytest = model.predict(test)

    print('target = ', target[:5].values)
    print('ytrain = ', ytrain[:5])

    acc_train_r2_num = round(r2_score(target, ytrain) * 100, 2)
    print('acc(r2_score) for train =', acc_train_r2_num)   
    acc_train_r2.insert(num, acc_train_r2_num)

    acc_train_d_num = round(acc_d(target, ytrain) * 100, 2)
    print('acc(relative error) for train =', acc_train_d_num)   
    acc_train_d.insert(num, acc_train_d_num)
    
    acc_train_rmse_num = round(acc_rmse(target, ytrain) * 100, 2)
    print('acc(rmse) for train =', acc_train_rmse_num)   
    acc_train_rmse.insert(num, acc_train_rmse_num)

    print('target_test =', target_test[:5].values)
    print('ytest =', ytest[:5])
    
    acc_test_r2_num = round(r2_score(target_test, ytest) * 100, 2)
    print('acc(r2_score) for test =', acc_test_r2_num)
    acc_test_r2.insert(num, acc_test_r2_num)
    
    acc_test_d_num = round(acc_d(target_test, ytest) * 100, 2)
    print('acc(relative error) for test =', acc_test_d_num)
    acc_test_d.insert(num, acc_test_d_num)
    
    acc_test_rmse_num = round(acc_rmse(target_test, ytest) * 100, 2)
    print('acc(rmse) for test =', acc_test_rmse_num)
    acc_test_rmse.insert(num, acc_test_rmse_num)
    
    


# # Linear regession

# In[27]:


# Linear Regression

linreg = LinearRegression()
linreg.fit(train, target)
acc_model(0,linreg,train,test)


# In[ ]:





# # Random forest

# In[28]:


# Random Forest

#random_forest = GridSearchCV(estimator=RandomForestRegressor(), param_grid={'n_estimators': [100, 1000]}, cv=5)
random_forest = RandomForestRegressor()
random_forest.fit(train, target)
#print(random_forest.best_params_)
acc_model(6,random_forest,train,test)


# In[ ]:




