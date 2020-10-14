#!/usr/bin/env python
# coding: utf-8

# In[76]:


# Libaries import
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from copy import copy
import warnings
warnings.filterwarnings("ignore")


# # Importing the data

# In[2]:


online_news_data = pd.read_csv("C:/Users/Kamesh/Desktop/Data mining project dataset/OnlineNewsPopularity/OnlineNewsPopularity.csv")
online_news_data.head(n=5)


# In[3]:


origianl_data_copy = copy(online_news_data)
online_news_data.columns


# # EDA

# In[4]:


#Dropping url and timedelta non-preditive attributes. 
online_news_data.drop(labels=['url', ' timedelta'], axis = 1, inplace=True)
online_news_data.head(n=5)


# In[68]:


# Describing the data
online_news_data.describe()


# In[6]:


#Inorder to build a classification model 
# creating a grading criteria for the shares
share_value = online_news_data[' shares']
online_news_data[' shares'].describe()


# Grading the shares values based on the precentage below
# Exceptional = Top 95%
# Excellent = Top 90%
# Very Good = Top 80%
# Good = Top 60%
# Average = Top 50%
# Poor = Top 35%
# Very Poor = Rest

# In[7]:


# creating label grades for classes
share_data_label= list()
for share in share_value:
    if share <= 645:
        share_data_label.append('Very Poor')
    elif share > 645 and share <= 861:
        share_data_label.append('Poor')
    elif share > 861 and share <= 1400:
        share_data_label.append('Average')
    elif share > 1400 and share <= 31300:
        share_data_label.append('Good')
    elif share > 31300 and share <= 53700:
        share_data_label.append('Very Good')
    elif share > 53700 and share <= 77200:
        share_data_label.append('Excellent')
    else:
        share_data_label.append('Exceptional')

# Update this class label into the dataframe
online_news_data = pd.concat([online_news_data, pd.DataFrame(share_data_label, columns=['popularity'])], axis=1)
online_news_data.head(5)


# In[8]:


# Merging the weekdays columns into one single column
weekdays_merge=online_news_data[[' weekday_is_monday',' weekday_is_tuesday',' weekday_is_wednesday', 
                      ' weekday_is_thursday', ' weekday_is_friday',' weekday_is_saturday' ,' weekday_is_sunday' ]]
temp_weekday=[]
for r in list(range(weekdays_merge.shape[0])):
    for c in list(range(weekdays_merge.shape[1])):
        if ((c==0) and (weekdays_merge.iloc[r,c])==1):
            temp_weekday.append('Monday')
        elif ((c==1) and (weekdays_merge.iloc[r,c])==1):
            temp_weekday.append('Tueday')
        elif ((c==2) and (weekdays_merge.iloc[r,c])==1):
            temp_weekday.append('Wednesday')
        elif ((c==3) and (weekdays_merge.iloc[r,c])==1):
            temp_weekday.append('Thursday')
        elif ((c==4) and (weekdays_merge.iloc[r,c])==1):
            temp_weekday.append('Friday')
        elif ((c==5) and (weekdays_merge.iloc[r,c])==1):
            temp_weekday.append('Saturday') 
        elif ((c==6) and (weekdays_merge.iloc[r,c])==1):
            temp_weekday.append('Sunday')
            


# In[9]:


# Merging the data Columns into one single column
Data_col_Merge=online_news_data[[' data_channel_is_lifestyle',' data_channel_is_entertainment' ,' data_channel_is_bus',
                        ' data_channel_is_socmed' ,' data_channel_is_tech',' data_channel_is_world' ]]
#logic to merge data channel
Data_col_arr=[]
for r in list(range(Data_col_Merge.shape[0])):
    if (((Data_col_Merge.iloc[r,0])==0) and ((Data_col_Merge.iloc[r,1])==0) and ((Data_col_Merge.iloc[r,2])==0) and ((Data_col_Merge.iloc[r,3])==0) and ((Data_col_Merge.iloc[r,4])==0) and ((Data_col_Merge.iloc[r,5])==0)):
        Data_col_arr.append('Others')
    for c in list(range(Data_col_Merge.shape[1])):
        if ((c==0) and (Data_col_Merge.iloc[r,c])==1):
            Data_col_arr.append('Lifestyle')
        elif ((c==1) and (Data_col_Merge.iloc[r,c])==1):
            Data_col_arr.append('Entertainment')
        elif ((c==2) and (Data_col_Merge.iloc[r,c])==1):
            Data_col_arr.append('Business')
        elif ((c==3) and (Data_col_Merge.iloc[r,c])==1):
            Data_col_arr.append('Social Media')
        elif ((c==4) and (Data_col_Merge.iloc[r,c])==1):
            Data_col_arr.append('Tech')
        elif ((c==5) and (Data_col_Merge.iloc[r,c])==1):
            Data_col_arr.append('World')


# In[10]:


# merging the new columns into the dataframe
online_news_data.insert(loc=11, column='weekdays', value=temp_weekday)
online_news_data.insert(loc=12, column='data_channel', value=Data_col_arr)

# dropping the old columns from the data
online_news_data.drop(labels=[' data_channel_is_lifestyle',' data_channel_is_entertainment' ,' data_channel_is_bus',
                        ' data_channel_is_socmed' ,' data_channel_is_tech',' data_channel_is_world', 
                 ' weekday_is_monday',' weekday_is_tuesday',' weekday_is_wednesday', 
                      ' weekday_is_thursday', ' weekday_is_friday',' weekday_is_saturday' ,' weekday_is_sunday'], axis = 1, inplace=True)
print(online_news_data.shape)
online_news_data.head(n=5)


# In[11]:


online_news_data.columns


# # Data clensing by removing noise values(zeros)

# In[12]:


#Removing noise data from the data
#Analysis on "n_non_stop_words" column
print(online_news_data[' n_non_stop_words'].describe())


# In[13]:


# this column contains lots of zeros. This record is classifed as a noise therefore it can be removed.
# Here, we will go ahead and drop the field of ' n_non_stop_words'
online_news_data.drop(labels=[' n_non_stop_words'], axis = 1, inplace=True)


# In[14]:


online_news_data.columns


# In[15]:


# removing noise from n_tokens_content. ignoring zeros
online_news_data  = online_news_data[online_news_data[' n_tokens_content'] != 0]
print ("After noise removal - ",online_news_data.shape)


# # Buliding the classification models - KNN and Random forest

# In[16]:


# shares column data is not required for classification model
ignore_shares = online_news_data.drop(labels=[' shares'], axis = 1, inplace=False)


# In[17]:


training_data_set = ignore_shares.iloc[:, :(ignore_shares.shape[1]-1)]
# convert categorical variables into dummy - it use one-hot encoding


# In[18]:


# converting the categorical variables into dummy
training_data_set = pd.get_dummies(training_data_set)

# extract the label data in this case popularity
label_data_set = ignore_shares.iloc[:, (ignore_shares.shape[1]-1):].values

data_feature = copy(training_data_set)


# In[19]:


# only the best observed features are extracted here
data_feature =training_data_set[[' n_tokens_title',' n_tokens_content',' n_unique_tokens',' num_hrefs',
                       ' num_self_hrefs',' num_imgs',' num_videos',' average_token_length',' num_keywords',
                       ' kw_avg_avg',' self_reference_avg_sharess',' global_subjectivity',
                       ' global_sentiment_polarity',' global_rate_positive_words',' global_rate_negative_words',' avg_positive_polarity',
                       ' avg_negative_polarity',' title_sentiment_polarity','weekdays_Friday', 'weekdays_Monday', 'weekdays_Saturday',
       'weekdays_Sunday', 'weekdays_Thursday', 'weekdays_Tueday',
       'weekdays_Wednesday', 'data_channel_Business',
       'data_channel_Entertainment', 'data_channel_Lifestyle',
       'data_channel_Others', 'data_channel_Social Media', 'data_channel_Tech',
       'data_channel_World']]


# In[20]:


# use log transformation to transform each features to a normal distribution
training_set_normal = copy(training_data_set)

# note log transformation can only be performed on data without zero value
for col in training_set_normal.columns:
    #applying log transformation
    temp = training_set_normal[training_set_normal[col] == 0]
    # only apply to non-zero features
    if temp.shape[0] == 0:
        training_set_normal[col] = np.log(training_set_normal[col])
        print (col)
    else:
        # attempt to only transform the positive values alone
        training_set_normal.loc[training_set_normal[col] > 0, col] = np.log(training_set_normal.loc[training_set_normal[col] > 0, col])


# In[21]:


# only the best observed features are extracted here
data_feature1_normal =training_set_normal[[' n_tokens_title',' n_tokens_content',' n_unique_tokens',' num_hrefs',
                       ' num_self_hrefs',' num_imgs',' num_videos',' average_token_length',' num_keywords',
                       ' kw_avg_avg',' self_reference_avg_sharess',' global_subjectivity',
                       ' global_sentiment_polarity',' global_rate_positive_words',' global_rate_negative_words',' avg_positive_polarity',
                       ' avg_negative_polarity',' title_sentiment_polarity','weekdays_Friday', 'weekdays_Monday', 'weekdays_Saturday',
       'weekdays_Sunday', 'weekdays_Thursday', 'weekdays_Tueday',
       'weekdays_Wednesday', 'data_channel_Business',
       'data_channel_Entertainment', 'data_channel_Lifestyle',
       'data_channel_Others', 'data_channel_Social Media', 'data_channel_Tech',
       'data_channel_World']]


# In[ ]:





# In[22]:


# normalizaling the data using standard scaler 

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

#applying scaler to normalize
data_feature_normalize = scaler.fit_transform(data_feature.values)


# In[23]:


online_news_data.head(5)


# In[24]:


# encoding the label set with a label encoder
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(online_news_data.loc[:, 'popularity'].values)
class_names = label_encoder.classes_
class_names 


# In[69]:


#Splitting the data for Training and Testing the model
from sklearn.model_selection import train_test_split, GridSearchCV

x_train, x_test, y_train, y_test = train_test_split(data_feature_normalize, encoded_labels, test_size=0.3, shuffle=False)


# # KNN - Classification model

# In[ ]:


# fitting the model for k=7
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, make_scorer 

neighor = KNeighborsClassifier(n_neighbors =7)

neighor.fit(X_train, y_train)  

# predicting the target
y_predict = neighor.predict(X_test)
accuary_knn=(100*accuracy_score(y_predict, y_test))


# In[27]:


accuary_knn


# In[28]:


# fitting the model for k=7
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, make_scorer 

neighor = KNeighborsClassifier(n_neighbors =7)

neighor.fit(X_train, y_train)  

# predicting the target
y_predict = neighor.predict(X_test)
accuary_knn=(100*accuracy_score(y_predict, y_test))


# In[29]:


accuracy_knn


# # Random Forest - Classification model

# In[30]:


from sklearn.ensemble import RandomForestClassifier

nns = [1, 5, 10, 50, 100, 200, 500, 1000, 2000, 3000]
accuracy_rf = []

for n in nns:    
    clf = RandomForestClassifier(n_estimators=n, n_jobs=5, max_depth=50,
                                 random_state=0)
    clf.fit(X_train, y_train)  

# predict the result
y_pred = clf.predict(X_test)
accuracy_rf.append(100*accuracy_score(y_predict, y_test))
print(accuracy_rf)


# In[31]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1, max_depth=50,
                             random_state=0)
clf.fit(X_train, y_train)  

# predict the result
y_pred = clf.predict(X_test)
print ("Random Forest Classifer Result")
print ("Performance - " + str(100*accuracy_score(y_pred, y_test)) + "%")


# In[77]:


from sklearn.metrics import accuracy_score, classification_report
print('Accuracy on Test Data: %2.2f%%' % (accuracy_score(y_test, y_pred)))
print(classification_report(y_test, y_pred))


# In[33]:


# function for confusion matrix

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print(cm)

    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.32f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout(pad=5, rect= (0, 0, 1, 1))
    return ax


# In[57]:


# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names,
                      title='Confusion matrix For Random Forest')


# In[46]:





# In[58]:





# In[ ]:





# In[ ]:





# In[ ]:




