#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns 
import warnings
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# In[2]:


hospital_details=pd.read_csv("Hospitalisation details.csv")


# In[3]:


medical_exam=pd.read_csv("Medical Examinations.csv")


# In[4]:


names=pd.read_excel("Names.xlsx")


# In[5]:


hospital_details.head()


# In[6]:


medical_exam.head()


# In[8]:


names.head()


# In[9]:


names.shape


# In[10]:


medical_exam.shape


# In[11]:


hospital_details.shape


# # Project Task:  Week 1

# ### 1. Collate the files so that all the information is in one place
# 

# In[12]:


# To merge the data frames we can do it on Customer ID column as they are common in all
# We will have to merge names and medical exam first then the hospital details because as we can see the hospital details DF is in reverse order so we will have to reverse the rows first them merge it


# In[13]:


new_data=pd.merge(hospital_details,medical_exam, on='Customer ID',how='outer')


# In[14]:


new_data


# In[15]:


## now we can merge the hospital details DF with the other DF
final_hospital_df=pd.merge(new_data,names,on='Customer ID',how='outer')


# In[16]:


final_hospital_df


# In[17]:


final_hospital_df=final_hospital_df.loc[::-1]


# In[18]:


final_hospital_df


# In[19]:


final_hospital_df=final_hospital_df.reset_index(drop=True)


# In[20]:


final_hospital_df


# #### 2. Check for missing values in the dataset
# 

# In[21]:


final_hospital_df.isnull().sum()


# In[22]:


final_hospital_df.isnull().sum()/len(final_hospital_df)*100


# In[23]:


# We can see that 8 columns have missing values in the dataset which is even lesser than 1% so we can remove the missing values 
# as we cannot properly fill in missing values for health section as it will be wrongly diagnosed


# In[24]:


final_hospital_df.shape


# In[25]:


final_hospital_df.dropna(inplace=True)


# In[26]:


final_hospital_df.isnull().sum()


# In[27]:


final_hospital_df.shape


# #### 3. Find the percentage of rows that have trivial value (for example, ?), and delete such rows if they do not contain significant information

# In[28]:


trivial_value= final_hospital_df[final_hospital_df.eq("?").any(1)]
trivial_value


# In[29]:


round(trivial_value.shape[0]/final_hospital_df.shape[0]*100,2)


# In[30]:


# There is total 0.43% of rows contain the trivial values.
# Now lets drop the all row that contain the trivial values in the data set


# In[32]:


final_hospital_df.drop(final_hospital_df[final_hospital_df.eq("?").any(1)].index, axis=0, inplace=True)


# In[33]:


final_hospital_df.shape


# #### 4. Use the necessary transformation methods to deal with the nominal and ordinal categorical variables in the dataset

# In[34]:


df_categorical=final_hospital_df.select_dtypes(exclude='number')


# In[35]:


df_categorical.head()


# In[36]:


final_hospital_df['month'].value_counts()


# In[37]:


final_hospital_df['Hospital tier'].value_counts()


# In[38]:


final_hospital_df['City tier'].value_counts()


# In[39]:


final_hospital_df['State ID'].value_counts()


# In[40]:


final_hospital_df['Heart Issues'].value_counts()


# In[41]:


final_hospital_df['Any Transplants'].value_counts()


# In[42]:


final_hospital_df['Cancer history'].value_counts()


# In[43]:


final_hospital_df['NumberOfMajorSurgeries'].value_counts()


# In[44]:


final_hospital_df['smoker'].value_counts()


# In[45]:


df_categorical.head()


# In[46]:


# We have some categorical values so first of all we will transform then by using the label encoder.


# In[47]:


from sklearn.preprocessing import LabelEncoder


# In[48]:


le=LabelEncoder()
final_hospital_df["Heart Issues"] =le.fit_transform(final_hospital_df["Heart Issues"])
final_hospital_df["Any Transplants"] =le.fit_transform(final_hospital_df["Any Transplants"])
final_hospital_df["Cancer history"] =le.fit_transform(final_hospital_df["Cancer history"])
final_hospital_df["smoker"] =le.fit_transform(final_hospital_df["smoker"])


# In[49]:


final_hospital_df['smoker'].value_counts()


# In[50]:


final_hospital_df['month'].unique()


# In[51]:


# As we saw above that there are 7 months in the months column we will map them in the a numerical sequence
final_hospital_df['month']=final_hospital_df['month'].map({'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12})


# In[52]:


final_hospital_df['month'].unique()


# In[53]:


# Now we will transform the ordinal variables: Hospital tier and City tier
def ordinal_variable(val):
    return int(val.replace("tier", "").replace(" ", "").replace("-", ""))
final_hospital_df['Hospital tier']=final_hospital_df['Hospital tier'].map(ordinal_variable)
final_hospital_df['City tier']=final_hospital_df['City tier'].map(ordinal_variable)


# In[54]:


final_hospital_df


# ##### 5. The dataset has State ID, which has around 16 states. All states are not represented in equal proportions in the data. Creating dummy variables for all regions may also result in too many insignificant predictors. Nevertheless, only R1011, R1012, and R1013 are worth investigating further. Create a suitable strategy to create dummy variables with these restraints.

# In[55]:


final_hospital_df['State ID'].value_counts()/len(final_hospital_df)*100


# In[56]:


# As we can see only R1011, R1012 AND R1013 are worth investigating further


# In[57]:


df_copy=final_hospital_df[final_hospital_df['State ID'].isin(['R1011','R1012','R1013'])]


# In[58]:


df_copy.shape


# In[59]:


df_copy['State ID']=le.fit_transform(df_copy['State ID'])


# In[60]:


df_copy['State ID'].unique()


# #### 6. The variable NumberOfMajorSurgeries also appears to have string values. Apply a suitable method to clean up this variable.
# 

# In[61]:


df_copy['NumberOfMajorSurgeries'].value_counts()


# In[62]:


# The Number Of Major Surgeries variable contain string value 'no major Surgery' that means0 surgery. So we will replace this 
# value into int value equal to zero


# In[64]:


df_copy['NumberOfMajorSurgeries'].replace('No major surgery',0,inplace=True)


# In[65]:


df_copy['NumberOfMajorSurgeries'].value_counts()


# #### 7. Age appears to be a significant factor in this analysis. Calculate the patients' ages based on their dates of birth.

# In[66]:


from datetime import datetime


# In[67]:


df_copy['date'].dtypes


# In[68]:


# we will create a column DOB by concatenating year month and date columns
df_copy['DOB']=df_copy['year']+' '+df_copy['month'].astype(str)+' '+df_copy['date'].astype(str)


# In[69]:


df_copy['DOB']=pd.to_datetime(df_copy['DOB'])


# In[70]:


df_copy['Age']=(datetime.now()-df_copy['DOB']).apply(lambda x:x.days/365).astype(int)


# In[71]:


# we will drop the DOB column as it will be dificult to compute 1-1-1 value for modeling 
df_copy.drop(columns=['DOB'],axis=1,inplace=True)


# In[72]:


df_copy


# #### 8.The gender of the patient may be an important factor in determining the cost of hospitalization. The salutations in a beneficiary's name can be used to determine their gender. Make a new field for the beneficiary's gender.
# 

# The salutation (Ms.) denote the female and (Mr.) denote the male. The gender plays an important role to predict the hospitalization cost so for model building we directly denote the gender by int values as Male = 0 & Female = 1

# In[73]:


df_copy['Gender']=['0' if 'Mr.' in name else '1' for name in df_copy['name']]


# In[74]:


df_copy.head()


# #### 9. You should also visualize the distribution of costs using a histogram, box and whisker plot, and swarm plot.

# In[75]:


# Histogram
plt.figure(figsize=(10,5))
sns.histplot(df_copy['charges'])


# In[76]:


# Box Plot / Whisker Plot
plt.figure(figsize=(15,5))
sns.boxplot(df_copy['charges'])


# In[77]:


# Swarm Plot
plt.figure(figsize=(10,5))
sns.swarmplot(df_copy['charges'])


# #### 10. State how the distribution is different across gender and tiers of hospitals

# In[78]:


sns.countplot(x='Hospital tier',hue='Gender',data=df_copy)
plt.xlabel('Hospital tier')
plt.ylabel('Count')
plt.title("Distribution of data by Hospital tier and Gender")


# #### 11. Create a radar chart to showcase the median hospitalization cost for each tier of hospitals

# In[79]:


pd.set_option('display.max_rows',None)
df_copy[['charges','Hospital tier']]


# In[80]:


df_copy['charges'].unique()


# #### 12. Create a frequency table and a stacked bar chart to visualize the count of people in the different tiers of cities and hospitals

# In[81]:


## To make the frequency table we will use pd.crosstab 
freq_table=pd.crosstab(df_copy['Hospital tier'],'Count Of People')
freq_table


# In[82]:


# we will noe create the stacked bar chart
sns.histplot(df_copy,x='City tier' ,hue ='Hospital tier',multiple='stack')


# #### 13. Test the following null hypotheses:
# ##### a. The average hospitalization costs for the three types of hospitals are not significantly different
# ##### b. The average hospitalization costs for the three types of cities are not significantly different
# ##### c. The average hospitalization cost for smokers is not significantly different from the average cost for nonsmokers
# ##### d. Smoking and heart issues are independent

# a.The average hospitalization costs for the three types of hospitals are not significantly different

# In[83]:


import scipy.stats as stats


# In[84]:


print('Null Hypothesis => Average hospitalization costs for the three types of hospitals are not significantly different.')


# In[85]:


# Perform ANOVA test using the `charges` column and grouping by the `Hospital tier` column
f_val, p_val = stats.f_oneway(df_copy[df_copy['Hospital tier'] =='tier - 1']['charges'],df_copy[df_copy['Hospital tier'] =='tier - 2']['charges'],df_copy[df_copy['Hospital tier'] =='tier - 3']['charges'])


# In[86]:


# Print the p-value
print('P-value :',p_val)


# In[87]:


# Compare p-value with significance value(0.05)
if p_val < 0.05:
    print("Reject null hypothesis")
else:
    print("Accept null hypothesis")


# b.The average hospitalization costs for the three types of cities are not significantly different

# In[88]:


# Perform ANOVA test using the `charges` column and grouping by the `City tier` column
f_val, p_val = stats.f_oneway(df_copy[df_copy['City tier'] == 'tier - 1']['charges'], df_copy[df_copy['City tier'] == 'tier - 2']['charges'],df_copy[df_copy['City tier'] == 'tier - 3']['charges'])


# In[89]:


# Print the p-value
print('P-value :',p_val)


# In[90]:


# Compare p-value with significance value(0.05)
if p_val < 0.05:
    print("Reject null hypothesis")
else:
    print("Accept null hypothesis")


# c.The average hospitalization cost for smokers is not significantly different from the average cost for nonsmokers.

# In[91]:


print('Null Hypothesis => Average hospitalization cost for smokers is not significantly different from the average cost for nonsmokers.')


# In[94]:


# Perform ANOVA test using the `charges` column and grouping by the `smoker` column
t_val, p_val = stats.ttest_ind(df_copy[df_copy['smoker'] == 'yes']['charges'],df_copy[df_copy['smoker'] == 'No']['charges'])


# In[95]:


# Print the p-value
print('P-value :',p_val)


# In[96]:


# Compare p-value with significance value(0.05)
if p_val < 0.05:
    print("Reject null hypothesis")
else:
    print("Accept null hypothesis")


# d.Smoking and heart issues are independent

# In[99]:


from scipy.stats import chi2_contingency


# In[100]:


# create a contingency table of the observed counts
contingency_table = pd.crosstab(df_copy['smoker'], df_copy['Heart Issues'])


# In[101]:


# conduct the chi-squared test
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f'P-value = {p}')


# In[102]:


# interpret the p-value
if p < 0.05:
    print("Reject the null hypothesis, Smoking and heart issues are independent.")
else:
    print("Accept null hypothesis, Smoking and heart issues are independent.")


# ### Project Task: Week 2
# 
# ### Machine Learning

# #### 1. Examine the correlation between predictors to identify highly correlated predictors. Use a heatmap to visualize this.

# In[103]:


df_copy.columns


# In[104]:


df_copy=df_copy[['Customer ID','name','Age', 'Gender' ,'year', 'month', 'date', 'children','BMI', 'HBA1C','Heart Issues', 'Any Transplants', 'Cancer history','NumberOfMajorSurgeries', 'smoker','Hospital tier', 'City tier', 'State ID', 'charges' ]]


# In[105]:


df_copy.shape


# In[106]:


df_copy.drop(["Customer ID",'name'],inplace=True,axis=1)


# In[109]:


plt.figure(figsize=(12,12))
sns.heatmap(df_copy.corr(),square=True,annot=True,linewidths=1,cmap='Reds')


# #### 2. Develop and evaluate the final model using regression with a stochastic gradient descent optimizer. Also, ensure that you apply all the following suggestions:

# ##### Note: 
# ##### • Perform the stratified 5-fold cross-validation technique for model building and validation
# ##### • Use standardization and hyperparameter tuning effectively
# ##### • Use sklearn-pipelines
# ##### • Use appropriate regularization techniques to address the bias-variance trade-off
# ##### a. Create five folds in the data, and introduce a variable to identify the folds
# ##### b. For each fold, run a for loop and ensure that 80 percent of the data is used to train the model and the remaining 20 percent is used to validate it in each iteration
# ##### c. Develop five distinct models and five distinct validation scores (root mean squared error values)
# ##### d. Determine the variable importance scores, and identify the redundant variables
# 

# In[110]:


# lets first seperate the input and output data.
x = df_copy.drop(["charges",'year','month','date'], axis=1)
y = df_copy[['charges']]


# In[111]:


# Lets split the data set into the training and testing data.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, 
test_size=.20, random_state=10)


# In[112]:


# Now standardize the data.
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


# In[113]:


x_train = sc.fit_transform(x_train)


# In[114]:


x_test = sc.fit_transform(x_test)


# In[115]:


from sklearn.linear_model import SGDRegressor


# In[116]:


from sklearn.model_selection import GridSearchCV


# In[117]:


params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2,0.3,0.4,0.5, 0.6,0.7,0.8,0.9,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,20,50,100,500,1000],'penalty': ['l2', 'l1', 'elasticnet']}


# In[118]:


sgd = SGDRegressor()


# In[119]:


# Cross Validation 
folds = 5
model_cv = GridSearchCV(estimator = sgd,
                        param_grid = params,
                        scoring = 'neg_mean_absolute_error',
                        cv = folds,
                        return_train_score = True,
                        verbose = 1)
model_cv.fit(x_train,y_train)


# In[120]:


GridSearchCV(cv=5, estimator=SGDRegressor(),
 param_grid={'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 
                       0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0,4.0, 5.0, 
                       6.0, 7.0, 8.0, 9.0, 10.0,20, 50,100, 500, 1000],
             'penalty': ['l2', 'l1', 'elasticnet']},
             return_train_score=True,
             scoring='neg_mean_absolute_error',
             verbose=1)


# In[121]:


model_cv.best_params_


# In[122]:


sgd = SGDRegressor(alpha= 100, penalty= 'l1')


# In[123]:


sgd.fit(x_train, y_train)


# In[124]:


SGDRegressor(alpha=100, penalty='l1')


# In[125]:


sgd.score(x_test, y_test)


# In[126]:


y_pred = sgd.predict(x_test)


# In[127]:


from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[128]:


sgd_mae = mean_absolute_error(y_test, y_pred)
sgd_mse = mean_squared_error(y_test, y_pred)
sgd_rmse = sgd_mse*(1/2.0)


# In[129]:


print("MAE:", sgd_mae)
print("MSE:", sgd_mse)
print("RMSE:", sgd_rmse)


# In[130]:


importance = sgd.coef_


# In[131]:


pd.DataFrame(importance, index = x.columns, columns=['Feature_imp'])


# #### 3. Use random forest and extreme gradient boosting for cost prediction, share your crossvalidation results, and calculate the variable importance scores

# In[132]:


# Random Forest
from sklearn.ensemble import RandomForestRegressor


# In[133]:


# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)


# In[134]:


# Train the model on training data
rf.fit(x_train, y_train)


# In[135]:


score = rf.score(x_test,y_test)
score


# In[136]:


y_pred = rf.predict(x_test)


# In[137]:


rf_mae = mean_absolute_error(y_test, y_pred)


# In[138]:


rf_mae


# In[139]:


# Extreme Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor


# In[140]:


# Instantiate model with 1000 decision trees
gbr = GradientBoostingRegressor(n_estimators = 1000, random_state =42)


# In[141]:


# Train the model on training data
gbr.fit(x_train, y_train)


# In[142]:


score = gbr.score(x_test,y_test)
score


# In[143]:


y_pred = gbr.predict(x_test)


# In[144]:


gbr_mae = mean_absolute_error(y_test, y_pred)
gbr_mae


# #### 4. Case scenario:
# #### Estimate the cost of hospitalization for Christopher, Ms. Jayna (her date of birth is 12/28/1988, height is 170 cm, and weight is 85 kgs). She lives in a tier-1 city and her state’s State ID is R1011. She lives with her partner and two children. She was found to be nondiabetic (HbA1c = 5.8). She smokes but is otherwise healthy. She has had no transplants or major surgeries. Her father died of lung cancer. Hospitalization costs will be estimated using tier-1 hospitals.
# 

# In[145]:


import datetime as dt


# In[146]:


# First we need to calculate the age of the person.
date = str(19881228)
date1 = pd.to_datetime(date, format = "%Y%m%d")


# In[147]:


current_date = dt.datetime.now()
current_date


# In[148]:


datetime.datetime(2023, 3, 10, 8, 44, 17, 188860)


# In[149]:


age = (current_date - date1)
age


# In[150]:


age = int(12421/365)
age


# In[151]:


# now with the help of height and weight we will calculate the BMI.
height_m = 170/100
height_sq = height_m*height_m
BMI = 85/height_sq
np.round(BMI,2)


# In[152]:


# Now lets gen
list = [[2,1,1,24.41,5.8,0,0,0,0,1,1,34,0]]


# In[153]:


dfj = pd.DataFrame(list, columns = ['children', 'Hospital tier', 'City tier', 'BMI', 'HBA1C','Heart Issues', 'Any Transplants', 'Cancer history','NumberOfMajorSurgeries', 'smoker', 'State_ID', 'age', 'gender'] )
dfj


# #### 5.Find the predicted hospitalization cost using all five models. The predicted value should be the mean of the five models' predicted values.

# In[155]:


Hospital_cost = []


# In[156]:


# Now lets predict the hospitalization cost through SGDRegressor
Cost1 = sgd.predict(dfj)
Hospital_cost.append(Cost1)


# In[157]:


# Now lets predict the hospitalization cost through Random Forest
Cost2 = rf.predict(dfj)
Hospital_cost.append(Cost2)


# In[159]:


# Now lets predict the hospitalization cost throug Extreme gradient Booster
Cost3 = gbr.predict(dfj)
Hospital_cost.append(Cost3)


# In[160]:


avg_cost = np.mean(Hospital_cost)
avg_cost


# The average cost for Ms. Jenya's hospital bills will be: 64,462.62
