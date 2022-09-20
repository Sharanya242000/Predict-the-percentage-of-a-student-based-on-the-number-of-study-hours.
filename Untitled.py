#!/usr/bin/env python
# coding: utf-8

# ## **Linear Regression with Python Scikit Learn**
# In this section we will see how the Python Scikit-Learn library for machine learning can be used to implement regression functions. We will start with simple linear regression involving two variables.
# 
# ### **Simple Linear Regression**
# In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression task as it involves just two variables.

# In[ ]:


# Importing all libraries 
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[31]:


# Reading data 
data=pd.read_csv(r"D:\\Grip_Internship\\student_data_grip_internship_task_1.csv")
data=pd.DataFrame(data)
data=data.dropna()
data


# Let's plot our data points on 2-D graph to eyeball our dataset and see if we can manually find any relationship between the data. 

# In[32]:


data.head()


# In[33]:


# Plotting the distribution of scores
data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# **From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score.**

# In[34]:


data.corr()


# **Preparing the data**|

# The next step is to divide the data into "attributes" (inputs) and "labels" (outputs).

# In[35]:


x = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values  


# Now that we have our attributes and labels, the next step is to split this data into training and test sets. We'll do this by using Scikit-Learn's built-in train_test_split() method:

# In[36]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=0) 


# **Training the Algorithm**

# We have split our data into training and testing sets, and now is finally the time to train our algorithm.

# In[43]:


from sklearn.linear_model import LinearRegression  
lm= LinearRegression() 
lm.fit(x,y)


# In[38]:


# Plotting the regression line
line = lm.coef_*x+lm.intercept_

# Plotting for the test data
plt.scatter(x, y)
plt.plot(x, line);
plt.show()


# ### **Making Predictions**

# Now that we have trained our algorithm, it's time to make some predictions.

# In[48]:


print(X_test) # Testing data - In Hours
y_pred = lm.predict(X_test) # Predicting the scores


# In[40]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# In[41]:


hours =np.array( [9.25]).reshape(-1,1)
own_pred =lm.predict(hours)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# In[42]:


from sklearn import metrics  
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred)) 


# In[ ]:




