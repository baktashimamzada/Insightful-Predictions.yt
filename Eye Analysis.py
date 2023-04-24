#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir('C:\\Users\\LENOVO\\Downloads')


# In[2]:


import pandas as pd
#Loading the data
df = pd.read_csv('Eye.csv')
#Printing first 10 rows
df.head(10)


# In[3]:


# List of column names to be removed
cols_to_remove = ['ID', 'Recording timestamp', 'Recording monitor latency','Computer timestamp', 'Project name', 'Export date', 
                  'Participant name', 'Recording name', 'Recording date', 'Recording date UTC', 
                  'Recording start time', 'Recording start time UTC', 'Recording duration', 
                  'Timeline name', 'Recording Fixation filter name', 'Recording software version', 
                  'Recording resolution height', 'Recording resolution width', 'Eyetracker timestamp', 
                  'Event', 'Event value', 'Gaze direction left X', 'Gaze direction left Y', 
                  'Gaze direction left Z', 'Gaze direction right X', 'Gaze direction right Y', 
                  'Gaze direction right Z', 'Pupil diameter left', 'Pupil diameter right', 
                  'Eye position left X (DACSmm)', 'Eye position left Y (DACSmm)', 'Eye position left Z (DACSmm)', 
                  'Eye position right X (DACSmm)', 'Eye position right Y (DACSmm)', 'Eye position right Z (DACSmm)', 
                  'Gaze point left X (DACSmm)', 'Gaze point left Y (DACSmm)', 'Gaze point right X (DACSmm)', 
                  'Gaze point right Y (DACSmm)', 'Gaze point X (MCSnorm)', 'Gaze point Y (MCSnorm)', 
                  'Gaze point left X (MCSnorm)', 'Gaze point left Y (MCSnorm)', 'Gaze point right X (MCSnorm)', 
                  'Gaze point right Y (MCSnorm)', 'Presented Stimulus name', 'Presented Media name', 
                  'Fixation point X (MCSnorm)', 'Fixation point Y (MCSnorm)', 'Mouse position X', 'Mouse position Y']

# Remove the specified columns
df = df.drop(columns=cols_to_remove)


# In[4]:


df.head()


# In[5]:


# Replace "Eye Tracker" with 0 and "Mouse" with 1
df['Sensor'] = df['Sensor'].replace({'Eye Tracker': 0, 'Mouse': 1})
# Replace missing values with the mode value of the column
df['Sensor'] = df['Sensor'].fillna(df['Sensor'].mode()[0])
df.head()


# In[6]:


# Compute the mean values of each column
gaze_point_x_mean = df['Gaze point X'].mean()
gaze_point_y_mean = df['Gaze point Y'].mean()
gaze_point_left_x_mean = df['Gaze point left X'].mean()
gaze_point_left_y_mean = df['Gaze point left Y'].mean()
gaze_point_right_x_mean = df['Gaze point right X'].mean()
gaze_point_right_y_mean = df['Gaze point right Y'].mean()

# Fill the missing values with the mean values
df['Gaze point X'] = df['Gaze point X'].fillna(gaze_point_x_mean)
df['Gaze point Y'] = df['Gaze point Y'].fillna(gaze_point_y_mean)
df['Gaze point left X'] = df['Gaze point left X'].fillna(gaze_point_left_x_mean)
df['Gaze point left Y'] = df['Gaze point left Y'].fillna(gaze_point_left_y_mean)
df['Gaze point right X'] = df['Gaze point right X'].fillna(gaze_point_right_x_mean)
df['Gaze point right Y'] = df['Gaze point right Y'].fillna(gaze_point_right_y_mean)
df.head()


# In[7]:


# Replace "Valid" with 0 and "Invalid" with 1
df['Validity left'] = df['Validity left'].replace({'Valid': 0, 'Invalid': 1})
df['Validity right'] = df['Validity right'].replace({'Valid': 0, 'Invalid': 1})
# Replace missing values with the mode value of the respective column
df['Validity left'] = df['Validity left'].fillna(df['Validity left'].mode()[0])
df['Validity right'] = df['Validity right'].fillna(df['Validity right'].mode()[0])
df.head()


# In[8]:


# Replace the values in the "Eye movement type" column
df['Eye movement type'] = df['Eye movement type'].replace({'Fixation': 0, 'Saccade': 1, 'Unclassified': 2, 'EyesNotFound': 3})
# Replace missing values with the mode value of the column
df['Eye movement type'] = df['Eye movement type'].fillna(df['Eye movement type'].mode()[0])
df.head()


# In[9]:


# Replace missing values with the mode value of the respective column
df['Presented Media width'] = df['Presented Media width'].fillna(df['Presented Media width'].mode()[0])
df['Presented Media height'] = df['Presented Media height'].fillna(df['Presented Media height'].mode()[0])
df['Presented Media position X (DACSpx)'] = df['Presented Media position X (DACSpx)'].fillna(df['Presented Media position X (DACSpx)'].mode()[0])
df['Presented Media position Y (DACSpx)'] = df['Presented Media position Y (DACSpx)'].fillna(df['Presented Media position Y (DACSpx)'].mode()[0])
df['Original Media width'] = df['Original Media width'].fillna(df['Original Media width'].mode()[0])
df['Original Media height'] = df['Original Media height'].fillna(df['Original Media height'].mode()[0])
df['Gaze event duration'] = df['Gaze event duration'].fillna(df['Gaze event duration'].mode()[0])
df['Eye movement type index'] = df['Eye movement type index'].fillna(df['Eye movement type index'].mode()[0])
df['Fixation point X'] = df['Fixation point X'].fillna(df['Fixation point X'].mode()[0])
df['Fixation point Y'] = df['Fixation point Y'].fillna(df['Fixation point Y'].mode()[0])
df.head()


# In[12]:


import matplotlib.pyplot as plt
plt.scatter(df['Gaze point X'], df['Gaze point Y'])
plt.xlabel('Gaze point X')
plt.ylabel('Gaze point Y')
plt.title('Scatter plot of Gaze point X vs Gaze point Y')
plt.show()


# In[13]:


plt.hist(df['Validity left'])
plt.xlabel('Validity left')
plt.ylabel('Count')
plt.title('Histogram of Validity left')
plt.show()


# In[16]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# split into features and target
X = df.drop(['Gaze point right Y'], axis=1)
y = df['Gaze point right Y']

# split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# create a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# fit the model to the training data
model.fit(X_train, y_train)

# predict on the testing data
y_pred = model.predict(X_test)

# calculate the R^2 score on the testing data
test_score = r2_score(y_test, y_pred)

# calculate the R^2 score on the training data
train_score = r2_score(y_train, model.predict(X_train))

print("Training R^2 score:", train_score)
print("Testing R^2 score:", test_score)

