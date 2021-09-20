#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


# In[36]:


iris = datasets.load_iris()


# In[39]:


iris


# In[3]:


#use dataframe to create table which consist of 3 principles (data, rows, and columns)
df = pd.DataFrame({
    "x" : iris.data[:,0],
    "y" : iris.data[:,1],
    "cluster" : iris.target
                  })


# In[4]:


df.head() # sepal.length is x and sepal.width is y


# In[5]:


#create a blank dictionary called "centroids" then use loop fuction and fixed the range i=3 means I will get value of i=0.1 and 2
#taken array as "result_list" and put the values of x and y
#to determine the values of x and y, first determine all the mean values of x and y
centroids = {}
for i in range(3):
    result_list = []
    result_list.append(df.loc[df["cluster"] == i]["x"].mean())
    result_list.append(df.loc[df["cluster"] == i]["y"].mean())
    
    centroids[i] = result_list


# In[6]:


centroids #check mean of all three cluster


# In[7]:


#plot "x" and "y" the main points with color
fig = plt.figure(figsize = (5, 5))
plt.scatter(df["x"], df["y"], c = iris.target)
plt.xlabel("Speal Length", fontsize = 18)
plt.ylabel("Sepal Width", fontsize = 18)


# In[8]:


#Lets plot the centroid values created for loop function
colmap = {0: "r", 1: "g", 2: "b"}
for i in range(3) :
         plt.scatter(centroids[i][0], centroids[i][1], color = colmap[i])
plt.show()


# In[28]:


#Lest put it together
fig = plt.figure(figsize = (5, 5))
plt.scatter(df["x"], df["y"], c = iris.target, alpha = 0.3)
colmap = {0: "r", 1: "g", 2: "b"}
col = [0,1]
for i in centroids.keys() :
    plt.scatter(centroids[i][0], centroids[i][1], c = colmap[i], edgecolor = "k")
    
    plt.show()


# In[16]:


#calculating distance and update DataFrame
#Take new function which named "assignment" pass through the existing dataframe
def assignment(df, centroids) :
    for i in range(3) : 
        #sqrt((x1 - x2)^2 + (y1 - y2)^2)
        df["distance_from_{}".format(i)] = (
            np.sqrt(
                (df["x"] - centroids[i][0]) ** 2
                + (df["y"] - centroids[i][1]) ** 2
           )
        )
        
    centroids_distance_cols = ["distance_from_{}" .format(i) for i in centroids.keys()]
    df["closest"] = df.loc[:, centroids_distance_cols].idxmin(axis = 1)
    df["closest"] = df["closest"].map(lambda x: int(x.lstrip("distance_from_")))
    df["color"] = df["closest"].map(lambda x: colmap[x])
    return df


# In[17]:


df = assignment(df, centroids)


# In[18]:


df #update dataframe


# In[27]:


#plot the cluster with updated dataframe
fig = plt.figure(figsize = (5, 5))
plt.scatter(df["x"], df["y"], color = df["color"], alpha = 0.3)
for i in centroids.keys():
    plt.scatter(*centroids[i], color = colmap[i], edgecolor = "k")
    
    plt.show()


# In[20]:


#Basis on updated values lets plot updated centroids
# create new function named "update" and use "closet" instead of "cluster" then pass exstinf centroids to the new centroids
def update(k) :
    for i in range(3) : 
        centroids[i][0] = np.mean(df[df["closest"] == i]["x"])
        centroids[i][1] = np.mean(df[df["closest"] == i]["y"])
        return k


# In[21]:


centroids = update(centroids)
centroids


# In[26]:


#Visualizing graph with updated centroids
fig = plt.figure(figsize = (5, 5))
plt.scatter(df["x"], df["y"], color = df["color"], alpha = 0.3)
for i in centroids.keys() :
    plt.scatter(*centroids[1], color = colmap[i], edgecolor ="k")
    
    plt.show()


# In[29]:


#continue till all assigned clusters dont change anymore
while True:
    closest_centroids = df["closest"].copy(deep = True)
    centroids = update(centroids)
    df = assignment(df, centroids)
    if closest_centroids.equals(df["closest"]) : 
        break  # for loop can be exited altogether with the break keyword


# In[34]:


fig = plt.figure(figsize = (5, 5))
plt.scatter(df["x"], df["y"], color = df["color"])
for i in centroids.keys() : 
    plt.scatter(centroids[i][0], centroids[i][1], color  = colmap[i], edgecolor = "k")
    
    plt.show()


# In[ ]:


# All the three classess are separated into their own cluster
# K-means clustering using sklearn  

