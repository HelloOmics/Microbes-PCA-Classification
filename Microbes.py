# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 11:11:28 2025

@author: Hello0mics
"""
# Spyder had difficulties finding kagglehub module even though it was installed
# needed this to know proper location to save kagglehub, so that Spyder could find it
#import sys
#print(sys.executable) 
# use path given from this command for next command in command prompt
# /path/to/spyder/python -m pip install kagglehub


import kagglehub
kagglehub.login()  # Authenticate if needed
# kagglehub.dataset_download("dataset-owner/dataset-name", path="./data")

# To browse Datasets via command prompt/bash:
#kaggle datasets list -s KEYWORD : #list datasets matching a search term
#kaggle datasets download -d [DATASET]: #download files associated with a dataset

#-----------------------------------------------
# To browse Datasets using bash via python script
# need to use subprocess
import subprocess

# Run the kaggle command
result = subprocess.run(["kaggle", "datasets", "list", "-s", "MICROBE"], capture_output=True, text=True)
#put in command as shows above, each word individually

# Print the output
print(result.stdout)

#----------------------------------------------

# working directory wasn't set correctly, save .py file in subfolder
# i.e change working directory
import os

# Change the working directory to the desired path
os.chdir(r"C:\Users\roxan\OneDrive\Desktop\Masters_Biology_FU\DIY_data_science\datasets\Microbe_classification")

# Verify the new working directory
print("New Working Directory:", os.getcwd())

#--------------------------------------------
# to downoad dataset using bash via python script
# use subprocess
result2 = subprocess.run(["kaggle", "datasets", "download", "-d", "sayansh001/microbes-dataset"],
                         capture_output=True, text=True)

# Print the output
print(result2.stdout)
# Dataset URL: https://www.kaggle.com/datasets/sayansh001/microbes-dataset
#License(s): CC0-1.0
# downloaded file is now found in working directory

#----------------------------------

# read in the dataset + explore the data

# to read in we need pandas
import pandas as pd

# Load CSV into a DataFrame
df = pd.read_csv(r"C:\Users\roxan\OneDrive\Desktop\Masters_Biology_FU\DIY_data_science\datasets\Microbe_classification\microbes.csv")

# Preview the first few rows
print(df.head())

# what does the dataset contain?
print(df.columns) # shows all columns, what data do we have?

microorganisms = df["microorganisms"]
print(microorganisms)
microorganisms.unique()
# we have labels for 10 different microorganisms
# we can use the other columns to train a model to differentiate between
# 10 microorganisms

# Explore data
print(df.head())  # Displays the first 5 rows
print(df.tail())  # Displays the last 5 rows
print(df.shape)  # Output: (rows, columns) => check dimensions of dataset
print(df.columns) # show columns names = what was measured
print(df.info()) # shows basic info like data types, non-null values and memory usage
print(df.describe()) # get quick statistics for numerical columns
print(df.isnull().sum())  # Sum of missing values per column
#df_cleaned = df.dropna() # Remove rows with any NaN values
#df_cleaned = df.dropna(axis=1) # Remove columns with any NaN values
#print(df['column_name'].unique())  # will give all possible values once
#print(df['column_name'].value_counts()) # Count occurrences of each unique value in a column:
    
# data manipulation:
#df.sort_values(by='column_name', ascending=False) # sorts by highest value
#filtered = df[df['column_name'] > 10]  # [is a list], so a list of a list where condition xyz is fulfilled?

#--------------------------------------------------------------------

# Let's do a PCA to try to differentiate these 10 
# microorganisms

# 1) import necessary packages
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 2) save names for labels later
names = df['microorganisms']

# 3) Make a subset + scale data
# all columns except "microorganisms" are numerical
# so remove microorganisms via drop function
subset_df = df.drop(columns=["microorganisms"])

# Step 3: Standardize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(subset_df)

# Step 4: Apply PCA
pca = PCA(n_components=2) # start simple then add complexity
principal_components = pca.fit_transform(df_scaled)

# Step 5: Create a Dataframe for PCA results
pca_result = pd.DataFrame( data=principal_components, columns =["PC1", "PC2"])
print(pca_result)
pca_result["microorganisms"] = microorganisms
print(pca_result)

# Step 6:  Is model a good fit? How much variance des it describe?
# should we add more complexity and principal components?
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)
# Explained Variance Ratio: [0.34892224 0.26599741]

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
print("Cumulative explained variance:", cumulative_variance)
#Cumulative explained variance: [0.34892224 0.61491965]
# model explains about 61% of the variance in the dataset

#----------------------------------------------------------
# how many principal components would be optimum?
#-----------------------------------------------------

# Step 7 Visualisations: 2 principal components

# 2D Seaborn
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize = (8, 8))
sns.scatterplot(
    x = pca_result['PC1'],
    y = pca_result['PC2'],
    hue = pca_result['microorganisms'],  
    palette = 'magma', 
    edgecolor = 'k',
    alpha = 0.5
)
plt.title('PCA Visualization of Microbes')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title = 'Microbes', loc = 'best')
plt.show()


# 3D interactive plotly
import plotly.express as px

# first thre components of PCA6
fig = px.scatter(
    pca_result, 
    x='PC1', 
    y='PC2',  
    hover_name='microorganisms',  # Use names as labels
    title='Mirobe classification',
    color="microorganisms",
    template="plotly_dark" 

)

# open plot as an html file
fig.write_html("plot_microbes.html")

import webbrowser

# Path to the HTML file
file_path = "plot_microbes.html"

# Open the file in the default web browser
webbrowser.open(file_path)

#---------------------------------------------------------------------
# 3 principal components

# Step 4: Apply PCA
pca3 = PCA(n_components=3) # start simple then add complexity
principal_components3 = pca3.fit_transform(df_scaled)

# Step 5: Create a Dataframe for PCA results
pca_result3 = pd.DataFrame( data=principal_components3, columns =["PC1", "PC2", "PC3"])
print(pca_result3)
pca_result3["microorganisms"] = microorganisms
print(pca_result3)

# Step 6:  Is model a good fit? How much variance des it describe?
# should we add more complexity and principal components?
print("Explained Variance Ratio:", pca3.explained_variance_ratio_)
explained_variance = pca3.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)
# Explained Variance Ratio: [0.34892224 0.26599741 0.08064918]

cumulative_variance = np.cumsum(pca3.explained_variance_ratio_)
print("Cumulative explained variance:", cumulative_variance)
#Cumulative explained variance: [0.34892224 0.61491965 0.69556883]
# model explains about 70% of the variance in the dataset


# Step 7 Visualisations
import plotly.express as px

# first thre components of PCA6
fig3 = px.scatter_3d(
    pca_result3, 
    x='PC1', 
    y='PC2',
    z="PC3",
    hover_name='microorganisms',  # Use names as labels
    title='Mirobe classification',
    color="microorganisms",
    template="plotly_dark" 

)

# open plot as an html file
fig3.write_html("plot_microbes3.html")

import webbrowser

# Path to the HTML file
file_path3 = "plot_microbes3.html"

# Open the file in the default web browser
webbrowser.open(file_path3)

#-----------------------------------------------------------------
# what is the optimum amounts of components?
#------------------------------------------------------------------

# 3 components

cumulative_variance = np.cumsum(pca3.explained_variance_ratio_)
print("Cumulative explained variance:", cumulative_variance)

plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Choosing the Optimal Number of Components')
plt.grid()
plt.show()

# elbow at 2 coomponents, however 3 components also only explains 70% of
# the variance in the data
# Let's add more complexity and then check this graph again

#--------- 

#  6 principal components

# Step 4: Apply PCA
pca6 = PCA(n_components=6) # start simple then add complexity
principal_components6 = pca6.fit_transform(df_scaled)

# Step 5: Create a Dataframe for PCA results
pca_result6 = pd.DataFrame( data=principal_components6, columns =["PC1", "PC2", "PC3", "PC4", "PC5", "PC6"])
print(pca_result6)
pca_result6["microorganisms"] = microorganisms
print(pca_result6)

# Calculate explained variance
explained_variance_ratio6 = pca6.explained_variance_ratio_
cumulative_variance6 = np.cumsum(explained_variance_ratio6)

# Plot cumulative explained variance
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--', color='b')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance Plot')
plt.legend(loc='best')
plt.grid()
plt.show()

# 6 components explain about 85% of the variance in the data

#-------------------------------------------------------------
# How to visualise a PCA with 6 components? 
# plot pairwise or use further dimensionality reduction
# like t-SNE or UMAP
#-------------------------------------------------------------

# UMAP

# Step 1: Apply PCA with 6 components
pca6 = PCA(n_components=6) # start simple then add complexity #already did this uptop
pca_result6 = pca6.fit_transform(df_scaled)

# Step 2: Apply UMAP on the PCA-reduced data
import umap
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42) # create umap
umap_result = umap_model.fit_transform(pca_result6) # fit data to umap

# Step 3: Visualize the UMAP results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(umap_result[:, 0], umap_result[:, 1], c="y", cmap='Spectral', s=10, alpha=0.8)
plt.colorbar(scatter, label='Digit Label')
plt.title('UMAP Visualization on PCA-Reduced Data')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.grid()
plt.show()

# UMAP doesn't make much sense?

#----------------

# Interactive Parallel Coordinates Plot

import plotly.express as px

parallel_data = pd.DataFrame(pca_result6, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6'])
fig6 = px.parallel_coordinates(
    parallel_data,
    color='PC1',  # Choose one component for coloring
    labels={'PC1': 'Principal Component 1', 'PC6': 'Principal Component 6'}
)
fig.show()

# spyder couldn't render

# open plot as an html file
fig6.write_html("plot6.html")

import webbrowser

# Path to the HTML file
file_path6 = "plot6.html"

# Save the plot as a PNG image =>  2d image for Github README
fig6.write_image("plot6.png")

# Open the file in the default web browser
webbrowser.open(file_path6)

#--------------------------

# pairwise scatter plots

import seaborn as sns
import pandas as pd

# Example DataFrame
pca_pairwise6 = pd.DataFrame(pca_result6, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6'])

# Pairwise scatter plot
sns.pairplot(pca_pairwise6)
plt.show()

#------------------















