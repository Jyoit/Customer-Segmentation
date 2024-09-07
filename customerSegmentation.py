# Import required libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

# Streamlit app title
st.title('Customer Segmentation using K-Means Clustering')

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load dataset
    dataset = pd.read_csv(uploaded_file)
    st.write("Dataset Overview")
    st.write(dataset.head())
    
    # Dataset Information
    st.write("Dataset Info")
    st.write(dataset.info())
    
    # Check for missing values
    st.write("Missing Values in Dataset:")
    st.write(dataset.isnull().sum())

    # Selecting features for clustering (Annual Income and Spending Score)
    x = dataset.iloc[:, 3:5].values

    # Elbow Method to find optimal number of clusters
    WCSS = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
        kmeans.fit(x)
        WCSS.append(kmeans.inertia_)

    # Plot the Elbow Graph
    st.write("Elbow Point Graph:")
    fig, ax = plt.subplots()
    sns.set()
    ax.plot(range(1, 11), WCSS)
    ax.set_title("The Elbow Point Graph")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("WCSS")
    st.pyplot(fig)

    # Using KMeans with 5 clusters
    st.write("KMeans Clustering with 5 Clusters")
    kmeans = KMeans(n_clusters=5, init="k-means++", random_state=0)
    y_kmeans = kmeans.fit_predict(x)

    # Visualize the clusters
    st.write("Visualizing the Clusters")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s=50, c="green", label="Cluster 1")
    ax.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s=50, c="red", label="Cluster 2")
    ax.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s=50, c="yellow", label="Cluster 3")
    ax.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s=50, c="violet", label="Cluster 4")
    ax.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s=50, c="blue", label="Cluster 5")
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c="cyan", label="Centroids")
    ax.set_title("Customer Groups")
    ax.set_xlabel("Annual Income")
    ax.set_ylabel("Spending Score")
    ax.legend()
    st.pyplot(fig)

    # Cluster feature analysis
    dataset['Cluster'] = y_kmeans
    st.write("Cluster Analysis (Mean of features for each cluster):")
    st.write(dataset.groupby('Cluster').mean())

    # DBSCAN Clustering
    st.write("DBSCAN Clustering")
    dbscan = DBSCAN(eps=0.5, min_samples=6)
    label_dbscan = dbscan.fit_predict(x)

    # Visualizing DBSCAN Clustering
    fig, ax = plt.subplots()
    ax.scatter(x[:, 0], x[:, 1], c=label_dbscan, cmap='viridis', marker='o')
    ax.set_title("DBSCAN Clustering")
    ax.set_xlabel("Annual Income")
    ax.set_ylabel("Spending Score")
    st.pyplot(fig)
