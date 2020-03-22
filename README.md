NYC Transit Clustering plus Recommendation Engine
-
- Data Source: 2018 Citywide Mobility Survey of New York City residents' travel choices and behaviors (NYC Open Data)

Goal
-
- Create clusters of types of trips taken throughout New York City
- Compare trips within clusters to find a viable mass transit alternative for trips completed by car
- Motivation: If another person has completed a trip similar to your car trip using mass transit instead, you might consider doing the same!

Data Processing
-
- Remove unrelated features and features with sparse data available
- Impute missing values using mode for categorical data, median for numerical data
- Normalize numerical features
- One-hot-encode categorical features

EDA
-
- Most trips were taken by walking or by car
<img src = "images/count_by_mode.png"> 

- The purpose of most trips is commuting, followed by social/recreation, then shopping
<img src = "images/count_by_purpose.png"> 

- There was no extreme variation in transit mode between days of the week 
<img src = "images/count_day_by_mode.png"> 

- Breakdown of transportation mode frequency by borough
<img src = "images/count_borough_by_mode.png"> 


Cluster
-
- Elbow Plot
<img src = "images/elbow_plot.png"> 

- PCA 4 components
<img src = "images/pca_4.png">

-PCA 42
<img src = "images/pca_42.png">


- Clusters with mode of transit identified
<img src = "images/clusters_with_modes.png"> 

Explore Cluster 1
-
- Precipitation in clusters
<img src = "images/count_precipitation_by_cluster.png"> 

- Cluster 1 mode breakdown
<img src = "images/cluster1_count_precipitation_mode.png"> 

Recommendation Engine
-
