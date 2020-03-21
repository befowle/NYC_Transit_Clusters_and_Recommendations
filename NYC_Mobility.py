#import required packages

from sodapy import Socrata
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import numpy as np
np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
%matplotlib inline
import mobility_config
from sklearn.decomposition import PCA #Principal Component Analysis
from sklearn.manifold import TSNE #T-Distributed Stochastic Neighbor Embedding
from sklearn.cluster import KMeans #K-Means Clustering
from sklearn.preprocessing import StandardScaler #used for 'Feature Scaling'
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics #used for labeling
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KernelDensity
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist


#Call NYC Open Data API
client = Socrata("data.cityofnewyork.us", '7YJroGSBVCt6gzuLz6whih0yc')

# Example authenticated client (needed for non-public datasets):
client = Socrata('data.cityofnewyork.us',
                  mobility_config.app_token,
                  username=mobility_config.app_user,
                  password=mobility_config.app_pw)

# First 8000 results, returned as JSON from API / converted to Python list of
# dictionaries by sodapy.
trip_diary = client.get("kcey-vehy", limit=8000)
# Convert to pandas DataFrame
trip_diary_df = pd.DataFrame.from_records(trip_diary)

#save as csv
trip_diary_df.to_csv('trip_diary.csv')

#drop irrelevant columns
drop_cols_trip_diary_df = trip_diary_df.drop(columns=['job', 'qday1tripstartcode', 'qday1tripstartcode', 'qday1tripstartam',
                                            'qday1tripstartnoon', 'qday1tripstartpm', 'qday1trip_outsidenyc_start',
                                            'qday1trip_outsidenyc_end', 'qday1triptransitto1', 'qday1triptransitto2',
                                            'qday1triptransitto3', 'qday1triptransitto4', 'qday1triptransitto5',
                                            'qday1triptransitto6', 'qday1triptransitto7', 'qday1triptransitfrom1',
                                            'qday1triptransitfrom2', 'qday1triptransitfrom3', 'qday1triptransitfrom4',
                                            'qday1triptransitfrom5', 'qday1triptransitfrom6', 'qday1triptransitfrom7',
                                            'qday1triptravelcode01', 'qday1triptravelcode02', 'qday1triptravelcode03',
                                            'qday1triptravelcode04', 'qday1triptravelcode05', 'qday1triptravelcode06',
                                            'qday1triptravelcode07', 'qday1triptravelcode08', 'qday1triptravelcode09',
                                            'qday1triptravelcode10', 'qday1triptravelcode11', 'qday1triptravelcode12',
                                            'qday1triptravelcode13', 'qday1triptravelcode14', 'qday1triptravelcode15',
                                            'qday1triptravelcode16', 'qday1triptravelcode17', 'qday1triptravelcode18',
                                            'qday1triptravelcode19', 'qday1triptravelcode20', 'qday1triptravelcode21',
                                            'qday1triptravelcode22', 'qday1triptravelcode23', 'qday1triptravelcode24',
                                            'qday1triptravelcode25', 'qday1triptravelcode26', 'qday1triptravelcode27',
                                            'qday1trippurpose', 'qday1triplength', 'qday1triplength_flt','qday1trippark',
                                            'qday1tripparkpay_amount', 'qday1tripbikestore','qage', 'qrace', 'qhispanic',
                                            'qcarchange', 'qsunrise', 'qsunset', 'qday1tripparkpay'])

#impute missing values
drop_cols_trip_diary_df['qday1typical']= drop_cols_trip_diary_df['qday1typical'].replace('Very Typical', 'Very typical')
drop_cols_trip_diary_df['qday1tripend']= drop_cols_trip_diary_df['qday1tripend'].replace('NA', 'Home')
drop_cols_trip_diary_df['qday1tripendcode']= drop_cols_trip_diary_df['qday1tripendcode'].replace('NA', 'Home')
drop_cols_trip_diary_df['qincome']= drop_cols_trip_diary_df['qincome'].replace('NA', '$50,000 - $74,999')
drop_cols_trip_diary_df['qeducation']= drop_cols_trip_diary_df['qeducation'].replace('9', "Bachelor's degree (i.e., BA, BS, AB)")
drop_cols_trip_diary_df['qlicense']= drop_cols_trip_diary_df['qlicense'].replace('NA', 'Yes')
drop_cols_trip_diary_df['qcaraccess']= drop_cols_trip_diary_df['qcaraccess'].replace('NA', 'I personally own or lease a car')
drop_cols_trip_diary_df['qwelfare2']= drop_cols_trip_diary_df['qwelfare2'].replace('NA', 'Yes')
drop_cols_trip_diary_df['qsmartphone']= drop_cols_trip_diary_df['qsmartphone'].replace('NA', 'Yes')
drop_cols_trip_diary_df['qcitibike']= drop_cols_trip_diary_df['qcitibike'].replace('NA', 'No')
drop_cols_trip_diary_df['qpurposerecode']= drop_cols_trip_diary_df['qpurposerecode'].replace('NA', 'Commute to/from work')

#drop NOT CODED elements
new_df = drop_cols_trip_diary_df[~drop_cols_trip_diary_df.qsurveyzone_home.str.contains('NOT CODED')]

#save to to_csv
new_df.to_csv('edited_trips_df.csv')

#impute values
edited_trips_df['qborough_start'] = np.where(edited_trips_df['qborough_start'] == 2, edited_trips_df['qsurveyzone_home'], edited_trips_df['qborough_start'])
edited_trips_df['surveyzone_start'] = np.where(edited_trips_df['surveyzone_start'] == 2, edited_trips_df['qsurveyzone_home'], edited_trips_df['surveyzone_start'])
edited_trips_df['qborough_end'] = np.where(edited_trips_df['qborough_end'] == 2, edited_trips_df['qborough_home'], edited_trips_df['qborough_end'])
edited_trips_df['qsurveyzone_start'] = np.where(edited_trips_df['qsurveyzone_start'] == 2, edited_trips_df['surveyzone_start'], edited_trips_df['qsurveyzone_start'])
edited_trips_df['qsurveyzone_end'] = np.where(edited_trips_df['qsurveyzone_end'] == 2, edited_trips_df['qsurveyzone_home'], edited_trips_df['qsurveyzone_end'])
edited_trips_df['surveyzone_end'] = np.where(edited_trips_df['surveyzone_end'] == 2, edited_trips_df['qsurveyzone_home'], edited_trips_df['surveyzone_end'])
edited_trips_df['qsurveyzone_end'] = np.where(edited_trips_df['surveyzone_end'] == 'NA', edited_trips_df['qsurveyzone_start'], edited_trips_df['surveyzone_end'])
edited_trips_df['number_of_trips_taken'] = edited_trips_df['number_of_trips_taken'].str.replace(r'\D', '')
edited_trips_df['number_of_trips_taken'] = edited_trips_df['number_of_trips_taken'].astype('float64')
ited_trips_df['qday1typical']= edited_trips_df['qday1typical'].replace(2 , 'Very typical')
edited_trips_df['qeducation']= edited_trips_df['qeducation'].replace(2 , "Bachelor's degree (i.e., BA, BS, AB)")

#save to csv
edited_trips_df.to_csv('edited_trips_df1.csv')

#fill NA
edited_trips_df = edited_trips_df.fillna(edited_trips_df['number_of_trips_taken'].value_counts().index[0])
edited_trips_df['qpurposerecode']= edited_trips_df['qpurposerecode'].replace('NA', 'Commute to/from work')
edited_trips_df = edited_trips_df.drop(columns=['disability'])

#drop NTA columns
edited_trips_df = edited_trips_df.drop(columns=['qntacode_start', 'qntacode_end', 'ntacode_home'])

#save as csv
clean_trip_df = edited_trips_df.to_csv('clean_trip_df.csv')

#import and rename csv
trip_df = pd.read_csv('clean_trip_df.csv', index_col=0)

#create dummy variables
trip_dummies_df = pd.get_dummies(trip_df

#save as csv
trip_dummies_df.to_csv('trip_dummies_df.csv')

#k-means clustering (method from Kaggle)
X = pd.read_csv('trip_dummies_df.csv', index_col=0)

#define numerical columns
numer = X[["number_of_trips_taken", "qtemphigh", "qtemplow", "qprecipitation"]]
IDs = X[["trip_id", "uniqueid"]]

#select desired variables
categ = X[["trip_id", "uniqueid""qday_Friday","qday_Monday","qday_Saturday","qday_Sunday","qday_Thursday","qday_Tuesday",
           "qday_Wednesday","qday1typical_Not at all typical",
           "qday1typical_Not very typical","qday1typical_Somewhat typical",
           "qday1typical_Very typical","qborough_home_Brooklyn","qborough_home_Manhattan",
           "qborough_home_Queens","qborough_home_Staten Island","qborough_home_The Bronx",
           "qsurveyzone_home_Inner Brooklyn","qsurveyzone_home_Inner Queens",
           "qsurveyzone_home_Manhattan Core","qsurveyzone_home_Middle Queens",
           "qsurveyzone_home_Northern Bronx","qsurveyzone_home_Northern Manhattan",
           "qsurveyzone_home_Outer Brooklyn","qsurveyzone_home_Outer Queens",
           "qsurveyzone_home_Southern Bronx", "qsurveyzone_home_Staten Island",
           "qtripdaytime_AM","qtripdaytime_NOON","qtripdaytime_PM","qborough_start_Brooklyn",
           "qborough_start_Inner Brooklyn",
           "qborough_start_Inner Queens","qborough_start_Manhattan","qborough_start_Manhattan Core",
           "qborough_start_Middle Queens","qborough_start_Northern Bronx","qborough_start_Northern Manhattan",
           "qborough_start_Outer Brooklyn","qborough_start_Outer Queens","qborough_start_Outside of NYC","qborough_start_Queens",
           "qborough_start_Southern Bronx","qborough_start_Staten Island",
           "qborough_start_The Bronx","surveyzone_start_Inner Brooklyn","surveyzone_start_Inner Queens",
           "surveyzone_start_Manhattan Core","surveyzone_start_Middle Queens","surveyzone_start_NOT CODED","surveyzone_start_Northern Bronx",
           "surveyzone_start_Northern Manhattan","surveyzone_start_Outer Brooklyn","surveyzone_start_Outer Queens",
           "surveyzone_start_Southern Bronx","surveyzone_start_Staten Island","qday1tripend_Child's daycare facility or school",
           "qday1tripend_Doctor's office or hospital","qday1tripend_Entertainment event (i.e. sporting event, play, etc.)",
           "qday1tripend_Friend or family member's home","qday1tripend_Grocery store or market (including deli, bodega, etc.)","qday1tripend_Home",
           "qday1tripend_Other","qday1tripend_Outside of New York City","qday1tripend_Park/Recreational area/Gym","qday1tripend_Restaurant or bar",
           "qday1tripend_Retail store (e.g. clothing, electronic, hardware, etc.)",
           "qday1tripend_School","qday1tripend_Work","qborough_end_Brooklyn",
           "qborough_end_Manhattan","qborough_end_Outside of NYC","qborough_end_Queens",
           "qborough_end_Staten Island","qborough_end_The Bronx",
           "surveyzone_end_Inner Brooklyn","surveyzone_end_Inner Queens",
           "surveyzone_end_Manhattan Core","surveyzone_end_Middle Queens",
           "surveyzone_end_Northern Bronx","surveyzone_end_Northern Manhattan",
           "surveyzone_end_Outer Brooklyn","surveyzone_end_Outer Queens",
           "surveyzone_end_Southern Bronx","surveyzone_end_Staten Island","qday1tripendcode_Airport",
           "qday1tripendcode_Bus stop","qday1tripendcode_Child's daycare facility or school",
           "qday1tripendcode_Commuter rail station","qday1tripendcode_Doctor's office or hospital",
           "qday1tripendcode_Entertainment event (i.e. sporting event, play, etc.)",
           "qday1tripendcode_Friend or family member's home","qday1tripendcode_Grand Central Station",
           "qday1tripendcode_Grocery store or market (including deli, bodega, etc.)",
           "qday1tripendcode_Home","qday1tripendcode_Other","qday1tripendcode_PATH Station","qday1tripendcode_Park and ride/parking lot",
           "qday1tripendcode_Park/Recreational area/Gym","qday1tripendcode_Penn Station",
           "qday1tripendcode_Restaurant or bar","qday1tripendcode_Retail store (e.g. clothing, electronic, hardware, etc.)",
           "qday1tripendcode_Road/tunnel/bridge","qday1tripendcode_School","qday1tripendcode_Work",
           "qday1triplength_cat_0 to 5","qday1triplength_cat_11 to 15","qday1triplength_cat_16 to 20",
           "qday1triplength_cat_180+","qday1triplength_cat_21 to 25","qday1triplength_cat_26 to 30",
           "qday1triplength_cat_31 to 35","qday1triplength_cat_36 to 40","qday1triplength_cat_41 to 45",
           "qday1triplength_cat_46 to 50","qday1triplength_cat_51 to 55","qday1triplength_cat_56 to 60",
           "qday1triplength_cat_6 to 10","qday1triplength_cat_61 to 180","qdisability1_No","qdisability1_Yes","qdisability2_No",
           "qdisability2_Yes","qdisability3_No","qdisability3_Yes","qdisability4_No","qdisability4_Yes",
           "qdisability5_No","qdisability5_Yes","qdisability6_No","qdisability6_Yes","qdisability7_No",
           "qdisability7_Yes","qdisability8_No","qdisability8_Yes","qdisability9_No","qdisability9_Yes","qagecode_18-24","qagecode_25-34",
           "qagecode_35-44","qagecode_45-54","qagecode_55-64","qagecode_65 or older",
           "qlicense_No","qlicense_Yes","qcaraccess_I do not have access to a car",
           "qcaraccess_I do not personally own or lease a car, but I have access to a car belonging to a member of my household",
           "qcaraccess_I personally own or lease a car","qcaraccess_Other","qwelfare1_No",
           "qwelfare1_Yes","qwelfare2_No","qwelfare2_Yes","qwelfare3_No","qwelfare3_Yes","qwelfare4_No",
           "qwelfare4_Yes","qwelfare5_No","qwelfare5_Yes","qcitibike_No","qcitibike_Yes","qday1triptravelcode_sp_Car Service",
           "qday1triptravelcode_sp_Carpool","qday1triptravelcode_sp_Carshare","qday1triptravelcode_sp_Citi Bike",
           "qday1triptravelcode_sp_Community van/dollar van","qday1triptravelcode_sp_Commuter rail",
           "qday1triptravelcode_sp_Electric bicycle","qday1triptravelcode_sp_Express bus","qday1triptravelcode_sp_Green taxi",
           "qday1triptravelcode_sp_Local bus","qday1triptravelcode_sp_Motorcycle","qday1triptravelcode_sp_Other",
           "qday1triptravelcode_sp_Other ferry","qday1triptravelcode_sp_PATH train","qday1triptravelcode_sp_Paratransit/ Access-A-Ride",
           "qday1triptravelcode_sp_Personal bicycle","qday1triptravelcode_sp_Personal car",
           "qday1triptravelcode_sp_Ride-hail service such as Uber or Lyft","qday1triptravelcode_sp_Select bus service",
           "qday1triptravelcode_sp_Shared-ride service such a Uber Pool, Via, or Lyft Line",
           "qday1triptravelcode_sp_Staten Island ferry","qday1triptravelcode_sp_Subway","qday1triptravelcode_sp_Walk",
           "qday1triptravelcode_sp_Yellow taxi","qpurposerecode_Accompanying other traveler",
           "qpurposerecode_Business","qpurposerecode_Commute to/from work","qpurposerecode_Dining",
           "qpurposerecode_Medical visit (doctor's office)","qpurposerecode_Other","qpurposerecode_Personal errands",
           "qpurposerecode_School","qpurposerecode_Shopping","qpurposerecode_Social/recreation"]]

#initialize scalar
scaler = StandardScaler()

#scale numerical data
numer = pd.DataFrame(scaler.fit_transform(numer))

#rename columns
numer.columns = ["number_of_trips_taken", "qtemphigh", "qtemplow", "qprecipitation"]

#join dataframes
X = pd.concat([numer, categ], axis=1, join='inner')
#add ID df
X = pd.concat([X, IDs], axis=1, join='inner')

#begin clustering
X = X.drop(columns = ["trip_id", "uniqueid"], axis=1)

#determine ideal number of clusters
# Distortion is the average of the squared distances from the clusters centers of each cluster
distortions = []
# Inertia is the sum of squared distances of samples to their closest cluster center.
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1,10)

for k in K:
    #Building and fitting the model
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)

    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                      'euclidean'),axis=1)) / X.shape[0])
    inertias.append(kmeanModel.inertia_)

    mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                 'euclidean'),axis=1)) / X.shape[0]
    mapping2[k] = kmeanModel.inertia_

for key,val in mapping1.items():
    print(str(key)+' : '+str(val))

plt.plot(K, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('Mean Squared Distance Between Clusters')
plt.show()

#Initialize model
kmeans = KMeans(n_clusters=2)

#fit model
kmeans.fit(X)

#Find which cluster each data-point belongs to
clusters = kmeans.predict(X)

#add cluster vector to X df
X["Cluster"] = clusters

#calculate silhouette score
labels = kmeans.labels_

metrics.silhouette_score(X, labels, metric='euclidean')

#calculate Calinsk-Harabasz score
metrics.calinski_harabasz_score(X, labels)

#add cluster vector to X df
X["Cluster"] = clusters2

#Run Principal Component Analysis
pca = PCA()
pca.fit(X)
PCA(copy=True, n_components=None, whiten=False)
len(pca.components_)
#visualize PCA
pca = PCA().fit(X)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

#Calculate explained variance ratio
print("Explained Variance Ratio = ", sum(pca.explained_variance_ratio_[: 4]))
print("Explained Variance Ratio = ", sum(pca.explained_variance_ratio_[: 42]))

#visualize for 42 principal components (90+% of variance)
pca = PCA(42).fit(X)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

pca42 = PCA(4)
k_means = KMeans(n_clusters = 2)
k_means.fit(X)
y_hat = k_means.predict(X)

pca42 = PCA(4)
k_means = KMeans(n_clusters = 2)
k_means.fit(X)
y_hat = k_means.predict(X)

#begin cluster visualizations
#PCA with one principal component
pca_1d = PCA(n_components=1)
#PCA with two principal components
pca_2d = PCA(n_components=2)
#PCA with three principal components
pca_3d = PCA(n_components=3)

#This DataFrame holds that single principal component mentioned above
PCs_1d = pd.DataFrame(pca_1d.fit_transform(X.drop(["Cluster"], axis=1)))
#This DataFrame contains the two principal components that will be used
#for the 2-D visualization mentioned above
PCs_2d = pd.DataFrame(pca_2d.fit_transform(X.drop(["Cluster"], axis=1)))
#And this DataFrame contains three principal components that will aid us
#in visualizing our clusters in 3-D
PCs_3d = pd.DataFrame(pca_3d.fit_transform(X.drop(["Cluster"], axis=1)))

PCs_1d.columns = ["PC1_1d"]
#"PC1_2d" means: 'The first principal component of the components created for 2-D visualization, by PCA.'
#And "PC2_2d" means: 'The second principal component of the components created for 2-D visualization, by PCA.'
PCs_2d.columns = ["PC1_2d", "PC2_2d"]
PCs_3d.columns = ["PC1_3d", "PC2_3d", "PC3_3d"]

# We concatenate these newly created DataFrames to X so
# that they can be used by X as columns.
X = pd.concat([X,PCs_1d,PCs_2d,PCs_3d], axis=1, join='inner')

# And we create one new column for plotX so that we can use it for
# 1-D visualization.
X["dummy"] = 0

# Each of these new DataFrames will hold all of the values contained
# in exacltly one of the clusters. For example, all of the values
# contained within the DataFrame, cluster0 will belong to 'cluster 0',
# and all the values contained in DataFrame, cluster1 will belong to
# 'cluster 1', etc.
#Note that all of the DataFrames below are sub-DataFrames of 'plotX'.
#This is because we intend to plot the values contained within each of these DataFrames.

cluster0 = X[X["Cluster"] == 0]
cluster1 = X[X["Cluster"] == 1]

#This is needed so we can display plotly plots properly
init_notebook_mode(connected=True)

# Instructions for building the 2-D plot

#trace1 is for 'Cluster 0'
trace1 = go.Scatter(
                    x = cluster0["PC1_2d"],
                    y = cluster0["PC2_2d"],
                    mode = "markers",
                    name = "Cluster 0",
                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                    text = None)

#trace2 is for 'Cluster 1'
trace2 = go.Scatter(
                    x = cluster1["PC1_2d"],
                    y = cluster1["PC2_2d"],
                    mode = "markers",
                    name = "Cluster 1",
                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                    text = None)


data = [trace1, trace2]

title = "Visualizing Clusters in Two Dimensions Using PCA"

layout = dict(title = title,
              xaxis= dict(title= 'PC1',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'PC2',ticklen= 5,zeroline= False)
             )

fig = dict(data = data, layout = layout)

iplot(fig)

#add ID columns back
X = pd.concat([X, IDs], axis=1, join='inner')
X.to_csv('X_w_clusters.csv')

#Instructions for building the 3-D plot

#trace1 is for 'Cluster 0'
trace1 = go.Scatter3d(
                    x = cluster0["PC1_3d"],
                    y = cluster0["PC2_3d"],
                    z = cluster0["PC3_3d"],
                    mode = "markers",
                    name = "Cluster 0",
                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                    text = None)

#trace2 is for 'Cluster 1'
trace2 = go.Scatter3d(
                    x = cluster1["PC1_3d"],
                    y = cluster1["PC2_3d"],
                    z = cluster1["PC3_3d"],
                    mode = "markers",
                    name = "Cluster 1",
                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                    text = None)

data = [trace1, trace2]

title = "Clusters with 3 Principal Components"

layout = dict(title = title,
              xaxis= dict(title= 'PC1',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'PC2',ticklen= 5,zeroline= False)
             )

fig = dict(data = data, layout = layout)

iplot(fig)

#Begin Hierarchical agglomerative Clustering (from https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/)
X = pd.read_csv('scaled_dummy_reduced_df.csv', index_col=0)
X = X.drop(columns = ["trip_id", "uniqueid"], axis=1)

# generate the linkage matrix
Z = linkage(X, 'ward')

c, coph_dists = cophenet(Z, pdist(X))
c
#The cophenetic distance between two observations that have been clustered is defined to be the
#intergroup dissimilarity at which the two observations are first combined into a single cluster.
def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata


fancy_dendrogram(
    Z,
    truncate_mode='lastp',
    p=12,
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    annotate_above=10,  # useful in small plots so annotations don't overlap
)
plt.show()

#select distance cutoff (number of clusters)
# set cut-off to 100
max_d = 100  # max_d as in max_distance
fancy_dendrogram(
    Z,
    truncate_mode='lastp',
    p=4,
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    annotate_above=10,
    max_d=max_d,  # plot a horizontal cut-off line
)
plt.show()

#has clusters, PC eigenvalues for 1-3 dims, rando dummy col
clusters = pd.read_csv('X_w_clusters.csv', index_col=0)

#has full dummied cols
modes = pd.read_csv('categIDs_for_clusters.csv', index_col=0)

PCs = clusters[["PC1_1d", "PC1_2d", "PC2_2d", "PC1_3d", "PC2_3d", "PC3_3d"]]
clusters = clusters.drop(columns = ["PC1_1d", "PC1_2d", "PC2_2d", "PC1_3d", "PC2_3d", "PC3_3d"], axis =1)
clusters.to_csv('clusters_to_join.csv')

#transit modes df
add_me = modes[["trip_id", "uniqueid","qmodegrouping_Bike","qmodegrouping_Bus","qmodegrouping_Car",
            "qmodegrouping_Commuter Rail","qmodegrouping_Ferry","qmodegrouping_For-Hire Vehicle","qmodegrouping_Other",
            "qmodegrouping_Subway","qmodegrouping_Walk","qsustainablemode_No","qsustainablemode_Yes",
]]
add_me.to_csv('modes_to_join.csv')

#begin join
#merge only where ids match
cluster_modes_df = clusters.merge(add_me, on='trip_id', how='inner')
cluster_modes_df=cluster_modes_df.drop(columns = ["uniqueid_x"], axis =1)
cluster_modes_df.to_csv('cluster_modes_df.csv')

#recluster clusters
clustered = pd.read_csv('X_w_clusters.csv', index_col=0)
cluster0 = clustered[clustered['Cluster'] == 0].copy()
cluster1 = clustered[clustered['Cluster'] == 1].copy()

cluster1 = clustered[clustered['Cluster'] == 1].copy()
kmeans.fit(cluster0)
cluster0_clusters = kmeans.predict(cluster0)
cluster0['Cluster2'] = cluster0_clusters

#cluster0 metrics
labels = kmeans.labels_
metrics.silhouette_score(cluster0, labels, metric='euclidean')
metrics.calinski_harabasz_score(cluster0, labels)

#cluster1 kmeans
kmeans.fit(cluster1)
cluster1_clusters = kmeans.predict(cluster1)
cluster1['Cluster2'] = cluster1_clusters
labels = kmeans.labels_
metrics.silhouette_score(cluster1, labels, metric='euclidean')
metrics.calinski_harabasz_score(cluster1, labels)

#cluster 1

# Distortion is the average of the squared distances from the clusters centers of each cluster
distortions = []
# Inertia is the sum of squared distances of samples to their closest cluster center.
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1,10)

for k in K:
    #Building and fitting the model
    kmeanModel = KMeans(n_clusters=k).fit(cluster1)
    kmeanModel.fit(cluster1)

    distortions.append(sum(np.min(cdist(cluster1, kmeanModel.cluster_centers_,
                      'euclidean'),axis=1)) / cluster1.shape[0])
    inertias.append(kmeanModel.inertia_)

    mapping1[k] = sum(np.min(cdist(cluster1, kmeanModel.cluster_centers_,
                 'euclidean'),axis=1)) / cluster1.shape[0]
    mapping2[k] = kmeanModel.inertia_

#cluster1
for key,val in mapping1.items():
    print(str(key)+' : '+str(val))

#elbow plotfor cluster 1
plt.plot(K, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('Mean Squared Distance Between Clusters')
plt.show()

#cluster 0

# Distortion is the average of the squared distances from the clusters centers of each cluster
distortions = []
# Inertia is the sum of squared distances of samples to their closest cluster center.
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1,10)

for k in K:
    #Building and fitting the model
    kmeanModel = KMeans(n_clusters=k).fit(cluster0)
    kmeanModel.fit(cluster0)

    distortions.append(sum(np.min(cdist(cluster0, kmeanModel.cluster_centers_,
                      'euclidean'),axis=1)) / cluster0.shape[0])
    inertias.append(kmeanModel.inertia_)

    mapping1[k] = sum(np.min(cdist(cluster0, kmeanModel.cluster_centers_,
                 'euclidean'),axis=1)) / cluster1.shape[0]
    mapping2[k] = kmeanModel.inertia_

#cluster0
for key,val in mapping1.items():
    print(str(key)+' : '+str(val))

#for cluster 0
plt.plot(K, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('Mean Squared Distance Between Clusters')
plt.show()

#PCA for each cluster
pca_2d = PCA(n_components=2)
PCs_2d = pd.DataFrame(pca_2d.fit_transform(cluster0.drop(["Cluster2"], axis=1)))
PCs_2d.columns = ["PC1_2d", "PC2_2d"]
X = pd.concat([cluster0,PCs_2d], axis=1, join='inner')
X["dummy"] = 0
cluster0_0 = X[X["Cluster2"] == 0]
cluster0_1 = X[X["Cluster2"] == 1]
cluster0_2 = X[X["Cluster2"] == 2]
cluster0_3 = X[X["Cluster2"] == 3]
cluster0_4 = X[X["Cluster2"] == 4]
init_notebook_mode(connected=True)

#trace1 is for 'Cluster 0'
trace1 = go.Scatter(
                    x = cluster0_0["PC1_2d"],
                    y = cluster0_0["PC2_2d"],
                    mode = "markers",
                    name = "Cluster 0",
                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                    text = None)

#trace2 is for 'Cluster 1'
trace2 = go.Scatter(
                    x = cluster0_1["PC1_2d"],
                    y = cluster0_1["PC2_2d"],
                    mode = "markers",
                    name = "Cluster 1",
                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                    text = None)

#trace3 is for 'Cluster 2'
trace3 = go.Scatter(
                    x = cluster0_2["PC1_2d"],
                    y = cluster0_2["PC2_2d"],
                    mode = "markers",
                    name = "Cluster 2",
                    marker = dict(color = 'rgba(0, 255, 200, 0.8)'),
                    text = None)

#trace4 is for 'Cluster 3'
trace4 = go.Scatter(
                    x = cluster0_2["PC1_2d"],
                    y = cluster0_2["PC2_2d"],
                    mode = "markers",
                    name = "Cluster 3",
                    marker = dict(color = 'rgba(0, 255, 225, 0.8)'),
                    text = None)

#trace5 is for 'Cluster 4'
trace5 = go.Scatter(
                    x = cluster0_2["PC1_2d"],
                    y = cluster0_2["PC2_2d"],
                    mode = "markers",
                    name = "Cluster 4",
                    marker = dict(color = 'rgba(255, 200, 200, 0.8)'),
                    text = None)

data = [trace1, trace2, trace3, trace4, trace5] #, trace3

title = "Visualizing Clusters in Two Dimensions Using PCA"

layout = dict(title = title,
              xaxis= dict(title= 'PC1',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'PC2',ticklen= 5,zeroline= False)
             )

fig = dict(data = data, layout = layout)

iplot(fig)

#begin recommendation engine
cluster0 = pd.read_csv('cluster0_df.csv', index_col=0)
cluster1 = pd.read_csv('cluster1_df.csv', index_col=0)
df0 = cluster0[['trip_id', 'PC1_2d', 'PC2_2d', 'qmodegrouping']]
df1 = cluster1[['trip_id', 'PC1_2d', 'PC2_2d', 'qmodegrouping']]
#one hot encode transit modes
dummies0 = pd.get_dummies(df0)
dummies1 = pd.get_dummies(df1)

scaler = MinMaxScaler()
df0_scaled = scaler.fit_transform(dummies0)

# Convert into a DataFrame
scaled_df0 = pd.DataFrame(df0_scaled, columns=dummies0.columns, index = dummies0.trip_id) #maybe not trip_id...
scaled_df0 = scaled_df0.drop(['trip_id'], axis = 1)

scaler = MinMaxScaler()
df0_scaled = scaler.fit_transform(dummies0)

# Convert into a DataFrame
scaled_df0 = pd.DataFrame(df0_scaled, columns=dummies0.columns, index = dummies0.trip_id) #maybe not trip_id...
scaled_df0 = scaled_df0.drop(['trip_id'], axis = 1)

df1_scaled = scaler.fit_transform(dummies1)

# Convert into a DataFrame
scaled_df1 = pd.DataFrame(df1_scaled, columns=dummies1.columns, index = dummies1.trip_id) #maybe not trip_id...
scaled_df1 = scaled_df1.drop(['trip_id'], axis = 1)

cos_sim_scaled0 = cosine_similarity(scaled_df0)
cos_sim_scaled1 = cosine_similarity(scaled_df1)

indices0=pd.Series(scaled_df0.index)
indices1=pd.Series(scaled_df1.index)

# creating a Series for the trip_id so they are associated to an ordered numerical
# list I will use in the function to match the id
indices0 = pd.Series(scaled_df0.index)
#  defining the function that takes in movie title
# as input and returns the top 10 recommended movies
def recommendations0(trip_id, cos_sim = cos_sim_scaled0):

    # initializing the empty list of recommended movies
    rec_items = []

    # gettin the index of the movie that matches the title
    idx = indices0[indices0 == trip_id].index[0]
    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cos_sim[idx]).sort_values(ascending = False)
    # getting the indexes of the 10 most similar movies
    top_10_indexes = list(score_series.iloc[1:11].index)

    # populating the list with the titles of the best 10 matching movies
    for i in top_10_indexes:
        rec_items.append(list(scaled_df0.index)[i])

    return rec_items

recommendations0(55202675)

df0.ix[(df0.trip_id =='53201637').idxmin(), 'qmodegrouping']

indices1 = pd.Series(scaled_df1.index)
#  defining the function that takes in movie title
# as input and returns the top 10 recommended movies
def recommendations1(trip_id, cos_sim = cos_sim_scaled1):

    # initializing the empty list of recommended movies
    rec_items = []

    # gettin the index of the movie that matches the title
    idx = indices1[indices1 == trip_id].index[0]
    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cos_sim[idx]).sort_values(ascending = False)
    # getting the indexes of the 10 most similar movies
    top_10_indexes = list(score_series.iloc[1:11].index)

    # populating the list with the titles of the best 10 matching movies
    for i in top_10_indexes:
        rec_items.append(list(scaled_df1.index)[i])

    return

recommendations1(53202334)
df1.ix[(df1.trip_id =='54201972').idxmin(), 'qmodegrouping']
