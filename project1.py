#imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

#loading the data
data=pd.read_csv("Iris.csv")

x=data.iloc[:,1:5].values
print(data.info())

#data visualization
sns.set_style("whitegrid")
sns.pairplot(data.iloc[:,1:],hue="Species",size=3)
plt.show()

#Using the elbow method to find the optimal number of clusters for k-means clustering

wcss=[]

for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init="k-means++",random_state=123)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_) #inertia_ --> wcss 
    
plt.plot(range(1,11),wcss)
plt.title("The elbow method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

#k=3 

kmeans=KMeans(n_clusters=3,init="k-means++")
y_predict=kmeans.fit_predict(x)

#visualising the clusters
plt.scatter(x[y_predict==0,0],x[y_predict==0,1],s=100,c="blue",label="Iris-setosa")
plt.scatter(x[y_predict==1,0],x[y_predict==1,1],s=100,c="red",label="Iris-versicolour")
plt.scatter(x[y_predict==2,0],x[y_predict==2,1],s=100,c="green",label="Iris-virginica")

#plotting the centroids of the clusters

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c="black",label="Centroids")
plt.legend()
plt.title("Clusters and Centroids")
plt.xlabel("SepaLengthCm")
plt.ylabel("SepalWidthCm")
plt.show()

# 3d scatterplot using matplotlib

fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(111, projection='3d')
plt.scatter(x[y_predict == 0, 0], x[y_predict == 0, 1], s = 100, c = "blue", label = 'Iris-setosa')
plt.scatter(x[y_predict == 1, 0], x[y_predict == 1, 1], s = 100, c = "red", label = 'Iris-versicolour')
plt.scatter(x[y_predict == 2, 0], x[y_predict == 2, 1], s = 100, c = "green", label = 'Iris-virginica')

#Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 200, c = "black", label = 'Centroids',marker="x",linewidths=5)
plt.show()



