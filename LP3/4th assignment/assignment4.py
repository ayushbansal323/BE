from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

P1=[0.1,0.6]
P2=[0.15,0.71]
P3=[0.08,0.9]
P4=[0.16,0.85]
P5=[0.2,0.3]
P6=[0.25,0.5]
P7=[0.24,0.1]
P8=[0.3,0.2]
x = np.array([P1,P2,P3,P4,P5,P6,P7,P8])

kmeans = KMeans(n_clusters=2,init=np.array([P1,P8]), random_state=0,n_init=1).fit(x)

clabel=kmeans.labels_
print(clabel)

print("P6 :"+str(kmeans.predict([P6])))

ccenter=kmeans.cluster_centers_
print(ccenter)

ccolor=['b','r']
for i in range(len(x)):
    plt.scatter(x[i][0],x[i][1],c=ccolor[clabel[i]],label='P'+str(i+1))
plt.scatter(ccenter[0][0],ccenter[0][1],c='b',label='m1',s=200,marker="*")
plt.scatter(ccenter[1][0],ccenter[1][1],c='r',label='m2',s=200,marker="*")
plt.title('Points in the dataset'),
plt.legend(),
plt.show()

print("m1 count: "+str(clabel.tolist().count(1)))
