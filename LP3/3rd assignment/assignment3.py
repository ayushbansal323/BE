from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

x = [[2,4], [4,2], [4,4], [4,6], [6,2], [6,4]]
y = [0, 0, 1, 0, 1, 0]
xgraph=[i[0] for i in x]
ygraph=[i[1] for i in x]

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x, y)

pre=neigh.predict([[6,6]])
print('negative' if (pre[0]==0) else 'positive')

precolor = 'ro' if (pre[0]==0) else 'go'
plt.plot(xgraph[:4],ygraph[:4],'ro')
plt.plot(xgraph[4:],ygraph[4:],'bo')
plt.plot([6],[6],precolor)
plt.axis([0, 10, 0, 10])
plt.show()

neigh = KNeighborsClassifier(n_neighbors=3,weights="distance")
neigh.fit(x, y)

pre=neigh.predict([[6,6]])
print('negative' if (pre[0]==0) else 'positive')

precolor = 'ro' if (pre[0]==0) else 'go'
plt.plot(xgraph[:4],ygraph[:4],'ro')
plt.plot(xgraph[4:],ygraph[4:],'bo')
plt.plot([6],[6],precolor)
plt.axis([0, 10, 0, 10])
plt.show()
