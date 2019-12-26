import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

x = np.array([[10],[9],[2],[15],[10],[16],[11]])
y = np.array([95,80,10,50,45,98,38])

reg = LinearRegression().fit(x, y)

print("coef :"+str(reg.coef_))

print("intercept"+str(reg.intercept_))

print("predict :"+str(reg.predict(np.array([[16]]))))

print("predict :"+str(reg.coef_*16+reg.intercept_))

plt.scatter(x,y,color='red')
plt.plot(x,reg.predict(x),color='blue')
plt.title('Linear Regression')
plt.xlabel('Number of hrs spent driving')
plt.ylabel('Risk score')
plt.show()
