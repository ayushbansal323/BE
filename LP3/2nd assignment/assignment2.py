from sklearn import tree
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import pandas
from os import system

data = pandas.read_csv("dataset.csv",index_col="id")

encoder=LabelEncoder()
data["age"]=encoder.fit_transform(data["age"])
data["income"]=encoder.fit_transform(data["income"])
data["gender"]=encoder.fit_transform(data["gender"])
data["Marital Status"]=encoder.fit_transform(data["Marital Status"])

X_d=data[["age","income","gender","Marital Status"]]
X=X_d[:14]
Y=data[["Buys"]]
Y=Y[:14]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

print("pridict :"+str(clf.predict([X_d.iloc[14]])))

dotfile = tree.export_graphviz(clf, out_file = "Dtree.dot", feature_names = X.columns)
system("dot -Tpng *.dot -o dtree2.png")
