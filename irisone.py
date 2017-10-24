
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

iris=load_iris()

features=iris.data
labels=iris.target

from sklearn.cross_validation import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(features,labels,test_size=.3)

clf=DecisionTreeClassifier()
clf.fit(xtrain,ytrain)

p=clf.predict(xtest)

from sklearn.metrics import accuracy_score

print("Accuracy= ",accuracy_score(ytest,p))
#print(iris)
