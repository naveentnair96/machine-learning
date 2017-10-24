
#knn_score = knn_classifier.score(X_test, y_test)
from  sklearn.neighbors import KNeighborsClassifier

from sklearn.datasets import load_iris

iris=load_iris()

features=iris.data
labels=iris.target

from sklearn.cross_validation import train_test_split

(xtrain,xtest,ytrain,ytest)=train_test_split(features,labels,test_size=.3)

knn_classifier = KNeighborsClassifier()
knn_classifier.fit(xtrain,ytrain)
knn_score = knn_classifier.score(xtest, ytest)


print(knn_score)
#print(iris)

