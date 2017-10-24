from sklearn.tree import DecisionTreeClassifier
features=[[181,80,44],[177,70,43],[160,60,38],[154,54,37],[166,65,40],[190,90,47],[175,64,30],[177,70,40],[159,55,37],[171,75,42],[181,85,43]]
labels=['male','male','female','female','male','male','female','female','female','male','male']

clf=DecisionTreeClassifier()
clf.fit(features,labels)
p=clf.predict([[190,70,43]])
print("prediction",p)
