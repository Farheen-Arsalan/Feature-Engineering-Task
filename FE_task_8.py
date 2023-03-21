
from sklearn import datasets, svm,tree,metrics
from sklearn import ensemble
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.image as mimg
images  = np.zeros((400,112,92)) 
X = np.zeros((400,10304))
y = np.zeros((400))
count=0
for i in range(40):
    for j in range(10):
        path = '/Users/Arsalan/Desktop/orl_faces/orl_faces/u(%d)/%d.png'%(i+1,j+1)
        im = mimg.imread(path)
        feat = im.reshape(1,-1) # -1 means arrange all values on the columns side 
        print(im.shape)
        images[count,:,:]=im # images
        X[count,:]=feat # data
        y[count] = i
        count=count+1

Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=10)

print(Xtrain.shape,Xtest.shape)
print(ytrain.shape,ytest.shape)

# create the decision tree model 
treeModel = tree.DecisionTreeClassifier(criterion='gini',max_depth=10)
treeModel = treeModel.fit(Xtrain,ytrain)

opTrain = treeModel.predict(Xtrain) # passing training data to get
opTest = treeModel.predict(Xtest)
# its predicted values for training accuracy 
trAcc = metrics.accuracy_score(opTrain,ytrain)
print('Training accuracy DT : ',trAcc)


# testing accuracy 
tstAcc = metrics.accuracy_score(ytest,opTest)
print('Testing Accuracy DT : ',tstAcc)


# create the randomforest model 
rfModel = ensemble.RandomForestClassifier(n_estimators=500)
rfModel = rfModel.fit(Xtrain,ytrain)
opTrainrf = rfModel.predict(Xtrain)
opTestrf = rfModel.predict(Xtest)

trrAccRf = metrics.accuracy_score(ytrain,opTrainrf)
tstAccRf = metrics.accuracy_score(ytest,opTestrf)
print('Training Accuracy RF: ',trrAccRf)
print('Testing Accuracy RF: ',tstAccRf)

