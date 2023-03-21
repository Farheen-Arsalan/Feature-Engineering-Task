# load the face dataset 
# save all the images as stack in one numpy array 
# one array for X (whole dataset format)
# one array for the labelling (target)
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.image as mimg
from sklearn import metrics,svm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import decomposition
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
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
        
        # plt.figure(1)
        # plt.imshow(im,cmap = 'gray')
        # plt.title('User Number: '+str(i+1))
        # plt.axis('off')
        # plt.pause(0.3) # 0.3 second
Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.2,random_state=10)
print(Xtrain.shape,ytrain.shape)
print(Xtest.shape,ytest.shape)
svmModel = svm.SVC(kernel='rbf',C=2)
svmModel = svmModel.fit(Xtrain,ytrain)
op  = svmModel.predict(Xtest)
acc = metrics.accuracy_score(ytest,op)
print('Accuracy: ',acc)

report = metrics.classification_report(ytest,op)

print('Accuracy without DR: ',acc)
print('Complete Report:')
print(report)

pre = metrics.precision_score(ytest, op,average="macro")
print('precision without DR :' , pre)


rc = metrics.recall_score(ytest, op,average="macro")
print('recall without DR : ' , rc)

f1 = metrics.f1_score(ytest, op,average="macro")
print('f1_score without DR : ' , f1)


# reduce the dimensions pca

# reduce the dimensions pca

pcaModel = decomposition.PCA(n_components=4,random_state=1)

Xupdated = pcaModel.fit_transform(X)
print(Xupdated.shape)

Xtrain,Xtest,ytrain,ytest = train_test_split(Xupdated,y,test_size=0.2,random_state=10)

svmModel = svm.SVC(kernel = 'poly')

svmModel = svmModel.fit(Xtrain,ytrain) 

op =svmModel.predict(Xtest)


conf = metrics.confusion_matrix(ytest,op)
print('Confusion Matrix')
print(conf)


acc = metrics.accuracy_score(ytest,op)
report = metrics.classification_report(ytest,op)

print('Accuracy with PCA : ',acc)
print('Complete Report:')
print(report)

pre1 = metrics.precision_score(ytest, op,average="macro")
print('precision with PCA :' , pre1)


rc1 = metrics.recall_score(ytest, op,average="macro")
print('recall with PCA :' , rc1)

f2 = metrics.f1_score(ytest, op,average="macro")
print('f1_score without PCA : ' , f2)

# ICA 
ICAModel = decomposition.FastICA(n_components=4,random_state=1)
Xupdated1 = pcaModel.fit_transform(X)
print(Xupdated1.shape)

Xtrain,Xtest,ytrain,ytest = train_test_split(Xupdated,y,test_size=0.2,random_state=10)

svmModel = svm.SVC(kernel = 'poly')

svmModel = svmModel.fit(Xtrain,ytrain) 

op =svmModel.predict(Xtest)

conf = metrics.confusion_matrix(ytest,op)
print('Confusion Matrix')
print(conf)



acc = metrics.accuracy_score(ytest,op)
report = metrics.classification_report(ytest,op)


print('Accuracy with ICA: ',acc)
print('Complete Report:')
print(report)

pre2 = metrics.precision_score(ytest, op,average="macro")
print('precision with ICA :' , pre2)


re2 = metrics.recall_score(ytest, op,average="macro")
print('recall with ICA :' , re2)

f3 = metrics.f1_score(ytest, op,average="macro")
print('f1_score with ICA :' , f3)










