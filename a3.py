import pickle
import numpy as np
import numpy.random as rnd
import sklearn.discriminant_analysis as da
import matplotlib.pyplot as plt
import sklearn.utils as uti
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import warnings
from sklearn.decomposition import PCA


print('\n')
print('Question 1')
print('---------------------')

#Implement Principle Component Analysis to MNIST data 

with open('mnistTVT.pickle','rb') as f:
    Xtrain,Ttrain,Xval,Tval,Xtest,Ttest = pickle.load(f)
    Xtrain = Xtrain.astype(np.float64)
    Xval = Xval.astype(np.float64)
    Xtest = Xtest.astype(np.float64)
    
warnings.filterwarnings("ignore") 

print("*** Using only 5,000 MNIST training points ***")
X = Xtrain[0:5000]

print('\n')
print('Question 1(a)')
print('-------------')

np.random.seed(0)
def fit_pca(X,n,question):
    
    pca = PCA(n_components=n)
    pca.fit(X)
    
    reduced = pca.transform(Xtest)
    project = pca.inverse_transform(reduced)
    
    row = np.shape(project)[0]
    project = np.reshape(project,(row,28,28))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.axis("off")
        plt.imshow(project[i],cmap = 'Greys')
        plt.suptitle('Question 1(' + question +' ): MNIST test data projected onto '+ str(n) + ' dimensions.')
    plt.show()
 
      
fit_pca(X,30,'a')


print('\n')
print('Question 1(b)')
print('-------------')

fit_pca(X,3,'b')

print('\n')
print('Question 1(c)')
print('-------------')

fit_pca(X,300,'c')




print('\n')
print('Question 1(d)')
print('-------------')

#Implementaation of PCA from scratch

def myPCA(X,K):
     
    mean = np.mean(X,0)
    cov = np.matmul(X-mean,(X-mean).T)/5000
    eigen_val,eigen_vec = np.linalg.eigh(cov)
    # U is the basis corresponds to the largest K amount of eigen values
    
    index = np.shape(eigen_vec)[1]-K
    k_1 = eigen_vec[:,index:]
    U = np.flip(k_1,1)
  
    Z = np.matmul(U.T,(X-mean))
    x_recon = mean + np.matmul(U,Z)    
    return x_recon



print('\n')
print('Question 1(f)')
print('-------------')


#Use the PCA from 1d to train data onto a 100-dimentsional subspace

myXtrainP = myPCA(X,100)
row = np.shape(myXtrainP)[0]
myXtrainP_resize = np.reshape(myXtrainP,(row,28,28))


for i in range(25):
    plt.subplot(5,5,i+1)
    plt.axis("off")
    plt.imshow(myXtrainP_resize[i],cmap = 'Greys')
    plt.suptitle('Question 1(f ): MNIST data projected onto 100 dimensions (mine).')
plt.show()

#sklearn
np.random.seed(0)
pca = PCA(n_components=100,svd_solver="full")
pca.fit(X)    
reduced = pca.transform(X)
XtrainP = pca.inverse_transform(reduced)

row = np.shape(X)[0]
XtrainP_resize = np.reshape(X,(5000,28,28))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.axis("off")
    plt.imshow(XtrainP_resize[i],cmap = 'Greys')
    plt.suptitle('Question 1(f ): MNIST data projected onto 100 dimensions (sklearn).')
plt.show()
rms = np.sqrt(np.sum(np.square(XtrainP-myXtrainP))/row)
print("The RMS is ",rms)



#Q2

print('\n')
print('Question 2')
print('---------------------')

#Regularization with dimentional reduction to reduce overfitting.

print('\n')
print('Question 2(a)')
print('-------------')

#Fitting a QDA into a smaller dataset of the orignal one

small_x = Xtrain[0:200]
small_t = Ttrain[0:200]
debug_x = Xtrain[0:300]
debug_t = Ttrain[0:300] 


np.random.seed(0)
clf = da.QuadraticDiscriminantAnalysis()
fit = clf.fit(small_x,small_t)


accuracy1 = fit.score(small_x,small_t)
accuracy2 = fit.score(Xtest,Ttest)

print("This is accuracy of small training set %.4f" %accuracy1)
print("This is accuracy of full test set %.4f " %accuracy2)


print('\n')
print('Question 2(b)')
print('-------------')

#Creating 20 QDA cluster to find the best regularization parameter for QDA
#reg_param range from 2^-0 to 2^-20

val = []
small = []
size = []

for n in range(20):
    clf = da.QuadraticDiscriminantAnalysis(reg_param=2**-n)
    fit = clf.fit(small_x,small_t)
    # fit = clf.fit(debug_x,debug_t)
    
    accuracy1 = fit.score(small_x,small_t)
    accuracy2 = fit.score(Xval,Tval)
    
    small.append(accuracy1)
    val.append(accuracy2)
    size.append(2**-n)
    
    print("The small training accuracy for "+ str(2**-n)+" is ",accuracy1)
    
    
    
index = val.index(max(val))
print("The maximum validation accuracy is ",val[index])

 
plt.semilogx(size,small,"b")
plt.semilogx(size,val,"r")
plt.title("Question 2(b): Training and Validation Accuracy for Regularized QDA")
plt.show()   
  
 
 #2c is a word question

    
print('\n')
print('Question 2(d)')
print('-------------')

#Used PCA to avoid overffiting before applying QDA
 
#pca,qca, and training
def train2d(K,X,T):   
    pca = PCA(n_components=K,
              svd_solver='full')
    pca.fit(X)
    reduced = pca.transform(X)

    
    qda = da.QuadraticDiscriminantAnalysis()
    qda.fit(reduced,T)
    accuracy = qda.score(reduced,T)
    val_accuracy = test2d(pca,qda,Xval,Tval)  
    
    return pca,qda,accuracy

#validation accuracy 
def test2d(pca,qda,X,T):
    Xreduced = pca.transform(X)
    accuracy = qda.score(Xreduced,T)
    return accuracy
    
K=[]
val=[]
train=[]
i=1
while i <51:
    np.random.seed(0)
    pca,qca,train_accuracy = train2d(i,small_x,small_t)
    # pca,qca,train_accuracy = train2d(i,debug_x,debug_t)
    val_accuracy = test2d(pca,qca,Xval,Tval)
    
    K.append(i)
    val.append(val_accuracy)
    train.append(train_accuracy)
    i+=1

index = val.index(max(val))

print("The maximum validation accuracy is %.4f"%val[index])
print("The corresponding K is",K[index])
print("The corresponding training accuracy is %.4f"%train[index])

plt.plot(K,train,"b")
plt.plot(K,val,"r")
plt.title("Question 2(d): Training and Validation Accuracy for PCA + QDA.")
plt.ylabel("Reduced dimension")
plt.xlabel("Accuracy")
plt.show()


print('\n')
print('Question 2(f)')
print('-------------')

#size = reg_param
#k = k
maxK = []
for i in K:
    pca = PCA(n_components=i,
              svd_solver='full')
    pca.fit(small_x)
    # pca.fit(debug_x)
    reduced = pca.transform(small_x)
    # reduced = pca.transform(debug_x)
    Xval_red = pca.transform(Xval)

    accMaxK = 0
    for j in size:
        clf = da.QuadraticDiscriminantAnalysis(reg_param=2**-j)
        fit = clf.fit(reduced,small_t)
        # fit = clf.fit(reduced,debug_t)
        accuracy = clf.score(Xval_red,Tval)
        
        if accuracy > accMaxK:
            accMaxK = accuracy
    maxK.append(accMaxK)

accMax = max(maxK)

print("The maximum validation accuracy is ",accMax)

plt.plot(K,maxK)
plt.title("Question 2(f ): Maximum validation accuracy for QDA.")
plt.xlabel("Reduced dimension")
plt.ylabel("maximum accuracy")
plt.show()







print('\n')
print('Question 3')
print('---------------------')

#Implementing bagging

print('\n')
print('Question 3(a)')
print('-------------')

#Create a boot strap samples that contain at least 3 training points of each class/digit

def myBootstrap(X,T):
    bol = False
    while bol == False:
        x,t = uti.resample(X,T)
        digit,counts = np.unique(t,return_counts=True)
        has_3 = np.where(counts>=3,True,False)
        if False in has_3:
            bol = False
        else:
            bol = True
    return x,t
        

print('\n')
print('Question 3(b)')
print('-------------')

#Implementing QDA classfier with boot straping 50 times 

qda = da.QuadraticDiscriminantAnalysis(reg_param=0.004)
qda.fit(small_x,small_t)
base_accuracy = qda.score(Xval,Tval)

print("The base classifier accuracy is %.4f"%base_accuracy)


predict = np.empty([50,10000,10])
for i in range(50):

    x,t = myBootstrap(small_x,small_t)
    qda2 = da.QuadraticDiscriminantAnalysis(reg_param=0.004)
    qda2.fit(x,t)

    xval_pre = qda2.predict_proba(Xval)
    predict[i] = xval_pre
    
vec = np.mean(predict,0)
index = np.argmax(vec,1)
print("The accuracy of validation with bagging is",np.mean(index==Tval))




print('\n')
print('Question 3(c)')
print('-------------')

#Creating 500 boot straps samples and 500 qda classfiers to see how they co-relate in a plot.

accuracy = []
num = []
predict = np.empty([500,10000,10])
for i in range(500):

    x,t = myBootstrap(small_x,small_t)
    # x,t = myBootstrap(debug_x,debug_t)
    qda2 = da.QuadraticDiscriminantAnalysis(reg_param=0.004)
    qda2.fit(x,t)
    

    xval_pre = qda2.predict_proba(Xval)
    predict[i] = xval_pre
    
    vec = np.mean(predict,0)
    index = np.argmax(vec,1)
    accuracy.append(np.mean(index==Tval))
    num.append(i+1)
    
plt.plot(num,accuracy)  
plt.title("Question 3(c): Validation accuracy.")
plt.xlabel("Number of bootstrap samples")
plt.ylabel("Accuracy")
plt.show()
   
plt.semilogx(num,accuracy)  
plt.title("Question 3(c): Validation accuracy  (log scale).")
plt.xlabel("Number of bootstrap samples")
plt.ylabel("Accuracy")
plt.show()
   


print('\n')
print('Question 3(d)')
print('-------------')

#Combining pca and qda to train a classfier on data 

def train3d(K,R,X,T):
    
    pca=PCA(n_components=K)
    pca.fit(X)
    reduced = pca.transform(X)

    qda = da.QuadraticDiscriminantAnalysis(reg_param=R)
    qda.fit(reduced,T)
    
    return pca,qda

def proba3d(pca,qda,X):
    
    reduced = pca.transform(X)
    prob = qda.predict_proba(reduced)
    
    return prob

print('\n')
print('Question 3(e)')
print('-------------')

#Similar to part b but incluides dimensionality reduction.

predict2 = np.empty([200,10000,10])

def myBag(K,R):
    
    for i in range(200):
        pca,qda = train3d(K,R,small_x,small_t)
        # pca,qda = train3d(K,R,debug_x,debug_t)
        reduced_xval = pca.transform(Xval)

    
        #base accuracy
        base = qda.score(reduced_xval,Tval)
    
        #boot trap
        bs_x,bs_t = myBootstrap(small_x,small_t)
        # bs_x,bs_t = myBootstrap(debug_x,debug_t)
        pca2,qda2 = train3d(K,R,bs_x,bs_t)
        predict2[i] = proba3d(pca2,qda2,Xval)
    
    vec = np.mean(predict2,0)
    index = np.argmax(vec,1)
    return base,np.mean(index==Tval)

print('\n')
print('Question 3(f)')
print('-------------')
    
base,bag = myBag(100, 0.01)

print("The base accuracy is ",base)
print("The bagged classifier accuracy is ",bag)    
    

print('\n')
print('Question 3(g)')
print('-------------')

#A comparison between bagged and normal base data to see the effect on accuracy 
base_acc = []
bag_acc = []
for i in range(50):
    k = rnd.uniform(1,10)
    r = rnd.uniform(0.2,1)
    k = int(k)
    base,bag = myBag(k,r)
    base_acc.append(base)
    bag_acc.append(bag)
    print(i)
    
plt.scatter(base_acc,bag_acc,c='b')
plt.title("Question  3(g): Bagged v.s. base validation accuracy")
plt.xlabel("Base validation accuracy")
plt.ylabel("Bagged validation accuracy")
plt.xlim(0,1)
plt.ylim(0,1)
plt.plot([0,1],[0,1],c='r')
plt.show()


print('\n')
print('Question 3(h)')
print('-------------')

#Similar to part g but used to find the best accuracy 

base_acc = []
bag_acc = []
for i in range(50):
    k = np.random.uniform(5,200)
    r = np.random.uniform(0,0.5)
    k = int(k)
    base,bag = myBag(k,r)
    base_acc.append(base)
    bag_acc.append(bag)
    print(i)
 
max_acc = np.max(bag_acc)
print("The maximum bagged accuracy is",max_acc)

plt.scatter(base_acc,bag_acc,c='b')
plt.title("Question 3(h): Bagged v.s. base validation accuracy")
plt.xlabel("Base validation accuracy")
plt.ylabel("Bagged validation accuracy")
plt.ylim(0,1)
plt.plot([0,1],[max_acc,max_acc],c='r')
plt.show()




print('\n')
print('Question 4')
print('---------------------')

#Clustering

with open('dataA2Q2.pickle','rb') as file:
    dataTrain,dataTest = pickle.load(file)
    
Xtrain,Ttrain = dataTrain
Xtest,Ttest = dataTest


print('\n')
print('Question 4(a)')
print('-------------')

#Data visualization
#x is Nx2
#R is Nx3
#Mu is 3x2 matrix

def plot_clusters(X,R,Mu):
    total = np.sum(R,0)
    sort = np.argsort(total)
    R_sort= R[:,sort]
    plt.scatter(X[:, 0], X[:, 1], color=R_sort, s=5)
    plt.scatter(Mu[:, 0], Mu[:, 1], color="k")



print('\n')
print('Question 4(b)')
print('-------------')    

#Implemented K-means

clf = KMeans(n_clusters=3)
clf.fit(Xtrain)

accuracy = clf.score(Xtrain)
accuracy2 = clf.score(Xtest)
print("The accuracy of training set is ",accuracy)
print("The accuracy of test set is ",accuracy2) 

Mu = clf.cluster_centers_
label = clf.labels_

row = np.shape(Xtrain)[0]
col = np.max(label) + 1   #since start from 0
R = np.zeros((row,col))

#Setting each row's specific column matching label to 1 
R[np.arange(row),label] = 1 

plot_clusters(Xtrain, R, Mu)
plt.title("Question 4(b): K means")
plt.show()

    
    
    
    
print('\n')
print('Question 4(c)')
print('-------------')

#Implemented Gaussian Mixture

clf = GaussianMixture(n_components=3,covariance_type='spherical')
clf.fit(Xtrain)

accuracy = clf.score(Xtrain)
accuracy2c = clf.score(Xtest)
print("The accuracy of training set is ",accuracy)
print("The accuracy of test set is ",accuracy2c) 

label = clf.predict(Xtrain)
Mu = clf.means_

row = np.shape(Xtrain)[0]
col = np.max(label) + 1   #since start from 0
R = clf.predict_proba(Xtrain)   
    
plot_clusters(Xtrain, R, Mu)
plt.title("Question 4(c): Gaussian mixture model (spherical). ")
plt.show()
   
    
print('\n')
print('Question 4(d)')
print('-------------')

#Similar to 4b but different in convarience_type is full

clf = GaussianMixture(n_components=3,covariance_type='full')
clf.fit(Xtrain)

accuracy = clf.score(Xtrain)
accuracy2 = clf.score(Xtest)
print("The accuracy of training set is ",accuracy)
print("The accuracy of test set is ",accuracy2)
print("The difference of accuracy is ",accuracy2-accuracy2c)   
    
label = clf.predict(Xtrain)
Mu = clf.means_

row = np.shape(Xtrain)[0]
col = np.max(label) + 1   #since start from 0
R = clf.predict_proba(Xtrain)     
    
plot_clusters(Xtrain, R, Mu)
plt.title("Question 4(d): Gaussian mixture model (full).")
plt.show()
    







