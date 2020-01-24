# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 23:47:21 2019

@author: 1
"""
"""
import pandas as pd
import numpy as np
import csv
import os.path
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

data_dir='D:/data Competition/digit-recognizer/'

def opencsv():
    data_train=pd.read_csv(os.path.join(data_dir,'train.csv'))
    data_test=pd.read_csv(os.path.join(data_dir,'test.csv'))

    train_data=data_train.values[0:,1:] #读入全部训练数据,  [行，列]
    train_label=data_train.values[0:,0] # 读取列表的第一列
    test_data=data_test.values[0:,0:]  # 测试全部测试个数据

    return train_data,train_label,test_data
def saveResults(result,csvName):
    with open(csvName,'w') as myfile:
        '''
       创建记录输出结果的文件（w 和 wb 使用的时候有问题）
       python3里面对 str和bytes类型做了严格的区分，不像python2里面某些函数里可以混用。
       所以用python3来写wirterow时，打开文件不要用wb模式，只需要使用w模式，然后带上newline=''
       '''
        mywrite=csv.writer(myfile)
        mywrite.writerow(["ImageId","Label"])
        index=0
        for r in result:
            index+=1
            mywrite.writerow([index,int(r)])
        print('Saved successfully....')
def dpPCA(x_train,x_test,Com):
    print('dimension reduction....')
    trainData=np.array(x_train)
    testData=np.array(x_test)
    '''
    n_components>=1
      n_components=NUM   设置占特征数量比
    0 < n_components < 1
      n_components=0.99  设置阈值总方差占比
    '''
    pca=PCA(n_components=Com,whiten=False)
    pca.fit(trainData) #fit the model with X
    pcaTrainData=pca.transform(trainData)# 在 X上进行降维
    pcaTestData=pca.transform(testData)
    #pca 方差大小 方差占比 特征数量
   # print(pca.explained_variance_,'\n',pca.explained_variance_ratio_,'\n',pca.components_)
    return pcaTrainData,pcaTestData
def dtClassify(traindata,trainlabel):
    print('Train decision tree...')
    dtClf=DecisionTreeClassifier()
    dtClf.fit(traindata,np.ravel(trainlabel))
    return dtClf

def rfClassify(traindata,trainlabel):
    print('Train Random forest...')
    rfClf=RandomForestClassifier()
    rfClf.fit(traindata,np.ravel(trainlabel))
    return rfClf

def svmClassify(traindata,trainlabel):
    print('Train svm...')
    svmClf=SVC(C=4,kernel='rbf')
    svmClf.fit(traindata,np.ravel(trainlabel))
    return svmClf
trainData,trainLabel,testData = opencsv()
trainData,testData=dpPCA(trainData,testData,0.8)
'''
knnClf=knnClassify(trainData,trainLabel)
trainlabel_knn=knnClf.predict(trainData)
testLabel_knn=knnClf.predict(testData)
'''
'''
mlpClf = MLPClassify(trainData,trainLabel)
trainlabel_mlp=mlpClf.predict(trainData)
testLabel_mlp=mlpClf.predict(testData)
'''
svmClf = dtClassify(trainData,trainLabel)
trainlabel_mlp=svmClf.predict(trainData)
testLabel_mlp=svmClf.predict(testData)
saveResults(testLabel_mlp,os.path.join(data_dir,'Result_knn.csv'))
print(svmClf.score(trainData,trainLabel))
print(svmClf.score(testData,testLabel_mlp))
"""
#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 11:03:20 2018

@author: Gunther17
"""

"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

from keras.callbacks import ReduceLROnPlateau
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical  # convert to one-hot-encoding


np.random.seed(2)

# 数据路径
data_dir = 'D:\data Competition\digit-recognizer/'

# Load the data
train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test = pd.read_csv(os.path.join(data_dir, 'test.csv'))

X_train = train.values[:, 1:]
Y_train = train.values[:, 0]
test = test.values

# Normalization
X_train = X_train / 255.0
test = test / 255.0

# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
X_train = X_train.reshape(-1, 28, 28, 1)
test = test.reshape(-1, 28, 28, 1)

# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
Y_train = to_categorical(Y_train, num_classes=10)

# Set the random seed
random_seed = 2

# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train, Y_train, test_size=0.1, random_state=random_seed)

# Set the CNN model 
# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

model = Sequential()

model.add(
    Conv2D(
        filters=32,
        kernel_size=(5, 5),
        padding='Same',
        activation='relu',
        input_shape=(28, 28, 1)))
model.add(
    Conv2D(
        filters=32, kernel_size=(5, 5), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(
    Conv2D(
        filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(
    Conv2D(
        filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

# Define the optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# Compile the model
model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

epochs = 30
batch_size = 86

# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.1,  # Randomly zoom image 
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(X_train)

history = model.fit_generator(
    datagen.flow(
        X_train, Y_train, batch_size=batch_size),
    epochs=epochs,
    validation_data=(X_val, Y_val),
    verbose=2,
    steps_per_epoch=X_train.shape[0] // batch_size,
    callbacks=[learning_rate_reduction])

# predict results
results = model.predict(test)

# select the indix with the maximum probability
results = np.argmax(results, axis=1)

results = pd.Series(results, name="Label")

submission = pd.concat(
    [pd.Series(
        range(1, 28001), name="ImageId"), results], axis=1)

submission.to_csv(os.path.join(data_dir, 'Result_knn.csv'),index=False)
print('finished')

def delblankline(infile, outfile):
    ''' Delete blanklines of infile '''
    infp = open(infile, "r")
    outfp = open(outfile, "w")
    lines = infp.readlines()
    for li in lines:
        if li.split():
            outfp.writelines(li)
    infp.close()
    outfp.close()
#调用示例

if __name__ == "__main__":
  delblankline("Result_knn.csv","ok.csv")
"""