import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA

import pickle
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_data(filename):
    batch = unpickle(filename)
    labels = np.array(batch[b'labels']).astype(np.int)
    data = np.array(batch[b'data']).astype(np.int)
    return data,labels 

def mean_image(x,y):
    mean = np.zeros((10,x.shape[1]))
    for label in np.arange(len(label_names)):
            separated   = x[y==label]
            mean[label] = np.mean(separated,axis=0)
    return mean

def mean_squared_error(origin,prediction):
    mse = ((origin - prediction)**2).sum(axis=1)
    return mse.mean(axis=0)

def MDS(D):
    A = np.identity(10)-(np.ones((10,10)))/10
    W = -0.5*np.dot(np.dot(A,D),A)
    U,S,vh =  np.linalg.svd(W)
    s  = 2
    Us = U[:,:s]
    Ss = np.zeros((s,s))
    for i in range(s):
        Ss[i,i] = S[i]
    Ss = np.sqrt(Ss)
    Y = np.dot(Us,Ss)
    return Y

file_names=['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch']

label_names = unpickle('batches.meta')[b'label_names']
for i in range(len(label_names)):
    label_names[i]=label_names[i].decode('utf8')

def main():
    X=[]
    Y=[]
    for file_name in file_names:
        x,y=load_data(file_name)
        X.append(x)
        Y.append(y)
    # data include all 5 batches
    data,labels=X[0],Y[0]
    for i in range(1,len(X)):
        data=np.vstack((data,X[i]))
        labels=np.append(labels,Y[i])
    data_by_class=[]
    for i,name in enumerate(label_names):
        data_by_class.append(data[labels==i])

    #part A1: calculate the mean images
    mean = mean_image(data,labels)
    
    ## show image
    plt.figure(1,figsize=(10,2))
    for i,name in enumerate(label_names):
        plt.subplot(5, 2,i+1)
        img = np.reshape(mean[i]/255,(3,32,32)).transpose(1, 2, 0)
        plt.imshow(img)
        plt.title(name, fontsize = 8)

    #Part A2 & A3: compute first 20 pinrcipal components and MSE
    MSE=[]
    PCA_by_class=[]
    for i,name in enumerate(label_names):
        pca = PCA(n_components=20)
        zero_mean=data_by_class[i]-mean[i]
        pca.fit(zero_mean)
        PCA_by_class.append(pca)
        tr=pca.transform(zero_mean)
        inversed_data=pca.inverse_transform(tr)
        mse=mean_squared_error(inversed_data+mean[i], data_by_class[i])
        MSE.append(mse)
    
    # part A4: plot a bar graph
    fig_mse, ax_mse = plt.subplots(2,figsize=(10,5))
    colors = cm.rainbow(np.linspace(0, 1, len(MSE)))
    rects1 = ax_mse.bar(range(len(MSE)), MSE,color=colors)
    ax_mse.set_title('MSE for each Class')
    ax_mse.set_ylabel('MSE')
    ax_mse.set_xticks(np.arange(len(label_names)))
    ax_mse.set_xticklabels(label_names)
    
    # part B1: compute and save 10 x 10 distance matrix D
    D_B = np.zeros((10,10))
    for i in range(10):
        for j in range(i+1,10):
            d = sum((mean[i]-mean[j])**2)
            D_B[i,j] = d
            D_B[j,i] = d
    np.savetxt("partb_distances.csv", D_B, delimiter=",",fmt='%f')
    
    # part B2: multi-dimensional scaling MDS
    Y_B = MDS(D_B)
    
    # part B3: scatter plot of points
    plt.figure(3)
    colors = cm.rainbow(np.linspace(0, 1, len(Y_B)))
    for y, c in zip(Y_B, colors):
        plt.scatter(y[0], y[1], color=c)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('Part B: Scaled points in 2-D Space')
    plt.show()
    
    
    # part C1: compute and save 10x10 distance matrix D
    D_C = np.zeros((10,10))
    for a in range(10):
        for b in range(a,10):           
            # compute E(A|B)
            tr = PCA_by_class[b].transform(data_by_class[a]-mean[a])
            reconstruct_data = mean[a] + PCA_by_class[b].inverse_transform(tr)
            d1 = mean_squared_error(reconstruct_data, data_by_class[a])
            # compute E(B|A)
            tr = PCA_by_class[a].transform(data_by_class[b]-mean[b])
            reconstruct_data = mean[b] + PCA_by_class[a].inverse_transform(tr)
            d2 = mean_squared_error(reconstruct_data, data_by_class[b])
            # compute E(A,B)
            d = (d1+d2)/2
            D_C[a,b] = d
            D_C[b,a] = d
            
    np.savetxt("partc_distances.csv", D_C, delimiter=",",fmt='%f')
    
    # part C2: multi-dimensional scaling MDS
    Y_C = MDS(D_C)

    # scatter plot of points
    plt.figure(4)
    colors = cm.rainbow(np.linspace(0, 1, len(Y_C)))
    for y, c in zip(Y_C, colors):
        plt.scatter(y[0], y[1], color=c)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('Part C: Scaled points in 2-D Space')
    plt.show()

main()