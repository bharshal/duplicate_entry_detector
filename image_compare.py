import pandas as pd
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np
from scipy import spatial
from shutil import copy
import os



def distance(im1,im2,path,model):
	im1=path+im1+'.jpeg'
	im2=path+im2+'./jpeg'
	im1=im_preprocess(im1)
	im2=im_preprocess(im2)
	out1=model.predict(im1)
	out2 = model.predict(im2)
	res=1-(spatial.distance.euclidean(out1,out2))              #calculates euclidean distance between 2 embedddings
	return (res)

def VGG_16(weights_path=None):
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(224,224,3)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), dim_ordering='tf',strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), dim_ordering='tf',strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2),dim_ordering='tf', strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), dim_ordering='tf',strides=(2,2)))

	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(64,activation='tanh'))			#custom layer to get embedding
	#model.add(Dense(1000, activation='softmax')) original classify layes

	#if weights_path:
	 #   model.load_weights(weights_path)
	return model

def create_groups(clusters,i):
	#a=pd.DataFrame(columns=['productId',''])
	group=clusters.loc[clusters['cluster']==i]['data_index'].tolist()
	return (group)

def im_preprocess(path): #to format image as per model input
	im = cv2.resize(cv2.imread(path), (224, 224)).astype(np.float32)
	im[:,:,0] -= 103.939
	im[:,:,1] -= 116.779
	im[:,:,2] -= 123.68
	im = im.transpose((1,0,2))
	im = np.expand_dims(im, axis=0)
	return im

def create_similar(clusters):
	model = VGG_16()
	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=sgd, loss='categorical_crossentropy')
	path="./images/"
	#titles=[]
	lists=[]
	a=pd.read_csv('./tops.csv',usecols=['title'],dtype='unicode')
	a=a['title'].tolist()
	for i in range(0,20000):     #20000 clusters
		group=create_groups(clusters,i)
		#n=len(group)
		for i in range(0,(len(group)-1)):
			im1=a[i]
			for j in range(i+1,len(group)):
				im2=a[j]
				res=distance(im1,im2,path,model)
				if res>0.5:
					lists.append([group[i],group[j]])		#creates list of similar images
	return(lists)







