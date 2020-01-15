from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import word_tokenize
import pandas as pd
import re
import threading 

global a    #global required to return them to next function
global b
global c
global d

def make_vectors():

	a=pd.read_csv('./tops.csv',dtype='unicode',usecols=['title'])
	split=len(a)
	split=int(split/4)             #divide dataset into 4 parts
	w=a[0:split]
	x=a[split:(2*split)]
	y=a[(2*split):(3*split)]
	z=a[(3*split):]

	data1=w['title'].tolist()
	data2=x['title'].tolist()
	data3=y['title'].tolist()
	data4=z['title'].tolist()
	global a
	a=pd.DataFrame(columns=range(500))
	global b
	b=pd.DataFrame(columns=range(500))
	global c
	c=pd.DataFrame(columns=range(500))
	global d
	d=pd.DataFrame(columns=range(500))

	model= Doc2Vec.load("final.model")

	t1=threading.Thread(target=Thread1,args=[data1])	#4 threads to run paralelly processing 1/4th of the dataset
	t2=threading.Thread(target=Thread2,args=[data2])
	t3=threading.Thread(target=Thread3,args=[data3])
	t4=threading.Thread(target=Thread4,args=[data4])

	threads=[t1,t2,t3,t4]

	threads.start()
	
	threads.join()	#to make thread wait till other threads are done

	return(a,b,c,d)

def Thread1(data1):
	global a
	i=0
	for word in data1:
		print(i)
		test1 = word_tokenize((re.sub('[^A-Za-z]+', '',str(word))).lower())
		v1 = model.infer_vector(test1)
		#print(v1.tolist())
		a.loc[i]=v1.tolist()
		i=i+1

def Thread2(data2):
	global b
	j=0
	for word2 in data2:
		print(j)
		test2 = word_tokenize((re.sub('[^A-Za-z]+', '',str(word2))).lower())
		v2 = model.infer_vector(test2)
		#print(v1.tolist())
		b.loc[j]=v2.tolist()
		j=j+1

def Thread3(data3):
	global c
	k=0
	for word3 in data3:
		print(k)
		test3 = word_tokenize((re.sub('[^A-Za-z]+', '',str(word3))).lower())
		v3 = model.infer_vector(test3)
		#print(v1.tolist())
		c.loc[k]=v3.tolist()
		k=k+1


def Thread4(data4):
	global d
	l=0
	for word4 in data4:
		print(l)
		test4 = word_tokenize((re.sub('[^A-Za-z]+', '',str(word4))).lower())
		v4 = model.infer_vector(test4)
		#print(v1.tolist())
		d.loc[l]=v4.tolist()
		l=l+1

if __name__ == "__main__":
	a,b,c,d=make_vectors()
	#a.to_csv('vec1.csv',index=False)
	#b.to_csv('vec2.csv',index=False)
	#c.to_csv('vec3.csv',index=False)
	#d.to_csv('vec4.csv',index=False)


