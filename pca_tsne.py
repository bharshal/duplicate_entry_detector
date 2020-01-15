from sklearn.manifold import TSNE
import pandas as pd
from sklearn.decomposition import PCA
import threading


def pca(a,b,c,d):
	vec=pd.concat([a,b,c,d]) 	#join 4 datasets from previous function
	b=pd.DataFrame(columns=range(10))
	pca = PCA(n_components=10)
	pca_result = pca.fit_transform(vec)
	del vec
	for i in range (0,10):
		b[i]=pca_result[:,i]
	return (b)


def tsne(b):
	tsne = TSNE(n_components=3, perplexity=5,verbose=3, n_iter=2000)
	tsne_results = tsne.fit_transform(b)
	vec1 = tsne_results[:,0]
	vec2 = tsne_results[:,1]
	vec3 = tsne_results[:,2]
	res=pd.DataFrame({'vector1':vec1,'vector2':vec2,'vector3':vec3})
	return(res)