import pandas as pd
from data_clean import clean_data
from doc2vec_multithread import make_vectors
from pca_tsne import pca,tsne
from kmeans_cluster import cluster
import json
from image_compare import create_similar
import warnings
warnings.simplefilter("ignore")


def final(similar):

	a=pd.read_csv('tops.csv',usecols=['productId','mrp','size','color','productBrand','sellerName'])
	f=open('result.txt','r')
	data=json.loads(f.read())
	prodid=a['productId'].tolist()
	mrp=a['mrp'].tolist()
	size=a['size'].tolist()
	color=a['color'].tolist()
	brand=a['productBrand'].tolist()
	seller=a['sellerName'].tolist()
	del a
	for i in range(0,len(similar)):
		conf=5
		a=similar[i]
		index1=a[0]
		index2=a[1]
		if (((mrp[index1]*0.8)<mrp[index2]<(mrp[index1]*1.2))):
			conf=conf+1
		if (size[index1]==size[index2]):
			conf=conf+1
		if (color[index1]==color[index2]):
			conf=conf+1
		if (brand[index1]==brand[index2]):
			conf=conf+1
		if (seller[index1]==seller[index2]):
			conf=conf+1
		if conf>=9:
			prod1=str(prodid[index1])
			prod2=str(prodid[index2])
			if (prod1 in data.keys()):
				data[prod1].append([prod2])
			elif (prod2 in data.keys()):
				data[prod2].append([prod1])
			else:
				data[prod1]=[prod2]

	f=open('result.txt','w')
	f.write(json.dumps(data))
	f.close()


if __name__ == "__main__":
	clean_data()
	a,b,c,d=make_vectors()
	pca=pca(a,b,c,d)
	tsne=tsne(pca)
	clusters=cluster(tsne)
	similar=create_similar(cluster)
	final(similar)
	


