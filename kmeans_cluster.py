import pandas as pd
from sklearn.cluster import KMeans

def cluster(res):

	km = KMeans(n_clusters=20000,max_iter=3000, random_state=0).fit(res)
	cluster_map = pd.DataFrame()
	cluster_map['data_index'] = res.index.values
	cluster_map['cluster'] = km.labels_
	cluster_map.to_csv('clusters.csv',index=False)
	return(cluster_map)
