import pandas as pd
import urllib.request
import json
from pathlib import Path

def download_all_images():
	g=open('na.txt','w')

	a=pd.read_csv("tops.csv",dtype='unicode',usecols=['imageUrlStr','productId'])

	b=a['imageUrlStr'].tolist()
	c=a['productId'].tolist()

	opener = urllib.request.build_opener()
	opener.addheaders = [('User-agent', 'Mozilla/5.0')]
	urllib.request.install_opener(opener)
	na=[]
	for i in range(0,len(b)):
		d=b[i].split(';')[2]
		e=c[i]
		print(e)
		path='./images/'+e+'.jpeg'
		a=Path(path)
		if a.is_file():
			print("Already downloaded")
		else:
			try:
				urllib.request.urlretrieve(d,path)
			except Exception as f:
				print("failed",f)          	#some images are not downloadable by automated scripts hence keeping a log
				na.append(e)
	#print(na)
	g.write(json.dumps(na))
	g.close()

	a=pd.read_csv('tops.csv',dtype='unicode')
	a.drop(['imageUrlStr'],axis=1,inplace=True)                 #drop url cloumn as no longer required
	a.to_csv('tops.csv',index=False)


if __name__ == "__main__":
	download_all_images()

