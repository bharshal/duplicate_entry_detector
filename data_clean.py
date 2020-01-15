import pandas as pd

def clean_data():
	df=pd.read_csv("2oq-c1r.csv",error_bad_lines=False,dtype='unicode',usecols=['productId','title','imageUrlStr','mrp','categories','productBrand','size','color','sellerName'])

	b=df[df['categories'].str.contains('Apparels>Women>Western Wear>Shirts, Tops & Tunics>Tops|Apparels>Women>Fusion Wear>Shirts, Tops & Tunics>Tops',na=False)]

	b.drop(['categories'],axis=1,inplace=True)

	b.dropna(subset=['title'])

	b.to_csv('tops.csv',index=False)

if __name__ == "__main__":
	clean_data()