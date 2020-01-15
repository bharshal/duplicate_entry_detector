from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import pandas as pd 
import re


df=pd.read_csv('./tops.csv',dtype='unicode',usecols=['title'])

data=df['title'].tolist()
tagged_data = [TaggedDocument(words=word_tokenize((re.sub('[^A-Za-z]+', '',str(_d))).lower()), tags=[str(i)]) for i, _d in enumerate(data)]
#tokenize words to run through model
max_epochs = 1000
vec_size = 500
alpha = 0.025

model = Doc2Vec(vector_size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                workers=4,      #multithreaded
                dm =1,dbow_words=0)
  
model.build_vocab(tagged_data)
i=1
for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    
    if model.alpha >= 0.00015:
        model.alpha -= (0.004/i) 
    #print(model.alpha)
    i=i+1

model.save("final.model")
print("Model Saved")
