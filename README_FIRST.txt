Entire code has been made and tested on Linux (Ubuntu 16.04), behaviour on Windows is not known

The python modules used are:

urllib3                 1.22
nltk                    3.3        
notebook                5.5.0      
numpy                   1.14.5     
pandas                  0.22.0 
tensorflow-gpu          1.8.0
requests                2.18.4     
Keras                   2.1.6       
scikit-learn            0.19.1     
scipy                   1.0.0 
gensim                  3.6.0   

*may have other dependencies



main.py calls all other modules except download_iamges.py and train_doc2vec_model.py
To test, paste 2oq-c1r.csv in this folder,
	 run clean_data.py,
	 run download_images.py,
	 run train_doc2vec_model.py or  paste final.model in this folder
	 and run main.py
