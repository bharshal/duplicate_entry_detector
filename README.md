# Duplicate entry detector
## in large e-commerce database of clothing products
### using word vectorisation, image embeddings, clustering and some more





The flow of the app is :
![alt text](https://github.com/bharshal/duplicate_entry_detector/blob/master/FLow.jpg)

To test, copy the database into this folder,
1) run clean_data.py,
2) run download_images.py,
3) run train_doc2vec_model.py or  paste final.model in this folder
4) run main.py

Result will be stored in result.txt in json format. The json dictionary has all duplicates for each unique product. 

If anyone wants to delve deeper into the app please read this: 
[Report](https://github.com/bharshal/duplicate_entry_detector/blob/master/Report.pdf)


The database used is a 8gb csv datadump of Flipkart.
If anyone requires the database please contact me.


P.S.
This app was made by me in 2 nights as a demo and learning experience, years ago. 
I don't make any claims as to the execution of the code or even the technical accuracy.
