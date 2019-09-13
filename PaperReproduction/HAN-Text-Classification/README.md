# HAN-Text-Classification
This code belongs to the "[Implementation of Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)" .

# Structure:
![model](model.png)
 - embedding

 - Word Encoder: word level bi-directional GRU to get rich representation of words

 - Word Attention:word level attention to get important information in a sentence

 - Sentence Encoder: sentence level bi-directional GRU to get rich representation of sentences

 - Sentence Attetion: sentence level attention to get important sentence among sentences

 - FC+Softmax

# Requirement:

 - Python 3
 - Tensorflow 1.1 +
 - numpy
 
# Notes:
 Please load the dataset in this [address](https://github.com/rekiksab/Yelp/tree/master/yelp_challenge/yelp_phoenix_academic_dataset).
 And add the dataset to data folder.
