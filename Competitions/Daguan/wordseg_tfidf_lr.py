import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

train = pd.read_csv('./data/train_set.csv')
test = pd.read_csv('./data/test_set.csv')

lb = LabelEncoder()
train['label'] = lb.fit_transform(train['class'])
train['class_trans'] = lb.inverse_transform(train['label'])

word_vec = TfidfVectorizer(analyzer='word',
            ngram_range=(1,2),
            min_df=3,  
            max_df=0.9,  
            use_idf=True,
            smooth_idf=True, 
            sublinear_tf=True)

train_term_doc = word_vec.fit_transform(train['word_seg'])
test_term_doc = word_vec.transform(test['word_seg'])
