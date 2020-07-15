import pandas as pd
import numpy as np
from gensim.models import Word2Vec

train = pd.read_csv('./data/train_set.csv')
test = pd.read_csv('./data/test_set.csv')
df=pd.concat([train,test], axis=0)

sentences=[]
for document in df['word_seg'].tolist():
    sentences.append(document.split(" "))

# print('start training...')
# model = Word2Vec(sentences=sentences,
#      size=200,#维度
#      alpha=0.025, #默认
#      window=5, #默认
#      min_count=2,#2，3
#      sample=0.001,#
#      seed=2018, #
#      workers=11, #线程
#      min_alpha=0.0001,
#      sg=0, #cbow
#      hs=0, #负采样
#      negative=5,#负采样个数
#      ns_exponent=0.75,
#      cbow_mean=1,#求和再取平均
#      iter=10 #10到15
#      )
#
# model.save("./word2vec/word2vec_word_200.model")
# print('saved model')

print('model restoring ...')
model = Word2Vec.load("./word2vec/word2vec_word_200.model")
# print(model.wv['816903'].shape)
# print(model.most_similar("700659",topn=20))
# print(model.wv.similarity("816903","1226448"))
# print(model.wv.vocab.keys())

sentences_next=[]
for document in test['word_seg'].tolist():
    sentences_next.append(document.split(" "))
print('continue training ...')
model.train(sentences=sentences_next, total_examples=model.corpus_count,  epochs=model.iter)
model.save("./word2vec/word2vec_word_200_continue.model")
print('saved continue model')