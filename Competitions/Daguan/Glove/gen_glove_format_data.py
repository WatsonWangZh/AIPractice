import pandas as pd


#读取数据集
train = pd.read_csv('../data/train_set.csv')
test = pd.read_csv('../data/test_set.csv')

df=pd.concat([train['word_seg'],test['word_seg']])


with open("glove_word_data.txt",'w') as f:
    for line in df.values:
        f.write(line+'\n')
f.close()