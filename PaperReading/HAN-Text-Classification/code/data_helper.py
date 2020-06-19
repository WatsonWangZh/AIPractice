#coding=utf-8
import os
import json
import pickle
import nltk
#nltk.download()
import numpy as np
from nltk.tokenize import WordPunctTokenizer
from collections import defaultdict

# 使用nltk分词分句器
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
word_tokenizer = WordPunctTokenizer()


def build_vocab(vocab_path, yelp_json_path):

    if os.path.exists(vocab_path):
        vocab_file = open(vocab_path, 'rb')
        vocab = pickle.load(vocab_file)
        print ("load focab finish!")
    else:
        # 记录每个单词及其出现的频率
        word_freq = defaultdict(int)
        # 读取数据集，并进行分词，统计每个单词出现次数，保存在word freq中
        with open(yelp_json_path, 'rb') as f:
            for line in f:
                review = json.loads(line)
                words = word_tokenizer.tokenize(review['text'])
                for word in words:
                    word_freq[word] += 1
            print ("load finished")

        # 构建vocablary，并将出现次数小于5的单词全部去除，视为UNKNOW
        vocab = {}
        i = 1
        vocab['UNKNOW_TOKEN'] = 0
        for word, freq in word_freq.items():
            if freq > 5:
                vocab[word] = i
                i += 1

        # 将词汇表保存下来
        with open(vocab_path, 'wb') as g:
            pickle.dump(vocab, g)
            print (len(vocab))  # 159654
            print ("vocab save finished")

    return vocab

def load_dataset(yelp_json_path, max_sent_in_doc, max_word_in_sent):
    yelp_data_path = yelp_json_path[0:-5] + "_data.pickle"
    vocab_path = yelp_json_path[0:-5] + "_vocab.pickle"
    doc_num = 229907 #数据个数
    if not os.path.exists(yelp_data_path):

        vocab = build_vocab(vocab_path, yelp_json_path)
        num_classes = 5
        UNKNOWN = 0

        data_x = np.zeros([doc_num,max_sent_in_doc,max_word_in_sent])
        data_y = []

        #将所有的评论文件都转化为30*30的索引矩阵，也就是每篇都有30个句子，每个句子有30个单词
        # 不够的补零，多余的删除，并保存到最终的数据集文件之中
        with open(yelp_json_path, 'rb') as f:
            for line_index, line in enumerate(f):

                review = json.loads(line)
                sents = sent_tokenizer.tokenize(review['text'])
                doc = np.zeros([max_sent_in_doc, max_word_in_sent])

                for i, sent in enumerate(sents):
                    if i < max_sent_in_doc:
                        word_to_index = np.zeros([max_word_in_sent],dtype=int)
                        for j, word in enumerate(word_tokenizer.tokenize(sent)):
                            if j < max_word_in_sent:
                                    word_to_index[j] = vocab.get(word, UNKNOWN)
                        doc[i] = word_to_index

                data_x[line_index] = doc
                label = int(review['stars'])
                labels = [0] * num_classes
                labels[label-1] = 1
                data_y.append(labels)
                print (line_index)
            pickle.dump((data_x, data_y), open(yelp_data_path, 'wb'))
            print (len(data_x)) #229907
    else:
        data_file = open(yelp_data_path, 'rb')
        data_x, data_y = pickle.load(data_file)

    length = len(data_x)
    train_x, dev_x = data_x[:int(length*0.9)], data_x[int(length*0.9)+1 :]
    train_y, dev_y = data_y[:int(length*0.9)], data_y[int(length*0.9)+1 :]

    return train_x, train_y, dev_x, dev_y

if __name__ == '__main__':
    load_dataset("../data/yelp_academic_dataset_review.json", 30, 30)
