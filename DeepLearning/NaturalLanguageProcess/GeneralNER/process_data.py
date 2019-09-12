import numpy
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
import pickle
import platform
import numpy as np


def load_data():
    train = _parse_data(open('data/train_data.data', 'rb'))
    test = _parse_data(open('data/test_data.data', 'rb'))
    
    print(np.array(train).shape)
    word_counts = Counter(row.lower() for sample in train for row in sample)
    vocab = [w for w, f in iter(word_counts.items()) if f >= 2]
    #B表示开始的字节，I表示中间的字节
    chunk_tags = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', "B-ORG", "I-ORG"]

    # 保存数据
    with open('model/config.pkl', 'wb') as outp:
        pickle.dump((vocab, chunk_tags), outp)

    train = _process_data(train, vocab, chunk_tags)
    return train, (vocab, chunk_tags)


def _parse_data(fh):
    string = fh.read().decode('utf-8')
    data = []
    for sample in string.split(':'):
        data.append(sample.split())       
    fh.close()
    return data


def _process_data(data, vocab, chunk_tags, maxlen=None, onehot=False):
    if maxlen is None:
        maxlen = max(len(s) for s in data)
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    x = [[word2idx.get(w.lower(), 1)for w in s[::2]] for s in data]#奇数
    y_chunk = [[chunk_tags.index(w)for w in s[1::2]] for s in data]#偶数
    x = pad_sequences(x, maxlen)  # left padding
    y_chunk = pad_sequences(y_chunk, maxlen, value=-1)
    if onehot:
        y_chunk = numpy.eye(len(chunk_tags), dtype='float32')[y_chunk]
    else:
        y_chunk = numpy.expand_dims(y_chunk, 2)
    return x, y_chunk


def process_data(data, vocab, maxlen=100):
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    x = [word2idx.get(w[0].lower(), 1) for w in data]
    length = len(x)
    x = pad_sequences([x], maxlen)  # left padding
    return x, length
