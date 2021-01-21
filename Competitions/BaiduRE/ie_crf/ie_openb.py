#! -*- coding:utf-8 -*-

import json
import numpy as np
from bert4keras.backend import keras, K, batch_gather
from bert4keras.layers import LayerNormalization
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Input, Dense, Lambda, Reshape, RepeatVector
from keras.models import Model
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 基本信息
maxlen = 320
epochs = 20
batch_size = 1
learning_rate = 2e-5

path = "../bert/"
# bert配置
config_path = path + 'chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = path + 'chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = path + 'chinese_L-12_H-768_A-12/vocab.txt'


def load_data(filename):
    D = []
    with open(filename) as f:
        for l in f:
            l = json.loads(l)
            d = {'text': l['text'], 'spo_list': []}
            for spo in l['spo_list']:
                for k, v in spo['object'].items():
                    d['spo_list'].append(
                        (spo['subject'], spo['predicate'] + '_' + k, v)
                    )
            D.append(d)
    return D
def load_data_test(filename):
    D = []
    with open(filename) as f:
        for l in f:
            l = json.loads(l)
            
            D.append(l)
    return D

# 加载数据集
train_data = load_data('../data/train_data/train_data.json')
valid_data = load_data_test('../data/dev_data/dev_data.json')

# 读取schema
with open('../data/schema.json') as f:
    id2predicate, predicate2id, n = {}, {}, 0
    predicate2type = {}
    for l in f:
        l = json.loads(l)
        predicate2type[l['predicate']] = (l['subject_type'], l['object_type'])
        for k, _ in sorted(l['object_type'].items()):
            key = l['predicate'] + '_' + k
            id2predicate[n] = key
            predicate2id[key] = n
            n += 1

# with open('data/all_50_schemas') as f:
#     for l in f:
#         l = json.loads(l)
#         if l['predicate'] not in predicate2id:
#             id2predicate[len(predicate2id)] = l['predicate']
#             predicate2id[l['predicate']] = len(predicate2id)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], []
        for is_end, d in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(
                d['text'], max_length=maxlen
            )
            # 整理三元组 {s: [(o, p)]}
            spoes = {}
            for s, p, o in d['spo_list']:
                s = tokenizer.encode(s)[0][1:-1]
                p = predicate2id[p]
                o = tokenizer.encode(o)[0][1:-1]
                s_idx = search(s, token_ids)
                o_idx = search(o, token_ids)
                if s_idx != -1 and o_idx != -1:
                    s = (s_idx, s_idx + len(s) - 1)
                    o = (o_idx, o_idx + len(o) - 1, p)
                    if s not in spoes:
                        spoes[s] = []
                    spoes[s].append(o)
            if spoes:
                # subject标签
                subject_labels = np.zeros((len(token_ids), 2))
                for s in spoes:
                    subject_labels[s[0], 0] = 1
                    subject_labels[s[1], 1] = 1
                
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_subject_labels.append(subject_labels)
                
                
                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    batch_subject_labels = sequence_padding(
                        batch_subject_labels, padding=np.zeros(2)
                    )
                    
                    yield [
                        batch_token_ids, batch_segment_ids
                        
                    ], batch_subject_labels
                    batch_token_ids, batch_segment_ids = [], []
                    batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], []


def extrac_subject(inputs):
    """根据subject_ids从output中取出subject的向量表征
    """
    output, subject_ids = inputs
    subject_ids = K.cast(subject_ids, 'int32')
    # batch_gather 通过索引获取数组
    start = batch_gather(output, subject_ids[:, :1])
    end = batch_gather(output, subject_ids[:, 1:])
    subject = K.concatenate([start, end], 2)
    return subject[:, 0]
   


# 补充输入
# subject_labels = Input(shape=(None, 2), name='Subject-Labels')
# subject_ids = Input(shape=(2,), name='Subject-Ids')
# print("---------subject_ids",subject_ids.shape)
# object_labels = Input(shape=(None, len(predicate2id), 2), name='Object-Labels')

# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False,
)

# 预测subject
output = Dense(
    units=2, activation='sigmoid', kernel_initializer=bert.initializer
)(bert.model.output)
subject_preds = Lambda(lambda x: x**2)(output)

mask = bert.model.get_layer('Embedding-Token').output_mask
mask = K.cast(mask, K.floatx())

# subject_loss = K.binary_crossentropy(subject_labels, subject_preds)
# subject_loss = K.mean(subject_loss, 2)
# subject_loss = K.sum(subject_loss * mask) / K.sum(mask)

subject_model = Model(bert.model.inputs, subject_preds)

subject_model.compile(
    # loss = subject_loss,
    loss = "binary_crossentropy",
    optimizer=Adam(learning_rate),
    # metrics=['accuracy']
    )
# subject_model.load_weights('best_model.weights')

def extract_spoes(text):
    """抽取输入text所包含的三元组
    """
    tokens = tokenizer.tokenize(text, max_length=maxlen)
    mapping = tokenizer.rematch(text, tokens)
    token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
    # 抽取subject
    subject_preds = subject_model.predict([[token_ids], [segment_ids]])
    start = np.where(subject_preds[0, :, 0] > 0.4)[0]
    end = np.where(subject_preds[0, :, 1] > 0.4)[0]
    subjects = []
    for i in start:
        j = end[end >= i]
        if len(j) > 0:
            j = j[0]
            subjects.append(text[mapping[i][0]:mapping[j][-1]+1])
    
    return subjects

def combine_spoes(dlist):
    sub = []
    for i in dlist:
        # print(i)
        sub.append(i["subject"])
    return sub


def evaluate(data):
    """评估函数，计算f1、precision、recall
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = open('dev_pred.json', 'w', encoding='utf-8')

    for d in data:
        R = extract_spoes(d['text'])
        T = combine_spoes(d['spo_list'])
        R = set(R)
        T = set(T)
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        
        s = json.dumps({
            'text': d['text'],
            'spo_list': list(T),
            'spo_list_pred': list(R)
        },
                       ensure_ascii=False,
                       indent=4)
        f.write(s + '\n')
    f.close()
    return f1, precision, recall


def predict_to_file(in_file, out_file):
    """预测结果到文件，方便提交
    """
    fw = open(out_file, 'w', encoding='utf-8')
    with open(in_file) as fr:
        for l in tqdm(fr):
            l = json.loads(l)
            spoes = extract_spoes(l['text'])
            spoes = {
                'subject': spoes}
            s = json.dumps(spoes, ensure_ascii=False)
            fw.write(s + '\n')
    fw.close()


class Evaluator(keras.callbacks.Callback):
    """评估和保存模型
    """
    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = evaluate(valid_data)
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            subject_model.save_weights('ie_best_model.weights')
        print(
            'f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )


if __name__ == '__main__':

    train_generator = data_generator(train_data, batch_size)
    evaluator = Evaluator()
    train = True
    if train:
        subject_model.fit_generator(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            # steps_per_epoch=1,
            epochs=epochs,
            callbacks=[evaluator]
        )
    else:
        subject_model.load_weights('ie_best_model.weights')
        predict_to_file('../data/test1_data/test1_data.json', 'subjects_pred.json')