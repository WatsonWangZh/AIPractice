#! -*- coding:utf-8 -*-

import json
import numpy as np
from bert4keras.backend import keras, K, batch_gather
from bert4keras.layers import LayerNormalization
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
# from bert4keras.snippets import open, groupby
from keras.layers import Input, Dense, Lambda, Reshape
from keras.models import Model
from tqdm import tqdm
import os
import tensorflow as tf

# 设置gpu
os.environ["CUDA_VISIBLE_DEVICES"]="0,1" # 使用编号为1，2号的GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.98
session = tf.Session(config=config)
K.set_session(session)

# 基本信息
maxlen = 320
epochs = 20
batch_size = 16
learning_rate = 2e-5

# bert配置
path = "../bert/"
# path = "data/"
config_path = path+'chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = path+'chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = path+'chinese_L-12_H-768_A-12/vocab.txt'


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

def binary_focal_loss(gamma=2, alpha=0.25, axis=2):
    """
    Binary form of focal loss.
    适用于二分类问题的focal loss
    
    focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    # alpha = tf.constant(alpha, dtype=tf.float32)
    # gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true*alpha + (K.ones_like(y_true)-y_true)*(1-alpha)
    
        p_t = y_true*y_pred + (K.ones_like(y_true)-y_true)*(K.ones_like(y_true)-y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true)-p_t),gamma) * K.log(p_t)
        return K.mean(focal_loss,axis)
    return binary_focal_loss_fixed
# 加载数据集
train_data = load_data('../data/train_data/train_data.json')
valid_data = load_data('../data/dev_data/dev_data.json')
# train_data = train_data + valid_data

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
                # 随机选一个subject
                start, end = np.array(list(spoes.keys())).T
                start = np.random.choice(start)
                tp = start
                end = np.random.choice(end[end >= start])
                subject_ids = (start, end)
                # 对应的object标签
                object_labels = np.zeros((len(token_ids), len(predicate2id), 2))
                for o in spoes.get(subject_ids, []):
                    object_labels[o[0], o[2], 0] = 1
                    object_labels[o[1], o[2], 1] = 1
                # 构建batch
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_subject_labels.append(subject_labels)
                batch_subject_ids.append(subject_ids)
                batch_object_labels.append(object_labels)
                if len(spoes) > 1:
                    # 随机选一个subject
                    start1, end = np.array(list(spoes.keys())).T
                    start = np.random.choice(start1-[tp])
                    end = min(end[start1 >= start])
                    subject_ids = (start, end)
                    # 对应的object标签
                    object_labels = np.zeros((len(token_ids), len(predicate2id), 2))
                    for o in spoes.get(subject_ids, []):
                        object_labels[o[0], o[2], 0] = 1
                        object_labels[o[1], o[2], 1] = 1
                    # 构建batch
                    if len(batch_token_ids) < self.batch_size:
                        batch_token_ids.append(token_ids)
                        batch_segment_ids.append(segment_ids)
                        batch_subject_labels.append(subject_labels)
                        batch_subject_ids.append(subject_ids)
                        batch_object_labels.append(object_labels)
                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    batch_subject_labels = sequence_padding(
                        batch_subject_labels, padding=np.zeros(2)
                    )
                    batch_subject_ids = np.array(batch_subject_ids)
                    batch_object_labels = sequence_padding(
                        batch_object_labels,
                        padding=np.zeros((len(predicate2id), 2))
                    )
                    yield [
                        batch_token_ids, batch_segment_ids,
                        batch_subject_labels, batch_subject_ids,
                        batch_object_labels
                    ], None
                    batch_token_ids, batch_segment_ids = [], []
                    batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], []


def extrac_subject(inputs):
    """根据subject_ids从output中取出subject的向量表征
    """
    output, subject_ids = inputs
    subject_ids = K.cast(subject_ids, 'int32')
    start = batch_gather(output, subject_ids[:, :1])
    end = batch_gather(output, subject_ids[:, 1:])
    subject = K.concatenate([start, end], 2)
    return subject[:, 0]


# 补充输入
subject_labels = Input(shape=(None, 2), name='Subject-Labels')
subject_ids = Input(shape=(2,), name='Subject-Ids')
object_labels = Input(shape=(None, len(predicate2id), 2), name='Object-Labels')

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

subject_model = Model(bert.model.inputs, subject_preds)

# 传入subject，预测object
# 通过Conditional Layer Normalization将subject融入到object的预测中
output = bert.model.layers[-2].get_output_at(-1)
# output1 = bert.model.layers[-2].get_output_at(-1)
subject = Lambda(extrac_subject)([output, subject_ids])
output = LayerNormalization(conditional=True)([output, subject])
# def cc(inputs):
#     output1,output = inputs
#     in1 = K.concatenate([output1,output], axis=2)
#     return in1
# # print("-----------in1",in1.shape)
# in1 = Lambda(cc)([output1,output])
output = Dense(
    units=len(predicate2id) * 2,
    activation='sigmoid',
    kernel_initializer=bert.initializer
)(output)
output = Lambda(lambda x: x**4)(output)
object_preds = Reshape((-1, len(predicate2id), 2))(output)

object_model = Model(bert.model.inputs + [subject_ids], object_preds)

# 训练模型
train_model = Model(
    bert.model.inputs + [subject_labels, subject_ids, object_labels],
    [subject_preds, object_preds]
)
train_model.summary()

mask = bert.model.get_layer('Embedding-Token').output_mask
mask = K.cast(mask, K.floatx())

# subject_loss = K.binary_crossentropy(subject_labels, subject_preds)
# subject_loss = K.mean(subject_loss, 2)
subject_loss = binary_focal_loss()(subject_labels, subject_preds)
subject_loss = K.sum(subject_loss * mask) / K.sum(mask)

# object_loss = K.binary_crossentropy(object_labels, object_preds)
# object_loss = K.sum(K.mean(object_loss, 3), 2)
object_loss = binary_focal_loss(axis=3)(object_labels, object_preds)
object_loss = K.sum(object_loss, 2)
object_loss = K.sum(object_loss * mask) / K.sum(mask)

train_model.add_loss(subject_loss + object_loss)

optimizer = Adam(learning_rate)
train_model.compile(optimizer=optimizer)


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
            subjects.append((i, j))
    if subjects:
        spoes = []
        token_ids = np.repeat([token_ids], len(subjects), 0)
        segment_ids = np.repeat([segment_ids], len(subjects), 0)
        subjects = np.array(subjects)
        # 传入subject，抽取object和predicate
        object_preds = object_model.predict([token_ids, segment_ids, subjects])
        for subject, object_pred in zip(subjects, object_preds):
            start = np.where(object_pred[:, :, 0] > 0.4)
            end = np.where(object_pred[:, :, 1] > 0.4)
            for _start, predicate1 in zip(*start):
                for _end, predicate2 in zip(*end):
                    if _start <= _end and predicate1 == predicate2:
                        spoes.append(
                            ((mapping[subject[0]][0],
                              mapping[subject[1]][-1]), predicate1,
                             (mapping[_start][0], mapping[_end][-1]))
                        )
                        break
        return [(text[s[0]:s[1] + 1], id2predicate[p], text[o[0]:o[1] + 1])
                for s, p, o, in spoes]
    else:
        return []


def combine_spoes(spoes):
    """合并SPO成官方格式
    """
    new_spoes = {}
    for s, p, o in spoes:
        p1, p2 = p.split('_')
        if (s, p1) in new_spoes:
            new_spoes[(s, p1)][p2] = o
        else:
            new_spoes[(s, p1)] = {p2: o}

    return [(k[0], k[1], v) for k, v in new_spoes.items()]


class SPO(tuple):
    """用来存三元组的类
    表现跟tuple基本一致，只是重写了 __hash__ 和 __eq__ 方法，
    使得在判断两个三元组是否等价时容错性更好。
    """
    def __init__(self, spo):
        self.spox = (
            tuple(tokenizer.tokenize(spo[0])),
            spo[1],
            tuple(
                sorted([
                    (k, tuple(tokenizer.tokenize(v))) for k, v in spo[2].items()
                ])
            ),
        )

    def __hash__(self):
        return self.spox.__hash__()

    def __eq__(self, spo):
        return self.spox == spo.spox


def evaluate(data):
    """评估函数，计算f1、precision、recall
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    # f = open('dev_pred.json', 'w', encoding='utf-8')
    # pbar = tqdm()
    for d in data:
        R = combine_spoes(extract_spoes(d['text']))
        T = combine_spoes(d['spo_list'])
        R = set([SPO(spo) for spo in R])
        T = set([SPO(spo) for spo in T])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        # f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        # pbar.update()
        # pbar.set_description(
        #     'f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall)
        # )
        # s = json.dumps({
        #     'text': d['text'],
        #     'spo_list': list(T),
        #     'spo_list_pred': list(R),
        #     'new': list(R - T),
        #     'lack': list(T - R),
        # },
        #                ensure_ascii=False,
        #                indent=4)
        # f.write(s + '\n')
    # pbar.close()
    # f.close()
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


def predict_to_file(in_file, out_file):
    """预测结果到文件，方便提交
    """
    fw = open(out_file, 'w', encoding='utf-8')
    with open(in_file) as fr:
        for l in tqdm(fr):
            l = json.loads(l)
            spoes = combine_spoes(extract_spoes(l['text']))
            spoes = [{
                'subject': spo[0],
                'subject_type': predicate2type[spo[1]][0],
                'predicate': spo[1],
                'object': spo[2],
                'object_type': {
                    k: predicate2type[spo[1]][1][k]
                    for k in spo[2]
                }
            }
                     for spo in spoes]
            l['spo_list'] = spoes
            s = json.dumps(l, ensure_ascii=False)
            fw.write(s + '\n')
    fw.close()


class Evaluator(keras.callbacks.Callback):
    """评估和保存模型
    """
    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = evaluate(valid_data)
        if f1 >= self.best_val_f1 or epoch == 8:
            self.best_val_f1 = f1
            train_model.save_weights('best_model.weights')
        print(
            'f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )


if __name__ == '__main__':

    train_generator = data_generator(train_data, batch_size)
    evaluator = Evaluator()
    train = True
    if train:
        train_model.fit_generator(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            callbacks=[evaluator]
        )

    else:
        train_model.load_weights('best_model.weights')
        # predict_to_file('../data/test1_data/test1_data.json', 'ie_pred.json')
        predict_to_file('../data/test.json', 'ie_test_pred.json')
