#! -*- coding: utf-8 -*-
import json
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from bert4keras.layers import ConditionalRandomField
from keras.layers import Dense
from keras.models import Model
from tqdm import tqdm
import pylcs

# 基本信息
maxlen = 330
epochs = 20
batch_size = 16
learning_rate = 2e-5
crf_lr_multiplier = 100  # 必要时扩大CRF层的学习率

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
            d = {}
            for spo in l['spo_list']:
                for k, v in spo['object'].items():
                    if spo['subject'] not in d.keys():
                        d[spo['subject']] = []
                    d[spo['subject']].append((spo['predicate'] + '_' + k, v))
            D.append((l['text'],d))
    return D

# 读取数据
train_data = load_data('../data/train_data/train_data.json')
# train_data2 = load_data('data/ee/train_data/test1.json')
# train_data = train_data1 + train_data2
valid_data = load_data('../data/dev_data/dev_data.json')

# 读取schema
with open('../data/schema.json') as f:
    id2label, label2id, n = {}, {}, 0
    predicate_k2value = {}
    predicate2subtype = {}
    for l in f:
        l = json.loads(l)
        for k,v in l['object_type'].items():
            predicate2subtype[l['predicate']] = l["subject_type"]
            key = l['predicate']+"_"+k #("毕业院校","@value")
            predicate_k2value[l['predicate']+"_"+k] = v
            id2label[n] = key # {0:("财经/交易-出售/收购","时间")}
            label2id[key] = n # {("财经/交易-出售/收购","时间"):0}
            n += 1
    num_labels = len(id2label) * 2 + 1

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 类似找 答案的开始位置
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
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, arguments) in self.sample(random):
            
            for argument in arguments.items():
                token_ids, segment_ids = tokenizer.encode(argument[0]+"_"+text, max_length=maxlen)
                labels = [0] * len(token_ids)
                for pk, v in argument[1]:
                    a_token_ids = tokenizer.encode(v)[0][1:-1]
                    start_index = search(a_token_ids, token_ids)
                    if start_index != -1:
                        labels[start_index] = label2id[pk] * 2 + 1 #加1是因为 token_ids 包含[CLS]编码
                        for i in range(1, len(a_token_ids)):
                            labels[start_index + i] = label2id[pk] * 2 + 2 # 开始时a其他是b 类似 机构名标注 booo
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append(labels)
                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    batch_labels = sequence_padding(batch_labels)
                    yield [batch_token_ids, batch_segment_ids], batch_labels
                    batch_token_ids, batch_segment_ids, batch_labels = [], [], []


model = build_transformer_model(
    config_path,
    checkpoint_path,
)

output = Dense(num_labels)(model.output)
CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
output = CRF(output)

model = Model(model.input, output)
model.summary()

model.compile(
    loss=CRF.sparse_loss,
    optimizer=Adam(learning_rate),
    metrics=[CRF.sparse_accuracy]
)

model.load_weights('ee_best_model.weights')
def viterbi_decode(nodes, trans):
    """Viterbi算法求最优路径
    其中nodes.shape=[seq_len, num_labels],
        trans.shape=[num_labels, num_labels].
    """
    labels = np.arange(num_labels).reshape((1, -1))
    scores = nodes[0].reshape((-1, 1))
    scores[1:] -= np.inf  # 第一个标签必然是0
    paths = labels
    for l in range(1, len(nodes)):
        M = scores + trans + nodes[l].reshape((1, -1))
        idxs = M.argmax(0)
        scores = M.max(0).reshape((-1, 1))
        paths = np.concatenate([paths[:, idxs], labels], 0)
    return paths[:, scores[0].argmax()]


def extract_arguments(text,subject):
    """命名实体识别函数
    """
    text = subject+"_" + text
    tokens = tokenizer.tokenize(text)
    while len(tokens) > 510:
        tokens.pop(-2)
    mapping = tokenizer.rematch(text, tokens)
    token_ids = tokenizer.tokens_to_ids(tokens)
    segment_ids = [0] * len(token_ids)
    nodes = model.predict([[token_ids], [segment_ids]])[0]
    trans = K.eval(CRF.trans)
    labels = viterbi_decode(nodes, trans)
    arguments, starting = [], False
    for i, label in enumerate(labels):
        if label > 0:
            ch = text[mapping[i][0]:mapping[i][-1] + 1]
            if label % 2 == 1:
                starting = True
                arguments.append([[i], id2label[(label - 1) // 2]])
            elif starting:
                arguments[-1][0].append(i)
            else:
                starting = False
        else:
            starting = False

    return {
        text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1]: l
        for w, l in arguments
    }






# else:
#     model.load_weights('ee_best_model.weights')
#     predict_to_file('../../data/test1_data/test1.json', 'ee_pred.json')
