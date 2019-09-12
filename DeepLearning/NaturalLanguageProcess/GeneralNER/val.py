import bilsm_crf_model
import process_data
import numpy as np

model, (vocab, chunk_tags) = bilsm_crf_model.create_model(train=False)
predict_text = '国务院总理在外交部长陈毅陪同下，访问了埃塞俄比亚'
str, length = process_data.process_data(predict_text, vocab)
# print (str.shape)
model.load_weights('model/crf.h5')
raw = model.predict(str[0])[-length:]
# print (raw)
# print (raw.shape)
result = [np.argmax(row) for row in raw[:,0]]
result_tags = [chunk_tags[i] for i in result]

per, loc, org = '', '', ''

for s, t in zip(predict_text, result_tags):
    if t in ('B-PER', 'I-PER'):
        per += ' ' + s if (t == 'B-PER') else s
    if t in ('B-ORG', 'I-ORG'):
        org += ' ' + s if (t == 'B-ORG') else s
    if t in ('B-LOC', 'I-LOC'):
        loc += ' ' + s if (t == 'B-LOC') else s

print(['person:' + per, 'location:' + loc, 'organzation:' + org])
