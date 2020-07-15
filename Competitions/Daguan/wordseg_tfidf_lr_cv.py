import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

train = pd.read_csv('./data/train_set.csv')
test = pd.read_csv('./data/test_set.csv')
lb = LabelEncoder()
train['label'] = lb.fit_transform(train['class'].tolist())

word_vec = TfidfVectorizer(analyzer='word',
            ngram_range=(1,2),
            min_df=3,
            max_df=0.9,
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True)
train_term_doc = word_vec.fit_transform(train['word_seg'])
test_term_doc = word_vec.transform(test['word_seg'])


kf = KFold(n_splits=10, shuffle=True, random_state=666)
train_matrix = np.zeros((train.shape[0],19))
test_pre_matrix = np.zeros((10,test.shape[0],19))
cv_scores=[]


def cal_macro_f1(y_true, y_pred):
    score = f1_score(y_true, y_pred, average='macro')
    return score


for i, (train_index, eval_index) in enumerate(kf.split(train_term_doc)):
    print(len(train_index), len(eval_index))

    # 训练集
    X_train = train_term_doc[train_index]
    y_train = train['label'][train_index]

    # 验证集
    X_eval = train_term_doc[eval_index]
    y_eval = train['label'][eval_index]

    model = LogisticRegression(C=4, dual=False)
    model.fit(X_train, y_train)

    # 对于验证集进行预测
    eval_prob = model.predict_proba(X_eval)
    train_matrix[eval_index] = eval_prob.reshape((X_eval.shape[0], 19))  # array

    eval_pred = np.argmax(eval_prob, axis=1)
    eval_pred = lb.inverse_transform(eval_pred)
    score = cal_macro_f1(lb.inverse_transform(y_eval), eval_pred)
    cv_scores.append(score)
    print("validation score is", score)

    # 对于测试集进行预测
    test_prob = model.predict_proba(test_term_doc)
    test_pre_matrix[i, :, :] = test_prob.reshape((test_term_doc.shape[0], 19))

all_pred = np.argmax(train_matrix,axis=1)
all_pred = lb.inverse_transform(all_pred)
score = cal_macro_f1(lb.inverse_transform(train['label']),all_pred)
print("all validation score is",score)

test_pred = test_pre_matrix.mean(axis=0)
test_pred = np.argmax(test_pred,axis=1)
test_pred = lb.inverse_transform(test_pred)
test['class'] = test_pred
test[["id", "class"]].to_csv("submission_wordseg_tfidf_lr_cv.csv", index=False, header=True, encoding='utf-8')