import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb
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
print('start tfidf feature extraction')
train_term_doc = word_vec.fit_transform(train['word_seg'])
test_term_doc = word_vec.transform(test['word_seg'])
print('finish tfidf feature extraction')

X_train, X_eval, y_train, y_eval  = train_test_split(train_term_doc,train['label'],test_size=0.2,shuffle=True,random_state=2019)

print('start lgb training')
model = lgb.LGBMClassifier(boosting_type='gbdt',
                   num_leaves=2**5,
                   max_depth=-1,
                   learning_rate= 0.1,
                   n_estimators=2000,
                   objective='multiclass',
                   subsample=0.7,
                   colsample_bytree=0.5,
                   reg_lambda=10,#l2
                   n_jobs=16,
                   num_class=19,
                   silent=True,
                   random_state=2019,
#                    class_weight=20,
                   colsample_bylevel=0.5,
                   min_child_weight=1.5,
                   metric='multi_logloss',
                   # device='gpu',
                   # gpu_platform_id=0,
                   # gpu_device_id=0
                  )
model.fit(X_train,y_train,eval_set=(X_eval,y_eval), early_stopping_rounds=100)
print('finish lgb training')

def cal_macro_f1(y_true,y_pred):
    score = f1_score(y_true,y_pred,average='macro')
    return score

eval_prob = model.predict_proba(X_eval)
eval_pred = np.argmax(eval_prob,axis=1)
eval_pred = lb.inverse_transform(eval_pred)
score = cal_macro_f1(lb.inverse_transform(y_eval),eval_pred)
print("validation score is",score)

test_prob = model.predict_proba(test_term_doc)
test_pred = np.argmax(test_prob,axis=1)
test['class'] = lb.inverse_transform(test_pred)
test[["id","class"]].to_csv("submission_wordseg_tfidf_lgb.csv",index=False,header=True,encoding='utf-8')