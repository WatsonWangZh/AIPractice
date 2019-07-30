#coding: utf-8
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Merge
from sklearn.preprocessing import MinMaxScaler

#所有的数据列
COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status", 
    "occupation", "relationship", "race", "gender", "capital_gain", "capital_loss", 
    "hours_per_week", "native_country", "income_bracket"
]

#标签列
LABEL_COLUMN = "label"

#类别型特征变量
CATEGORICAL_COLUMNS = [
    "workclass", "education", "marital_status", "occupation", "relationship", 
    "race", "gender", "native_country"
]

#连续值特征变量
CONTINUOUS_COLUMNS = [
    "age", "education_num", "capital_gain", "capital_loss", "hours_per_week"
]

#加载文件
def load(filename):
    with open(filename, 'r') as f:
        skiprows = 1 if 'test' in filename else 0
        df = pd.read_csv(
            f, names=COLUMNS, skipinitialspace=True, skiprows=skiprows, engine='python'
        )
        #缺省值处理
        df = df.dropna(how='any', axis=0)
    return df

#预处理
def preprocess(df):
    df[LABEL_COLUMN] = df['income_bracket'].apply(lambda x: ">50K" in x).astype(int)
    df.pop("income_bracket")
    y = df[LABEL_COLUMN].values
    df.pop(LABEL_COLUMN)
    
    df = pd.get_dummies(df, columns=[x for x in CATEGORICAL_COLUMNS])
    # 规范化
    df = pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns)

    X = df.values
    # 特征工程，加入交叉与组合特征
    from sklearn.preprocessing import PolynomialFeatures
    X = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False).fit_transform(X)
    # TODO: 对特征进行选择，使得网络更高效
    return X, y

def main():
    df_train = load('data/adult.data')
    df_test = load('data/adult.test')
    df = pd.concat([df_train, df_test])
    train_len = len(df_train)
    
    X, y = preprocess(df)
    X_train = X[:train_len]
    y_train = y[:train_len]
    X_test = X[train_len:]
    y_test = y[train_len:]
    
    #Wide部分
    wide = Sequential()
    wide.add(Dense(1, input_dim=X_train.shape[1]))
    
    #Deep部分
    deep = Sequential()
    # TODO: 添加embedding层
    deep.add(Dense(input_dim=X_train.shape[1], output_dim=100, activation='relu'))
    #deep.add(Dense(100, activation='relu'))
    deep.add(Dense(input_dim=100, output_dim=32, activation='relu'))
    #deep.add(Dense(50, activation='relu'))
    deep.add(Dense(input_dim=32, output_dim=8))
    deep.add(Dense(1, activation='sigmoid'))
    
    #Wide和Deep拼接
    model = Sequential()
    model.add(Merge([wide, deep], mode='concat', concat_axis=1))
    model.add(Dense(1, activation='sigmoid'))
    
    #编译模型
    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    #模型训练
    model.fit([X_train, X_train], y_train, nb_epoch=10, batch_size=32)
    
    #loss与准确率评估
    loss, accuracy = model.evaluate([X_test, X_test], y_test)
    print('\n', 'test accuracy:', accuracy)
    
if __name__ == '__main__':
    main()