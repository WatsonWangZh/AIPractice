from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM
from keras_contrib.layers.crf import CRF
import process_data
import pickle
from keras.utils.vis_utils import plot_model
from keras.optimizers import SGD,Adam

EMBED_DIM = 200
BiRNN_UNITS = 200


def create_model(train=True):
    
    if train:
        (train_x, train_y), (vocab, chunk_tags) = process_data.load_data()
        with open('model/config.pkl', 'wb') as inp:
            pickle.dump((vocab, chunk_tags),inp)
    else:
        with open('model/config.pkl', 'rb') as inp:
            (vocab, chunk_tags) = pickle.load(inp)
    
    model = Sequential()
    model.add(Embedding(len(vocab), EMBED_DIM, mask_zero=True))  
    model.add(Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True)))
    crf = CRF(len(chunk_tags), sparse_target=True)
    model.add(crf)
    model.summary()
    plot_model(model, to_file="model.png",show_shapes=True)
    model.compile(loss=crf.loss_function,optimizer=Adam(lr=0.01), metrics=[crf.accuracy])
    
    
    if train:
        return model, (train_x, train_y)
    else:
        return model, (vocab, chunk_tags)
