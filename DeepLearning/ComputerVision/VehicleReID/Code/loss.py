import keras.backend as K

MARGIN = 1.

# Refer to https://github.com/maciejkula/triplet_recommendations_keras

def identity_loss(y_true, y_pred):
    return K.mean(y_pred - 0 * y_true)

def triplet_loss(vects):
    # f_anchor.shape = (batch_size, 256)
    f_anchor, f_positive, f_negative = vects
    # L2 normalize anchor, positive and negative, otherwise,
    # the loss will result in ''nan''!
    f_anchor = K.l2_normalize(f_anchors, axis = -1)
    f_positive = K.l2_normalize(f_positive, axis = -1)
    f_negative = K.l2_normalize(f_negative, axis = -1)

    dis_anchor_positive = K.sum(K.square(K.abs(f_anchor - f_positive)),
                                          axis = -1, keepdims = True)

    dis_anchor_negative = K.sum(K.square(K.abs(f_anchor - f_negative)),
                                         axis = -1, keepdims = True)
    loss = dis_anchor_positive + MARGIN - dis_anchor_negative 
    return loss
