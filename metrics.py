from tensorflow.keras import backend as K


# Metrics
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true[:,:,:,1])
    y_pred_f = K.flatten(y_pred[:,:,:,1])
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def iou(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true[:,:,:,1] * y_pred[:,:,:,1]), axis=-1)
    union = K.sum(y_true[:,:,:,1], -1) + K.sum(y_pred[:,:,:,1], -1) - intersection
    result = (intersection + smooth) / (union + smooth)
    return result


