import numpy as np
import cv2 
import tflite_runtime.interpreter as tflite

def load_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32')
    img /= 255.
    img = np.expand_dims(img, axis=0)
    return img

def get_tensor(frame):
    img = cv2.resize(frame, (224, 224))
    img_tensor = img.astype('float32')
    img_tensor /= 255.
    img_tensor = np.expand_dims(img_tensor, axis=0)
    return img_tensor

def load_net(model_path):
    return tflite.Interpreter(model_path=model_path)


def create_mask(pred_mask):
    pred_mask = np.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., np.newaxis]
    return pred_mask[0]


def predToGrayImage(predicted):
    img = np.zeros((predicted.shape[0], predicted.shape[1], 3))
    img[:, :, 0] = predicted[:, :, 0] * 254
    img[:, :, 1] = img[:, :, 0]  # Make greyscale
    img[:, :, 2] = img[:, :, 0]  # Make greyscale
    return img.astype('uint8')


def segment(frame, mask, rows, cols):
    mask = cv2.resize(mask, (cols, rows))
    #img = img_tensor + mask
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if  mask[y, x, 0] > 200.:
                frame[y, x, 0] = 254
    return frame


def predict_person(frame, interpreter, input_details, output_details):
    rows = frame.shape[0]
    cols = frame.shape[1]
    frame_tensor = get_tensor(frame)
    interpreter.set_tensor(input_details[0]['index'], frame_tensor)
    interpreter.invoke()
    tflite_result = interpreter.get_tensor(output_details[0]['index'])
    msk_predicted = create_mask(tflite_result)
    gray_img = predToGrayImage(msk_predicted)
    out_frame = segment(frame, gray_img, rows, cols)
    return out_frame
    