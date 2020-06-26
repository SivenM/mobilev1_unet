import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import matplotlib.pylab as plt
from tensorflow.keras.preprocessing import image
import metrics


def let_fit(train_gen, val_gen, model, logdir, chkpt_path, train_steps=1165, train_epochs=70, val_steps=180):
    callbacks_list = [
        TensorBoard(
            log_dir=logdir,
            histogram_freq=1,
        ),
        ModelCheckpoint(
            filepath=chkpt_path,
            monitor='val_loss',
            save_best_only=True,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,  # уменьшение скорости в 10 раз
            patience=5,
        )
    ]

    history = model.fit_generator(
        train_gen,
        steps_per_epoch=train_steps,
        epochs=train_epochs,
        callbacks=callbacks_list,
        validation_data=val_gen,
        validation_steps=val_steps
    )
    return history


def OHE(mask):
    gt = np.zeros((mask.shape[0], mask.shape[1], 2))
    np.place(gt[:, :, 0], mask[:, :, 0] < 0.5, 1)
    np.place(gt[:, :, 1], mask[:, :, 0] < 0.5, 0)
    np.place(gt[:, :, 0], mask[:, :, 0] >= 0.5, 0)
    np.place(gt[:, :, 1], mask[:, :, 0] >= 0.5, 1)
    return gt


def visualise(img, gt, predicted=0):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set(title='Input image')
    ax2.imshow(gt, cmap='gray')
    ax2.axis('off')
    ax2.set(title='Ground true')
    if predicted != 0:
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.imshow(predicted)
        ax3.axis('off')
        ax3.set(title='Predict')
    plt.show()


def get_img(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    return img


def get_img_mask(img_path, msk_path):
    img = image.load_img(img_path, target_size=(224, 224))
    msk = image.load_img(msk_path, target_size=(224, 224))
    return img, msk


def preproc_to_model(img, msk):
    img_tensor = image.img_to_array(img)
    img_tensor /= 255.
    img_tensor = np.expand_dims(img_tensor, axis=0)

    msk_tensor = image.img_to_array(msk)
    msk_tensor = tf.image.rgb_to_grayscale(msk_tensor)
    msk_tensor /= 255.
    msk_tensor = OHE(msk_tensor)
    msk_tensor = np.expand_dims(msk_tensor, axis=0)
    return img_tensor, msk_tensor


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def predToGrayImage(predicted):
    img = np.zeros((predicted.shape[0], predicted.shape[1], 3))
    img[:, :, 0] = predicted[:, :, 0] * 254
    img[:, :, 1] = img[:, :, 0]  # Make greyscale
    img[:, :, 2] = img[:, :, 0]  # Make greyscale
    return img.astype('uint8')


def predict_and_evaluate(img_path, msk_path, model):
    img, msk_true = get_img_mask(img_path, msk_path)
    img_tensor, msk_tensor = preproc_to_model(img, msk_true)
    result = model.predict(img_tensor)
    print(result.shape)
    print()
    loss_and_dice = model.evaluate(img_tensor, msk_tensor)
    print("\n======================================================================================\n")
    print("Dice: {}\n".format(loss_and_dice[1]))
    dice_test = metrics.dice_coef(msk_tensor, msk_tensor)
    print("Метрика для ground true: {}".format(dice_test))
    msk_predicted = create_mask(result)
    msk_predicted = predToGrayImage(msk_predicted)
    visualise(img, msk_true, msk_predicted)
