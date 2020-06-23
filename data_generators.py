from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import utils


def train_generator(img_train_dir, mask_train_dir, target_size=(224, 224), batch_size=10):
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=20,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    img_train_generator = train_datagen.flow_from_directory(
        img_train_dir,
        target_size=target_size,
        batch_size=batch_size,
        seed=1,
        class_mode=None, )
    mask_train_generator = train_datagen.flow_from_directory(
        mask_train_dir,
        target_size=target_size,
        batch_size=batch_size,
        color_mode="grayscale",
        seed=1,
        class_mode=None, )
    train_gen = zip(img_train_generator, mask_train_generator)
    for (img, mask) in train_gen:
        batch_mask = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], 2))
        for i in range(batch_size):
            batch_mask[i] = utils.OHE(mask[i])
        yield (img, batch_mask)


def val_generator(img_val_dir, mask_val_dir, target_size=(224, 224), batch_size=10):
    val_datagen = ImageDataGenerator(rescale=1. / 255)
    img_val_generator = val_datagen.flow_from_directory(
        img_val_dir,
        target_size=target_size,
        batch_size=batch_size,
        seed=1,
        class_mode=None, )
    mask_val_generator = val_datagen.flow_from_directory(
        mask_val_dir,
        target_size=target_size,
        batch_size=batch_size,
        color_mode="grayscale",
        seed=1,
        class_mode=None, )
    val_gen = zip(img_val_generator, mask_val_generator)
    for (img, mask) in val_gen:
        batch_mask = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], 2))
        for i in range(batch_size):
            batch_mask[i] = utils.OHE(mask[i])
        yield (img, batch_mask)


def test_generator(img_test_dir, mask_test_dir, target_size=(224, 224), batch_size=2):
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_img_generator = test_datagen.flow_from_directory(
        img_test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=None
    )
    test_mask_generator = test_datagen.flow_from_directory(
        mask_test_dir,
        target_size=target_size,
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode=None
    )
    test_gen = zip(test_img_generator, test_mask_generator)
    for (img, mask) in test_gen:
        batch_mask = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], 2))
        for i in range(batch_size):
            batch_mask[i] = utils.OHE(mask[i])
        yield (img, batch_mask)
