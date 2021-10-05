import os
from glob import glob
import cv2
import numpy as np
import tensorflow as tf
from deeplab.pascal_voc import VOC_COLORMAP, VOC_CLASSES
from deeplab.params import (
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_TRAIN_IMAGES,
    NUM_VAL_IMAGES,
    DATASET_DIR
)

train_data_dir = os.path.join(DATASET_DIR, "Training")
val_data_dir = os.path.join(DATASET_DIR, "Validation")

train_images = sorted(glob(os.path.join(train_data_dir, "images/*")))[:NUM_TRAIN_IMAGES]
train_masks = sorted(glob(os.path.join(train_data_dir, "annotations/*")))[:NUM_TRAIN_IMAGES]
val_images = sorted(glob(os.path.join(val_data_dir, "images/*")))[:NUM_VAL_IMAGES]
val_masks = sorted(glob(os.path.join(val_data_dir, "annotations/*")))[:NUM_VAL_IMAGES]


def _convert_to_segmentation_mask(mask_path):
    """
    :param mask_path: path to the VOC mask image
    :return: Single channel image. Pixel range (0 - 20)
    """
    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    height, width = mask.shape[:2]
    segmentation_mask = np.zeros((height, width), dtype=np.float32)
    for label_index, label in enumerate(VOC_COLORMAP):
        segmentation_mask += np.where(np.all(mask == label, axis=-1), label_index, 0).astype(float)
    # print(f'{mask_path} - classes:', [VOC_CLASSES[int(i)] for i in np.unique(segmentation_mask)])
    return segmentation_mask


def preprocess_image(im_path):
    image = cv2.imread(im_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, [IMAGE_SIZE[1], IMAGE_SIZE[0]], interpolation=cv2.INTER_AREA)
    image = image / 127.5 - 1
    return image


def generator_fn(image_list, mask_list):
    """Return a function that takes no arguments and returns a generator."""

    def generator():
        for im_path, mask_path in zip(image_list, mask_list):
            # Synthesize an image and a class label.
            mask = _convert_to_segmentation_mask(mask_path)
            mask = np.expand_dims(cv2.resize(mask, [IMAGE_SIZE[1], IMAGE_SIZE[0]], interpolation=cv2.INTER_AREA),
                                  axis=-1)
            image = preprocess_image(im_path)
            yield image, mask

    return generator


# ToDo: Add Augmentations
# def augment(image, mask):
#     return image, mask


def data_generator(image_list, mask_list):
    gen = generator_fn(image_list, mask_list)
    dataset = tf.data.Dataset.from_generator(gen,
                                             output_types=(tf.float32, tf.float32),
                                             output_shapes=((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), (IMAGE_SIZE[0], IMAGE_SIZE[1], 1)))
    # dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset


def load_dataset():
    train_dataset = data_generator(train_images, train_masks)
    val_dataset = data_generator(val_images, val_masks)

    print("Train Dataset:", train_dataset)
    print("Val Dataset:", val_dataset)

    return train_dataset, val_dataset


if __name__ == '__main__':
    mask_path = '../dataset/Validation/annotations/2007_000032.png'
    mask = _convert_to_segmentation_mask(mask_path)
    print(mask.shape)
    load_dataset()
