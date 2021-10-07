import os

import cv2
import numpy as np
import tensorflow as tf

from deeplab.augmentation import Augment
from deeplab.params import (
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_TRAIN_IMAGES,
    NUM_VAL_IMAGES,
    DATASET_DIR, train_txt_file_voc, val_txt_file_voc
)
from deeplab.pascal_voc import VOC_COLORMAP


def _get_image_list_from_file(filename):
    with open(filename, 'r') as fp:
        image_list = [_line.strip() for _line in fp.readlines()]
    return image_list


def _get_image_lists():
    train_image_list = _get_image_list_from_file(train_txt_file_voc)
    val_image_list = _get_image_list_from_file(val_txt_file_voc)
    all_trn_images = [os.path.join(DATASET_DIR, "JPEGImages", _im + '.jpg') for _im in train_image_list]
    all_val_images = [os.path.join(DATASET_DIR, "JPEGImages", _im + '.jpg') for _im in val_image_list]
    all_trn_masks = [os.path.join(DATASET_DIR, "SegmentationClass", _im + '.png') for _im in train_image_list]
    all_val_masks = [os.path.join(DATASET_DIR, "SegmentationClass", _im + '.png') for _im in val_image_list]
    return all_trn_images[:NUM_TRAIN_IMAGES], all_trn_masks[:NUM_TRAIN_IMAGES], all_val_images[:NUM_VAL_IMAGES], \
           all_val_masks[:NUM_VAL_IMAGES]


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
    return segmentation_mask


def read_image(im_path):
    image = cv2.imread(im_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_AREA)
    image = image / 127.5 - 1
    return image


def generator_fn(image_list, mask_list):
    """Return a function that takes no arguments and returns a generator."""

    def generator():
        for im_path, mask_path in zip(image_list, mask_list):
            mask = _convert_to_segmentation_mask(mask_path)
            mask = cv2.resize(mask, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_AREA)
            image = read_image(im_path)
            yield image, np.expand_dims(mask, axis=-1)

    return generator


def data_generator(image_list, mask_list):
    gen = generator_fn(image_list, mask_list)
    dataset = tf.data.Dataset.from_generator(gen, output_types=(tf.float32, tf.float32)
                                             , output_shapes=(
            (IMAGE_SIZE[0], IMAGE_SIZE[1], 3), (IMAGE_SIZE[0], IMAGE_SIZE[1], 1)))
    dataset = dataset.map(Augment(), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset


def load_dataset():
    train_images, train_masks, val_images, val_masks = _get_image_lists()
    train_dataset = data_generator(train_images, train_masks)
    val_dataset = data_generator(val_images, val_masks)

    print("Train Dataset:", train_dataset)
    print("Val Dataset:", val_dataset)

    return train_dataset, val_dataset


if __name__ == '__main__':
    # mask_path = '../dataset/Validation/annotations/2007_000032.png'
    # mask = _convert_to_segmentation_mask(mask_path)
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)
    # print(mask.shape)
    # print(f'{mask_path} - classes:', [VOC_CLASSES[int(i)] for i in np.unique(mask)])

    _train_dataset, _val_dataset = load_dataset()
    for element in _train_dataset:
        # print(element)
        for i, _mask in enumerate(element[1]):
            _name = f"mask_{i}.jpg"
            cv2.imwrite(_name, np.uint8(_mask.numpy() * 20.))
            print(f"Saved: {_name}")
        for i, _image in enumerate(element[0]):
            _name = f"image_{i}.jpg"
            cv2.imwrite(_name, np.uint8(_image.numpy() * 255.))
            print(f"Saved: {_name}")
        break
