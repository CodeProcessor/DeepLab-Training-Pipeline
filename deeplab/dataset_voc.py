import os

import cv2
import numpy as np
import tensorflow as tf
import glob

from deeplab.augmentation import Augment
from deeplab.params import (
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_TRAIN_IMAGES,
    NUM_VAL_IMAGES,
    USE_TF_RECORDS,
    DATASET_DIR, train_txt_file_voc, val_txt_file_voc, TF_RECORDS_DIR
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


def _get_tfrecord_paths_train_val():
    return [
        sorted(glob.glob(TF_RECORDS_DIR + '/train-*.tfrecord')),
        sorted(glob.glob(TF_RECORDS_DIR + '/val-*.tfrecord'))
    ]


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
    image = image / 255.  # 127.5 - 1
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


def tfrecord_decode(tf_record):
    features = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/format': tf.io.FixedLenFeature([], tf.string),
        'image/height': tf.io.FixedLenFeature([1], tf.int64),
        'image/width': tf.io.FixedLenFeature([1], tf.int64),
        'image/channels': tf.io.FixedLenFeature([1], tf.int64),
        'image/segmentation/class/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/segmentation/class/format': tf.io.FixedLenFeature([], tf.string),
    }
    sample = tf.io.parse_single_example(tf_record, features)
    raw_image = tf.io.decode_jpeg(sample['image/encoded'], 3)

    raw_height = tf.cast(sample['image/height'], tf.int32)
    raw_width = tf.cast(sample['image/width'], tf.int32)

    image = tf.image.resize(raw_image, size=IMAGE_SIZE)
    image = tf.cast(image, tf.float32) * (1. / 127.5) - 1
    raw_mask = tf.io.decode_jpeg(sample['image/segmentation/class/encoded'], 1)

    mask = tf.image.resize(raw_mask, size=IMAGE_SIZE)

    return image, mask


def data_generator_tf_records(record_paths, limit=-1, augmentations=True, batch_size=BATCH_SIZE) -> tf.data.TFRecordDataset:
    ds = tf.data.TFRecordDataset([name for name in record_paths]) \
        .map(tfrecord_decode, num_parallel_calls=tf.data.AUTOTUNE) \
        .prefetch(limit) \
        .cache()
    if augmentations:
        ds = ds.map(Augment(), num_parallel_calls=tf.data.AUTOTUNE) \
                .prefetch(BATCH_SIZE)
    return ds.batch(BATCH_SIZE, drop_remainder=True)


def load_dataset():
    if USE_TF_RECORDS:
        train, val = _get_tfrecord_paths_train_val()
        train_dataset = data_generator_tf_records(train, limit=NUM_TRAIN_IMAGES)
        val_dataset = data_generator_tf_records(val, limit=NUM_VAL_IMAGES, augmentations=False, batch_size=len(val))
    else:
        train_images, train_masks, val_images, val_masks = _get_image_lists()
        train_dataset = data_generator(train_images, train_masks)
        val_dataset = data_generator(val_images, val_masks)

    print("Train Dataset:", train_dataset)
    print("Val Dataset:", val_dataset)

    return train_dataset, val_dataset


if __name__ == '__main__':
    _train_dataset, _val_dataset = load_dataset()
    for element in _train_dataset:
        print(element[0].shape, element[1].shape)
        for i, _mask in enumerate(element[1]):
            _name = f"mask_{i}.jpg"
            cv2.imwrite(_name, np.uint8(_mask.numpy() * 20.))
            print(f"Saved: {_name}")
        for i, _image in enumerate(element[0]):
            _name = f"image_{i}.jpg"
            _image = cv2.cvtColor(_image.numpy(), cv2.COLOR_RGB2BGR)
            cv2.imwrite(_name, np.uint8(_image * 255.))  # (_image + 1) * 127.5
            print(f"Saved: {_name}")
        break
