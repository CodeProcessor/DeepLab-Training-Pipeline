import glob
import os

import cv2
import numpy as np
import tensorflow as tf

from deeplab.params import (
    IMAGE_SIZE,
    BATCH_SIZE,
    SHUFFLE_BUFFER_SIZE,
    NUM_TRAIN_IMAGES,
    NUM_VAL_IMAGES,
    USE_TF_RECORDS,
    DATASET_DIR, train_txt_file_voc, val_txt_file_voc, TF_RECORDS_DIR, BACKBONE
)
from deeplab.pascal_voc import VOC_COLORMAP
from deeplab.preprocess import PreProcess
from deeplab.utils import AugmentationWrapper, post_process


def _get_image_list_from_file(filename):
    """
    Get the list of image names from the file
    :param filename: file path
    :return: list of images
    """
    with open(filename, 'r') as fp:
        image_list = [_line.strip() for _line in fp.readlines()]
    return image_list


def _get_image_lists():
    """
    Get all training and validation images and maks from directories
    :return:
    """
    train_image_list = _get_image_list_from_file(train_txt_file_voc)
    val_image_list = _get_image_list_from_file(val_txt_file_voc)
    all_trn_images = [os.path.join(DATASET_DIR, "JPEGImages", _im + '.jpg') for _im in train_image_list]
    all_val_images = [os.path.join(DATASET_DIR, "JPEGImages", _im + '.jpg') for _im in val_image_list]
    all_trn_masks = [os.path.join(DATASET_DIR, "SegmentationClass", _im + '.png') for _im in train_image_list]
    all_val_masks = [os.path.join(DATASET_DIR, "SegmentationClass", _im + '.png') for _im in val_image_list]
    return all_trn_images[:NUM_TRAIN_IMAGES], all_trn_masks[:NUM_TRAIN_IMAGES], all_val_images[:NUM_VAL_IMAGES], \
           all_val_masks[:NUM_VAL_IMAGES]


def _get_tfrecord_paths_train_val():
    """
    List down all TF records
    :return:
    """
    return [
        sorted(glob.glob(TF_RECORDS_DIR + '/train*.tfrecord')),
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
    """
    Read image do preprocess and return
    :param im_path:path to image
    :return: Normalized image
    """
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
    """
    Normal datagenerator with image list and mask list
    :param image_list: list of paths for images
    :param mask_list: list of paths for masks
    :return: TF Data dataset
    """
    gen = generator_fn(image_list, mask_list)
    dataset = tf.data.Dataset.from_generator(gen, output_types=(tf.float32, tf.float32)
                                             , output_shapes=(
            (IMAGE_SIZE[0], IMAGE_SIZE[1], 3), (IMAGE_SIZE[0], IMAGE_SIZE[1], 1)))
    dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE)
    dataset = dataset.map(AugmentationWrapper(), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset


def tfrecord_decode(tf_record):
    """
    Decode TF record and get image and mack back
    :param tf_record:
    :return: image and mask arrays
    """
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
    image = tf.io.decode_jpeg(sample['image/encoded'], 3)
    mask = tf.io.decode_jpeg(sample['image/segmentation/class/encoded'], 1)

    # image = tf.cast(image, tf.float32) * (1. / 127.5) - 1

    return image, mask


def data_generator_tf_records(record_paths, limit=-1, augmentations=True, backbone="resnet50",
                              batch_size=BATCH_SIZE) -> tf.data.TFRecordDataset:
    """
    Create data-generator with TF records
    :param record_paths: path to TF records
    :param limit: Data load limit
    :param augmentations: Augmentation enable or disable
    :param backbone: Backbone name
    :param batch_size: Batch size
    :return: Data-generator
    """
    ds = tf.data.TFRecordDataset([name for name in record_paths], num_parallel_reads=tf.data.AUTOTUNE) \
        .take(limit) \
        .shuffle(SHUFFLE_BUFFER_SIZE) \
        .map(tfrecord_decode, num_parallel_calls=tf.data.AUTOTUNE) \
        .map(PreProcess(IMAGE_SIZE, backbone), num_parallel_calls=tf.data.AUTOTUNE)

    if augmentations:
        ds = ds.map(AugmentationWrapper(), num_parallel_calls=tf.data.AUTOTUNE)

    return ds.batch(batch_size, drop_remainder=True).prefetch(1)


def load_dataset():
    """
    Main function to load the dataset

    :return: two data generators for training and validation
    """
    if USE_TF_RECORDS:
        train, val = _get_tfrecord_paths_train_val()
        print(f"Training records: {train}")
        print(f"Validation records: {val}")
        # Raise an error if the tf records are empty
        if len(val) == 0 or len(train) == 0:
            raise "Train or Val records cannot be empty"
        train_dataset = data_generator_tf_records(train, limit=NUM_TRAIN_IMAGES, backbone=BACKBONE)
        val_dataset = data_generator_tf_records(val, limit=NUM_VAL_IMAGES, backbone=BACKBONE, augmentations=False)
    else:
        train_images, train_masks, val_images, val_masks = _get_image_lists()
        train_dataset = data_generator(train_images, train_masks)
        val_dataset = data_generator(val_images, val_masks)

    print("Train Dataset:", train_dataset)
    print("Val Dataset:", val_dataset)

    return train_dataset, val_dataset


"""
This main method for test purposes
To quickly check and verify the train and validation images and masks
"""
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
            _image = cv2.cvtColor(post_process(cv2.cvtColor(_image.numpy(), cv2.COLOR_BGR2RGB)), cv2.COLOR_RGB2BGR)
            cv2.imwrite(_name, np.uint8(_image))
            print(f"Saved: {_name}")
        break
