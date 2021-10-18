"""
@Filename:    dataset_voc.py
@Author:      sgx team
@Time:        02/10/2021 08:01
source: https://keras.io/examples/vision/deeplabv3_plus/
"""
import ssl
import math

import tensorflow as tf
from tensorflow.keras import layers

from deeplab.custom_metrics import UpdatedMeanIoU
from deeplab.loss import loss_initializer_v2
from deeplab.params import IMAGE_SIZE, NUM_CLASSES, LEARNING_RATE

ssl._create_default_https_context = ssl._create_unverified_context


def convolution_block(
        block_input,
        num_filters=256,
        kernel_size=3,
        dilation_rate=1,
        padding="same",
        use_bias=False,
        kernel_seed=None
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding=padding,
        use_bias=use_bias,
        kernel_initializer=tf.keras.initializers.GlorotNormal(seed=kernel_seed),  # xavier normal initializer
        kernel_regularizer=tf.keras.regularizers.L2(0.01),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.elu(x)


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear", )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output


def DeeplabV3Plus(image_size, num_classes) -> tf.keras.Model:
    model_input = tf.keras.Input(shape=(image_size[0], image_size[1], 3))
    resnet50 = tf.keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size[0] // 4 // x.shape[1], image_size[1] // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size[0] // x.shape[1], image_size[1] // x.shape[2]),
        interpolation="bilinear")(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    return tf.keras.Model(inputs=model_input, outputs=model_output)


def CompileModel(model: tf.keras.Model):
    # Loss, optimizer and metrics
    # loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    metrics = ["accuracy", UpdatedMeanIoU(num_classes=NUM_CLASSES)]

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss=loss_initializer_v2,
        metrics=metrics
    )

    return model


def load_model(model_path, custom_objects=None):
    _deeplab_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    # _deeplab_model = tf.keras.models.load_model(model_path)
    return _deeplab_model


if __name__ == '__main__':
    model = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
    deeplab_model = CompileModel(model)
    print(model.summary())
