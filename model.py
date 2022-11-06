# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
from data_loader import manga109_dataloader
import tensorflow as tf
import numpy as np
# from examples.tensorflow_examples.models.pix2pix import pix2pix # use custom one

from const import IMAGE_SIZE, BORDER_LABEL, CONTENT_LABEL, BACKGROUND_LABEL, FACE_LABEL, TEXT_LABEL, OUTPUT_CHANNELS


# -

# unet_model, using InceptionV3 as base_model
def unet_model(output_channels=OUTPUT_CHANNELS):
    tf.keras.backend.clear_session()
    
    def upsample(filters, size, apply_dropout=False, strides=2, padding='valid'):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=strides,
                                        padding=padding, # 'same'
                                        kernel_initializer=initializer,
                                        use_bias=False))

        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
#             result.add(tf.keras.layers.Dropout(0.5))
            result.add(tf.keras.layers.Dropout(0.2))

        result.add(tf.keras.layers.ReLU())

        return result


    print(" - Creating the model")
    base_model = tf.keras.applications.inception_v3.InceptionV3(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=[IMAGE_SIZE, IMAGE_SIZE, 3],
        pooling=None,
        classes=1000,
        classifier_activation='leakyrelu'#'softmax'
    )
    
    layers = [base_model.get_layer(name).output for name in [
        'activation', #111, 111, 32
    #     'activation_1', #109, 109, 32
#         'activation_2', #109, 109, 64
    #     'activation_3', #54, 54, 80
        'activation_4', #52, 52, 192
    #     'activation_11', #25, 25, 32
    #     'activation_18', #25, 25, 64
        'activation_25', #25, 25, 64
    #     'activation_39', #12, 12, 192
    #     'activation_59', #12, 12, 192
        'activation_69', #12, 12, 192
    #     'activation_84', #5, 5, 192
    #     'activation_93', #5, 5, 192    
        'mixed10', #5, 5, 2048
    ]]

    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
    down_stack.trainable = False
    #     down_stack.trainable = False
    up_stack = [
    #     upsample(192, 1, apply_dropout=True, padding='valid', strides=1),
        upsample(192, 4, apply_dropout=True, padding='valid', strides=2),
        upsample(64, 3, apply_dropout=True, padding='valid', strides=2),
    #     upsample(32, 3, apply_dropout=True, padding='valid', strides=2),
        upsample(192, 4, apply_dropout=True, padding='valid', strides=2),
#         upsample(64, 7, apply_dropout=True, padding='valid', strides=2),
        upsample(64, 9, apply_dropout=True, padding='valid', strides=2),
    ]

    #     def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=[IMAGE_SIZE, IMAGE_SIZE, 3])
    x = inputs
    # print(x.shape)

    # Downsampling through the model
    skips = down_stack(x)
    x = skips[-1]
    # print(x.shape)
    skips = list(reversed(skips[:-1]))

    # Upsampling and establishing the skip connections
    assert(len(up_stack) == len(skips))
    for up, skip in zip(up_stack, skips):
        print(x.shape)
        print(up_stack)
        x = up(x)
        print(x.shape)
        concat = tf.keras.layers.Concatenate()
        print(x.shape, skip.shape)
        x = concat([x, skip])
        print("---")

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 4, strides=2,
        padding='valid')  # 64x64 -> 128x128
#         padding='same')  # 64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


# +
#     tf.keras.backend.clear_session()
#     base_model = tf.keras.applications.inception_v3.InceptionV3(
#         include_top=False,
#         weights='imagenet',
#         input_tensor=None,
#         input_shape=[IMAGE_SIZE, IMAGE_SIZE, 3],
#         pooling=None,
#         classes=1000,
#         classifier_activation='leakyrelu'#'softmax'
#     )
#     base_model.summary()

if __name__ == "__main__":    
    tmp_model = unet_model(OUTPUT_CHANNELS)
    tmp_model.summary()
    # tf.keras.utils.plot_model(tmp_model)
# -


