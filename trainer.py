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

from data_loader import manga109_dataloader
import tensorflow as tf
import numpy as np
from model import unet_model

# +
# image.map(load_image_train)

IMAGE_SIZE = 224
BACKGROUND_LABEL = 0
BORDER_LABEL = 1
CONTENT_LABEL = 2
OUTPUT_CHANNELS = 3


# +
def tf_count(t, val):
    elements_equal_to_value = tf.equal(t, val)
    as_ints = tf.cast(elements_equal_to_value, tf.int32)
    count = tf.reduce_sum(as_ints)
    return count

@tf.function
def load_image_train(datapoint):
    # https://github.com/pedrovgs/DeepPanel
    mask = datapoint['segmentation_mask']
    mask = tf.where(mask == 255, np.dtype('uint8').type(BACKGROUND_LABEL), mask)
    # Dark values will use label the background label
    mask = tf.where(mask == 29, np.dtype('uint8').type(BACKGROUND_LABEL), mask)
    # Intermediate values will act as the border
    mask = tf.where(mask == 76, np.dtype('uint8').type(BORDER_LABEL), mask)
    mask = tf.where(mask == 134, np.dtype('uint8').type(BORDER_LABEL), mask)
    # Brighter values will act as the content
    mask = tf.where(mask == 149, np.dtype('uint8').type(CONTENT_LABEL), mask)

    # https://github.com/pedrovgs/DeepPanel
    input_image = tf.image.resize_with_pad(datapoint['image'], target_height=IMAGE_SIZE, target_width=IMAGE_SIZE)
    input_mask = tf.image.resize_with_pad(mask, target_height=IMAGE_SIZE,
                                          target_width=IMAGE_SIZE)
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)
    # input_image, input_mask = normalize(input_image, input_mask)
    number_of_pixels_per_image = IMAGE_SIZE * IMAGE_SIZE
    percentage_of_background_labels = tf_count(input_mask, BACKGROUND_LABEL) / number_of_pixels_per_image
    percentage_of_content_labels = tf_count(input_mask, CONTENT_LABEL) / number_of_pixels_per_image
    percentage_of_border_labels = tf_count(input_mask, BORDER_LABEL) / number_of_pixels_per_image
    background_weight = tf.cast(0.33 / percentage_of_background_labels, tf.float32)
    content_weight = tf.cast(0.34 / percentage_of_content_labels, tf.float32)
    border_weight = tf.cast(0.33 / percentage_of_border_labels, tf.float32)
    weights = tf.where(input_mask == BACKGROUND_LABEL, background_weight, input_mask)
    weights = tf.where(input_mask == BORDER_LABEL, border_weight, weights)
    weights = tf.where(input_mask == CONTENT_LABEL, content_weight, weights)
    return input_image, input_mask, weights
# -



if __name__ == "__main__":
    dataloader = manga109_dataloader()
    
    ## Network test
    key, image, mask = next(dataloader.load_all())
    datapoint = {"image": image, "segmentation_mask": mask}
    print(set(np.unique(datapoint['segmentation_mask'].numpy())))
    assert(set(np.unique(datapoint['segmentation_mask'].numpy())).issubset({29,76,134,149,255}))
    input_image, input_mask, weights = load_image_train(datapoint)
    
    ## Pipelining
    ds = tf.data.Dataset.from_generator(
    dataloader.load_all, 
    output_types=(tf.string, tf.uint8, tf.uint8), 
    output_shapes=(None, (None,None,3), (None,None,1)))
    
    unet_model = unet_model(OUTPUT_CHANNELS)
    
    ## dataset test
    train_batches = ds.take(10)
    for x in train_batches.batch(2).enumerate():
        print(x)
