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

from data_loader import manga109_dataloader, load_image_train, tf_count
import tensorflow as tf
import numpy as np
from model import unet_model
from metrics import *


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('\n    - Training finished for epoch {}\n'.format(epoch + 1))


# +
if __name__ == "__main__":
    ## model
    model = unet_model()
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[
                  'accuracy',
                  border_acc,
                  content_acc,
                  background_acc,
                  iou_coef,
                  dice_coef])
    # tf.keras.utils.plot_model(model, show_shapes=True)
    
#     ## data
    dataloader = manga109_dataloader()
#     key, image, mask = next(dataloader.load_all())
#     assert(set(np.unique(mask.numpy())).issubset({29,76,134,149,255}))
    
    
    ## Pipelining
    train_raw_dataset = tf.data.Dataset.from_generator(
        lambda : dataloader.load_all(shuffle=False, train=True), 
        output_types=(tf.string, tf.uint8, tf.uint8), 
        output_shapes=(None, (None,None,3), (None,None,1))
    )
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train = train_raw_dataset.map(load_image_train, AUTOTUNE)
    
    TRAINING_BATCH_SIZE=20
    BUFFER_SIZE = 20
    EPOCHS = 10
    CORES_COUNT = 4
    TESTING_BATCH_SIZE = 100
    
    train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(TRAINING_BATCH_SIZE)
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = tf.data.Dataset.from_generator(
        lambda : dataloader.load_all(shuffle=False, train=False), 
        output_types=(tf.string, tf.uint8, tf.uint8), 
        output_shapes=(None, (None,None,3), (None,None,1))
    ).map(load_image_train).batch(TESTING_BATCH_SIZE)

    print(" - Starting training stage")
    model_history = model.fit(train_dataset,
                              epochs=EPOCHS,
                              validation_data=test_dataset,
                              use_multiprocessing=True,
                              workers=CORES_COUNT,
                              callbacks=[DisplayCallback()])

    print(" - Training finished, saving metrics into ./graphs")
    save_model_history_metrics(EPOCHS, model_history)
    print(" - Training finished, saving model into ./model")
    output_path = "./model"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    model.save(output_path)
    print(" - Model updated and saved")
    
    ## infer
    key, image, mask = next(dataloader.load_all())
    input_image, input_mask, weights = load_image_train(key, image, mask)
    #     print(input_image.shape, input_mask.shape, weights.shape)
    batch = tf.stack([input_image, ])
    result = model(batch)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,10))
    np_result = result.numpy()[0,:,:,:]
    np_max = np.max(np_result, axis=2) # TODO : adjust weight
    np_result[:,:,0] = np.where(np_result[:,:,0] == np_max, 255, 0) 
    np_result[:,:,1] = np.where(np_result[:,:,1] == np_max, 255, 0)
    np_result[:,:,2] = np.where(np_result[:,:,2] == np_max, 255, 0)
    plt.imshow((np_result[:,:,:]/2+1).astype(np.uint8))
# -



# +
#     key, image, mask = next(dataloader.load_all())
#     assert(set(np.unique(mask.numpy())).issubset({29,76,134,149,255}))

    ## dataset test
#     train_batches = ds.take(10)
#     for x in train_batches.batch(2).enumerate():
#         print(x)


# -








