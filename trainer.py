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
from const import OUTPUT_CHANNELS
from metrics import *


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('\n    - Training finished for epoch {}\n'.format(epoch + 1))

# +
# dataloader = manga109_dataloader()
# for i in range(1000):
#         print(i)
#         key, image, mask = next(dataloader.load_all())
#         assert(set(np.unique(mask.numpy())).issubset({29,76,134,149,255,226,105}))

# print(key)
# # plt.figure(figsize=(20,20))
# # plt.imshow(image.numpy())

# plt.figure(figsize=(20,20))
# plt.imshow(mask.numpy())


# # set(np.unique(mask.numpy()))
# dataloader = manga109_dataloader()
# #0.3*R + 0.59*G + 0.11*B
# #76.5 + 150.45 + 28.05
# import json
# book = "HanzaiKousyouninMinegishiEitarou"
# index = "95"
# side = "R"
# a = {x["book"]+"_"+x["index"]:x for x in [json.loads(x) for x in dataloader.load_page_info()]}
# tf_img, tf_mask = dataloader.load(book=book,index=index,side=side,
#                 frames=a[book+"_"+index]["frames"],
#                 faces=a[book+"_"+index]["faces"],
#                 texts=a[book+"_"+index]["texts"],
#                )
# # print(key)


# plt.figure(figsize=(20,20))
# plt.imshow(tf_mask.numpy())

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
                  face_acc,
                  text_acc,
                  iou_coef,
                  dice_coef])
    # tf.keras.utils.plot_model(model, show_shapes=True)
    
    ## data
    ## Be careful, exception will cause nan loss
    dataloader = manga109_dataloader()
#     for i in range(1000):
#         print(i)
#         key, image, mask = next(dataloader.load_all())
#         assert(set(np.unique(mask.numpy())).issubset({29,76,134,149,255,226,105}))
        
    
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
    
#     EPOCHS = 1 # debug
#     train_dataset = train_raw_dataset.take(10).map(load_image_train).batch(1) # debug
#     test_dataset = train_raw_dataset.take(10).map(load_image_train).batch(1) # debug

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
    for i in range(OUTPUT_CHANNELS):
        np_result[:,:,i] = np.where(np_result[:,:,i] == np_max, 255, 0) 
    np_result = np.stack([
        np_result[:,:,0]+np_result[:,:,3]+np_result[:,:,4], 
        np_result[:,:,1]+np_result[:,:,3],
        np_result[:,:,2]+np_result[:,:,4], 
    ], axis=2)
    np_result = np_result / np_result.max() * 255
    plt.imshow((np_result[:,:,:]/2+1).astype(np.uint8))
# +
# batch = tf.stack([input_image, ])
# result = model(batch)
# np_result = result.numpy()[0,:,:,:]
# np_result
# -

plt.figure(figsize=(10,10))
plt.imshow(input_image.numpy().astype(np.uint8))
plt.savefig('demo-orig.png')
plt.figure(figsize=(10,10))
plt.imshow((np_result[:,:,:]/2+1).astype(np.uint8))
plt.savefig('demo.png')






