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
# Docker host : Ubuntu 22.04, 5.15.0-50-generic, Driver Version: 520.61.05    CUDA Version: 11.8
# Docker image : nvcr.io/nvidia/tensorflow:22.04-tf2-py3
# tf.__version__ = 2.8.0
# tfa.__version__ = 0.16.1

import os
import tensorflow as tf
import tensorflow_addons as tfa
import examples.tensorflow_examples
from examples.tensorflow_examples.models.pix2pix import pix2pix
import xml
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy
import tensorflow_addons as tfa
import math
import random
import json
et = xml.etree.ElementTree
# -

# Reference : 
# Manga109
"""
@article{mtap_matsui_2017,
    author={Yusuke Matsui and Kota Ito and Yuji Aramaki and Azuma Fujimoto and Toru Ogawa and Toshihiko Yamasaki and Kiyoharu Aizawa},
    title={Sketch-based Manga Retrieval using Manga109 Dataset},
    journal={Multimedia Tools and Applications},
    volume={76},
    number={20},
    pages={21811--21838},
    doi={10.1007/s11042-016-4020-z},
    year={2017}
}

@article{multimedia_aizawa_2020,
    author={Kiyoharu Aizawa and Azuma Fujimoto and Atsushi Otsubo and Toru Ogawa and Yusuke Matsui and Koki Tsubota and Hikaru Ikuta},
    title={Building a Manga Dataset ``Manga109'' with Annotations for Multimedia Applications},
    journal={IEEE MultiMedia},
    volume={27},
    number={2},
    pages={8--18},
    doi={10.1109/mmul.2020.2987895},
    year={2020}
}
"""


class manga109_dataloader:
    def __init__(self, skip_empty=True, path="Manga109s_released_2021_12_30"):
        with open(f"{path}/books.txt", "r") as f:
            self.books = [x.strip() for x in f.readlines()]
        self.path_annotations = f"{path}/annotations/"
        self.path_images = f"{path}/images/"
        self.skip_empty = skip_empty
    
    @staticmethod
    def make_mask(image_shape, frames, faces, texts):
        np_mask = np.zeros(image_shape, dtype=np.uint8)
        np_mask[:,:,2] = 255
        y_siz, x_siz = np_mask.shape[0:2]
        face_mask = np.zeros(np_mask.shape[0:2], dtype=bool)
        text_mask = np.zeros(np_mask.shape[0:2], dtype=bool)
        border_mask = np.zeros(np_mask.shape[0:2], dtype=bool)
        inside_mask = np.zeros(np_mask.shape[0:2], dtype=bool)
        r = 10
        kernel = np.fromfunction(lambda x, y: ((x-r)**2 + (y-r)**2 <= r**2)*1, (2*r+1, 2*r+1), dtype=int).astype(np.uint8)

        border_mask[:,:] = False
        for frame in frames:
            y0, y1, x0, x1 = int(frame["ymin"]), int(frame["ymax"]),int(frame["xmin"]), int(frame["xmax"])
            y0, y1, x0, x1 = max(r,y0), min(y_siz-r,y1), max(r,x0), min(x_siz-r,x1)
            if (y0+r < y1-r) and (x0+r < x1-r):
                inside_mask[y0:y1, x0:x1]=False
                border_mask[y0:y1, x0:x1]=True
#                 border_mask[y0+r:y1-r, x0+r:x1-r]=False
#                 inside_mask[y0+r:y1-r, x0+r:x1-r]=True
                inside_mask[y0+r:y1-r, x0+r:x1-r]=True
        for face in faces:
            y0, y1, x0, x1 = int(face["ymin"]), int(face["ymax"]),int(face["xmin"]), int(face["xmax"])
            face_mask[y0:y1, x0:x1]=True
        for text in texts:
            y0, y1, x0, x1 = int(text["ymin"]), int(text["ymax"]),int(text["xmin"]), int(text["xmax"])
            text_mask[y0:y1, x0:x1]=True
            

        np_mask[:,:,1] = 255 * (scipy.ndimage.binary_dilation(border_mask, kernel)^inside_mask)
        np_mask[:,:,0] = 255 * inside_mask
        np_mask[:,:,2] = np.where(np.sum(np_mask[:,:,[0,1]], axis=2),0,255)
        # np_mask[:,:,[1,2]] = np.where(((body_mask&inside_mask))[:,:,np.newaxis],255,np_mask[:,:,[1,2]])
        _inside_mask = (np_mask[:,:,0] == 255) & (np_mask[:,:,1] == 0) & (np_mask[:,:,2] == 0)
        np_mask[:,:,1] = np.where((face_mask&_inside_mask)&(~text_mask), 255, np_mask[:,:,1])
        np_mask[:,:,2] = np.where(text_mask&_inside_mask, 255, np_mask[:,:,2])
        ## Be careful, exception may cause nan loss
        
        return tf.convert_to_tensor(np_mask)

    @staticmethod
    def augment(image, mask, r=10):
        # TODO : add chanel for speech bubble
        # TODO : consider adding image size/shift noise
        max_angle = 3
        border_px = r * 2
        degrees = np.random.random(1)[0] * max_angle * 2 - max_angle
        fill_value = np.random.random(1)[0] * 255
        image = tfa.image.rotate(
            image, 
            angles=degrees * math.pi / 180, 
            interpolation="bilinear",
            fill_mode = 'constant',
            fill_value = fill_value
        )
        mask = tfa.image.rotate(
            mask, 
            angles=degrees * math.pi / 180, 
            interpolation="nearest",
            fill_mode = 'constant',
            fill_value = 0
        )
        # low frequency noise
        noise_freq = int(((image.shape[0] + image.shape[1]) / 5) * (1 + np.random.random(1)[0]))
        background = tf.random.uniform(shape=[x//noise_freq for x in image.shape[:2]]+[1], minval=0, maxval=255, dtype=tf.int32)
        background = tf.image.resize(background, image.shape[0:2], method='bilinear',#method='gaussian',
            preserve_aspect_ratio=False,
            antialias=False,
        )
        # image
        background = tf.repeat(background,3,axis=2)
        image = tfa.image.blend(image, background, 0.3)
        image = tf.dtypes.cast(image, tf.uint8)
        image = tf.image.random_brightness(image, 0.2)
        if np.random.random(1)[0] > 0.5:
            image = tfa.image.gaussian_filter2d(image, (5,5), sigma = 5)
        # mask
        np_mask = mask.numpy()
        np_mask[:,:,2] = np.where( np_mask.sum(axis=2) == 0 , 255 , np_mask[:,:,2] )
        mask = tf.convert_to_tensor(np_mask)
        # handle panel on border
        np_border = np.ones(np_mask.shape[:2], dtype=bool)
        np_border[border_px:np_mask.shape[0]-border_px, border_px:np_mask.shape[1]-border_px] = False
        _outside = (np_mask[:,:,0] == 0) & (np_mask[:,:,1] == 0) & (np_mask[:,:,2] == 255)
        np_border = (~_outside) & (np_border)
        np_mask[:,:,0] = np.where( np_border , 0 , np_mask[:,:,0] )
        np_mask[:,:,1] = np.where( np_border , 255 , np_mask[:,:,1] )
        np_mask[:,:,2] = np.where( np_border , 0 , np_mask[:,:,2] )
        mask = tf.convert_to_tensor(np_mask)
        
        return image, mask
    
    @staticmethod
    def _get_info(book, page, side):
        faces = sorted([x.attrib for x in page if x.tag=="face"], key=lambda x: x["id"], reverse=False)
        texts = sorted([x.attrib for x in page if x.tag=="text"], key=lambda x: x["id"], reverse=False)
        frames = sorted([x.attrib for x in page if x.tag=="frame"], key=lambda x: x["id"], reverse=False)
        info = {
            "book":book, "index":page.attrib["index"], "frames":[dict(frame) for frame in frames], 
            "faces":faces, "texts":texts, "side":side}
        return info

    def load_all(self, shuffle=True, train=True):
        q = []
        for book in self.books:
            with open(self.path_annotations+"/"+book+".xml", 'r') as f:
                annotation = et.fromstring(f.read())
            pages = annotation[1]
            for page in pages:
                for side in ["L", "R"]:
                    q.append((book, page, side))
        if train:
            q = [item for index, item in enumerate(q) if index % 10 != 0]
        else : # test
            q = [item for index, item in enumerate(q) if index % 10 == 0]
                
        if shuffle:
            random.shuffle(q)
            
        for book, page, side in q:
            info = self._get_info(book, page, side)
            image, mask = self.load(**info)
            if (image == None) or (mask==None):
                continue
            key = f"{book}_{str(page.attrib['index'])}_{side}"
            yield key, image, mask
    
    def load_page_info(self):
        for book in self.books:
            with open(self.path_annotations+"/"+book+".xml", 'r') as f:
                annotation = et.fromstring(f.read())
            pages = annotation[1]
            for page in pages:
                for side in ["L", "R"]:
                    info = self._get_info(book, page, side)
                    if len(info["frames"])> 0 :
                        yield json.dumps(info)
    
    def load(self, book, index, frames, faces, texts, side, grayscale=True):
        pagenum = index
        image_path = self.path_images+"/"+book+f"/{('000'+pagenum)[-3:]}.jpg"
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        
        if self.skip_empty and (len(frames) == 0):
            return None, None
        mask = self.make_mask(image.shape,frames, faces, texts)
        image, mask = self.augment(image, mask)
        if grayscale:
            mask = tf.image.rgb_to_grayscale(mask)
        
        image_width = image.shape[1]
        if side=="L":
            return image[:,:image_width//2,:], mask[:,:image_width//2,:]
        elif side=="R":
            return image[:,image_width//2:,:], mask[:,image_width//2:,:]
        else:
            return image, mask

# +
from const import IMAGE_SIZE, BORDER_LABEL, CONTENT_LABEL, BACKGROUND_LABEL, FACE_LABEL, TEXT_LABEL, OUTPUT_CHANNELS

def tf_count(t, val):
    elements_equal_to_value = tf.equal(t, val)
    as_ints = tf.cast(elements_equal_to_value, tf.int32)
    count = tf.reduce_sum(as_ints)
    return 1+count # +1 to prevent nan

@tf.function
def load_image_train(key, image, mask):
    # https://github.com/pedrovgs/DeepPanel
    mask = tf.where(mask == 255, np.dtype('uint8').type(BACKGROUND_LABEL), mask)
    # Dark values will use label the background label
    mask = tf.where(mask == 29, np.dtype('uint8').type(BACKGROUND_LABEL), mask)
    # Intermediate values will act as the border
    mask = tf.where(mask == 76, np.dtype('uint8').type(BORDER_LABEL), mask)
    mask = tf.where(mask == 134, np.dtype('uint8').type(BORDER_LABEL), mask)
    # Brighter values will act as the content
    mask = tf.where(mask == 149, np.dtype('uint8').type(CONTENT_LABEL), mask)
    # 
    mask = tf.where(mask == 226, np.dtype('uint8').type(FACE_LABEL), mask)
    mask = tf.where(mask == 105, np.dtype('uint8').type(TEXT_LABEL), mask)

    # https://github.com/pedrovgs/DeepPanel
    input_image = tf.image.resize_with_pad(image, target_height=IMAGE_SIZE, target_width=IMAGE_SIZE)
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
    percentage_of_text_labels = tf_count(input_mask, TEXT_LABEL) / number_of_pixels_per_image
    percentage_of_face_labels = tf_count(input_mask, FACE_LABEL) / number_of_pixels_per_image
    background_weight = tf.cast(0.2 / percentage_of_background_labels, tf.float32)
    content_weight = tf.cast(0.2 / percentage_of_content_labels, tf.float32)
    border_weight = tf.cast(0.2 / percentage_of_border_labels, tf.float32)
    text_weight = tf.cast(0.2 / percentage_of_text_labels, tf.float32)
    face_weight = tf.cast(0.2 / percentage_of_face_labels, tf.float32)
    weights = tf.where(input_mask == BACKGROUND_LABEL, background_weight, input_mask)
    weights = tf.where(input_mask == BORDER_LABEL, border_weight, weights)
    weights = tf.where(input_mask == CONTENT_LABEL, content_weight, weights)
    weights = tf.where(input_mask == TEXT_LABEL, text_weight, weights)
    weights = tf.where(input_mask == FACE_LABEL, face_weight, weights)
    
    return input_image, input_mask, weights


# -

if __name__ == "__main__":
    def show(image,mask):
        plt.figure()
        plt.imshow(image.numpy())
        plt.figure()
        # plt.figure(figsize=(20,20))
        plt.imshow(mask.numpy())
        
    loader = manga109_dataloader()
    
    dataset = list(loader.load_page_info())
    info = json.loads(dataset[10])
    info["grayscale"] = False
    image, mask = loader.load(**info)
    show(image, mask)
    
    img_mask = loader.load_all()
    for _ in range(3):
        key, image, mask = next(img_mask)
        show(image, mask)
        plt.colorbar()






