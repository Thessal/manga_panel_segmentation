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
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np


# +
def sort_contours(contours, rtl=True):
    # naively guess cell order
    scores = []
    for cont in contours:
        x,y,w,h = cv2.boundingRect(cont)
        xx = 2*x+w
        yy = 2*y+h
        score = -0.2*xx+yy if rtl else 0.2*xx+yy
        scores.append(score)
    
    contours = sorted(enumerate(contours), key=lambda i:scores[i[0]])
    return([x for i,x in contours])
    

def cell_mask(Dd, rtl=True, plot=False):
    """
    rtl : right to left sort 
    """
    
    # preprocessing
    kernel = np.ones((3,3),np.uint8)
    DdErode = 3;
    DdDilate = 3;
    Dd = cv2.erode(Dd,kernel,iterations = DdErode)
    Dd = cv2.dilate(Dd,kernel,iterations = DdDilate)
    
    if plot:
        plt.figure()
        plt.imshow(Dd)

    contours, hierarchy = cv2.findContours(Dd, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    # convexHull
    contours = [cv2.convexHull(cont) for cont in contours if cv2.contourArea(cont) >= 1000] 
    # polygon
#     epsilon = 5
#     contours = [cv2.approxPolyDP(cont,epsilon,True) for cont in contours if cv2.contourArea(cont) >=area_thres]

    # Sort order
    contours = sort_contours(contours, rtl=rtl)
    
    # Convert contour to mask
    Dd[:,:]=0
    for i, cont in enumerate(contours):
        cv2.drawContours(Dd, [cont], -1, 1+i, -1)
    if plot:
        plt.figure()
        plt.imshow(Dd)
        plt.colorbar()
    
    return Dd


def make_polygon(Dd, area_thres=300):
    
    # preprocessing
    kernel = np.ones((3,3),np.uint8)
    DdErode = 3;
    DdDilate = 3;
    Dd = cv2.erode(Dd,kernel,iterations = DdErode)
    Dd = cv2.dilate(Dd,kernel,iterations = DdDilate)
    
    # Todo : apply threshold using sum probability
    contours, hierarchy = cv2.findContours(Dd, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    contours = [cv2.approxPolyDP(cont,epsilon=0.01*cv2.arcLength(cont,True),closed=True) for cont in contours if cv2.contourArea(cont) >=area_thres]
    contours = [cv2.convexHull(cont) for cont in contours if cv2.contourArea(cont) >= area_thres] 
    
    # Convert contour to mask
    Dd[:,:]=0
    cv2.drawContours(Dd, contours, -1, 1, -1).astype(bool)
    
    return Dd
# plt.imshow(result_cell)
# cell_mask(result_cell.astype(np.uint8) ,plot=True)



# +
from const import IMAGE_SIZE, OUTPUT_CHANNELS

# TODO : refactor
def process_image(model, image_path, plot=False):
    if type(image_path)==str:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
    else:
        image = image_path
    input_image = tf.image.resize_with_pad(image, target_height=IMAGE_SIZE, target_width=IMAGE_SIZE)
    
    batch = tf.stack([input_image, ])
    result = model(batch)
    np_result = result.numpy()[0,:,:,:]
    np_max = np.max(np_result, axis=2) # TODO : adjust weight
    for i in range(OUTPUT_CHANNELS):
        np_result[:,:,i] = np.where(np_result[:,:,i] == np_max, 255, 0) 
        
    mask_background = np_result[:,:,0]
    mask_cell = np_result[:,:,1]
    mask_border = np_result[:,:,2]
    mask_text = np_result[:,:,3]
    mask_face = np_result[:,:,4]
    weights=np.mean(np_result,axis=(0,1))[np.newaxis,np.newaxis,:]
    _np_result= np_result/weights
    
    _result_background = (np.argmax(_np_result, axis=2) == 0)
    _result_border = (np.argmax(_np_result[:,:,:3], axis=2) == 2)
    _result_text = (np.argmax(np_result[:,:,[1,3]], axis=2) == 1)
    _result_face = (np.argmax(np_result[:,:,[1,4]], axis=2) == 1)
    result_cell = (~(_result_background|_result_border))
    
    mask_cell = cell_mask(result_cell.astype(np.uint8))
    mask_text = make_polygon(_result_text.astype(np.uint8)) & (mask_cell>0)
    for i in range(mask_cell.max()):
        # Remove too large text
        if mask_text[mask_cell==i].mean()>0.5:
            mask_text[mask_cell==i] = False
    mask_face = make_polygon(_result_face.astype(np.uint8)) & (mask_cell>0) & (~mask_text)
    for i in range(mask_cell.max()):
        if mask_face[mask_cell==i].mean()>0.5:
            mask_face[mask_cell==i] = False

    if plot:
        np_result = np.stack([
            mask_cell.astype(bool),
            mask_text.astype(bool),
            mask_face.astype(bool)
        ], axis=2)
        np_result = np_result / np_result.max() * 255
    
        plt.figure(figsize=(10,10))
        plt.imshow(image.numpy().astype(np.uint8))
        plt.figure(figsize=(10,10))
        plt.imshow(input_image.numpy().astype(np.uint8))
        plt.figure(figsize=(10,10))
        plt.imshow((np_result[:,:,:]/2+1).astype(np.uint8))
    return input_image, mask_cell, mask_text, mask_face


# -

if __name__=="__main__":
    from model import unet_model
    model = unet_model()
    model.load_weights("./model")
    
    with open("test_path.txt","r") as f:
        path = f.read()
    orig, mask_cell, mask_text, mask_face = process_image(model, image_path=path)
    
    demo_output = np.stack([mask_cell, mask_text, mask_face], axis=2)
    plt.figure(figsize=(10,10))
    plt.imshow(orig.numpy().astype(np.uint8))
    plt.figure(figsize=(10,10))
    plt.imshow(np.clip(demo_output.astype(float) / np.max(demo_output, axis=(0,1))[np.newaxis, np.newaxis, :],0,1))


