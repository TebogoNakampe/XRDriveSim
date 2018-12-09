
# coding: utf-8

# In[212]:

import numpy as np
import mxnet as mx
import time
import pandas as pd

import cv2

import logging
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout


# In[213]:

# import matplotlib.pyplot as plt
# get_ipython().magic(u'matplotlib inline')


# In[214]:

# Load the trained model
# img_w, img_h = 200, 200
# checkpoint = 210
img_w, img_h = 64, 64
checkpoint = 390

sym, arg_params, aux_params = mx.model.load_checkpoint('models/chkpt', checkpoint)
model = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
model.bind(for_training=False, data_shapes=[('data', (1,3,img_w,img_h))], 
         label_shapes=model._label_shapes)
model.set_params(arg_params, aux_params, allow_missing=True)


# In[215]:

# Load the gesture mappings:
import json

num_to_ges = None
with open('num2ges.json') as fin:
    num_to_ges = json.load(fin, encoding='ascii')
# num_to_ges


# In[216]:

def get_processed_image(img):
    global img_w, img_h

#     img = cv2.imread(im_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    res = cv2.resize(gray,(img_w, img_w), interpolation=cv2.INTER_CUBIC)

    res = np.swapaxes(res, 0, 2)
    res = np.swapaxes(res, 1, 2)
    res = res[np.newaxis, :]

    return res


# In[217]:

from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

def predict(img):
    global model
    
    im = get_processed_image(img)
    
    model.forward(Batch([mx.nd.array(im)]))
    
    prob = model.get_outputs()[0].asnumpy()
    
    prob = np.squeeze(prob)
    
    a = np.argsort(prob)[::-1]
    
    max_prob = None
    max_idx = None
    
    for i in a[:5]:
        idx = str(i)
        if max_prob < prob[i] : 
            max_prob = prob[i]
            max_idx = idx
        # print('probability=%f, class=%s' %(prob[i], num_to_ges[idx]))
        
    return num_to_ges[max_idx]



