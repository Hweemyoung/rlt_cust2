# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 21:37:08 2019

@author: hweem
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # -1 to disable GPU, Blank to enable