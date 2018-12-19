#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 14:51:45 2018

@author: hannahmoran
"""

import pickle

positives_remove = set(['strong', 
                        'fairly', 
                        'clearly',
                        'loose'])

with open("positives_remove.txt", "wb") as f:
	pickle.dump(positives_remove, f)