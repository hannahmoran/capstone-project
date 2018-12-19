#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 14:33:59 2018

@author: hannahmoran
"""

import pickle

negatives_remove = set(['cold', 
                        'shake',
                        'twist',
                        'loose'])

with open("negatives_remove.txt", "wb") as f:
	pickle.dump(negatives_remove, f)