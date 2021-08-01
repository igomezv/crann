# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 21:48:23 2018

@author: Guojian Wang
"""
#import sys
#sys.path.append('..')
#import refann.refann as rf
import refann as rf
import numpy as np
import matplotlib.pyplot as plt
import time
start_time = time.time()


fs8 = np.loadtxt('data/fs8Diagram.txt', usecols=(0,1,2), skiprows=3)

rec = rf.ANN(fs8,mid_node=4096,hidden_layer=1,hp_model='rec_2')
rec.train()
func = rec.predict(xpoint=np.linspace(0, 2.0, 151))
#func = rec.predict(xspace=(0, 1.5, 151)) #or use this
rec.save_func(path='rec_3', obsName='fs8') #save the reconstructed function

rec.plot_loss()
rec.plot_func()


print ("Time elapsed: %.3f mins" %((time.time()-start_time)/60))
plt.savefig("test.png")
plt.show()
