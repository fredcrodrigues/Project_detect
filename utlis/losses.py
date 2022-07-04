from keras import backend as K
import numpy as np
from keras import *
import keras

def dice_calc(im1, im2, empty_score=1.0):
	im1 = np.asarray(im1).astype(np.bool)
	im2 = np.asarray(im2).astype(np.bool)
	if im1.shape != im2.shape:
		raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

	im_sum = im1.sum() + im2.sum()
	if im_sum == 0:
		return empty_score

	# Compute dice_val coefficient
	intersection = np.logical_and(im1, im2)

	return 2. * intersection.sum() / im_sum

def dice_jonnison(y_true, y_pred, smooth=1):
	im_sum = K.sum(y_pred) + K.sum(y_true)
	intersection = y_true * y_pred
	return 2.*K.sum(intersection)/im_sum

#coeficente deice https://forums.fast.ai/t/understanding-the-dice-coefficient/5838


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def dice_jonnison_loss(y_true, y_pred):
	return 1-dice_jonnison(y_true, y_pred)



