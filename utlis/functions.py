from skimage import measure
import math
import re
import numpy as np
import cv2 as cv


########### ####### CALCULATE DICE ##########################

def dice_coef_test(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    union = np.sum(y_true_f) + np.sum(y_pred_f)
    if union==0: return 1
    intersection = np.sum(y_true_f * y_pred_f)
    return 2. * intersection / union

######  ###### ############ Diferencce between factores #############################

def calc_erro(value1, value2):
    value = float(value1) - float(value2)
    return value
    

####### #################  factor dilatation ##########################################

def diamenter(area):
    dm = 2 * math.sqrt(area/math.pi)
    return dm

def radius(area):
    r = math.sqrt(area/math.pi)
    return r 

def d_regions(img): 
    label = measure.label(img)
    regions = measure.regionprops(label)
    dm = diamenter(regions[0].area)
    r = radius(regions[0].area)
    return dm, r

def f_dilatation(iris, pupil):

   Di, ri = d_regions(iris)
   Dp, rp= d_regions(pupil)
   Ft = Dp/Di
   return Ft, Di,  Dp, ri, rp

def save_archive_txt(path, info):
    archive = open(path, 'w')
    archive.write(info)
    archive.close()

###################### save a image ###############
def r_images(img):
    img[np.where( img == 1 )] = 255
    img = cv.resize(img, dsize=(2048, 1360), interpolation=cv.INTER_CUBIC)
    return img

def s_image(i, image, path):
    ## verify the type image
    if(image.dtype != "uint8"):
        image = image.astype('uint8') 
        image = r_images(image)
        cv.imwrite(path + 's_img_' + str(i) + '.png',  image)
    else:
        image = r_images(image)
        cv.imwrite(path + 's_img_' + str(i) + '.png',  image)

############################ AJUSTMETED IMAGE ############################

def ajustSmall(small):

    small = cv.resize(small, dsize=(224,224), interpolation=cv.INTER_CUBIC)
    small = small/255.0
    small = small.astype(np.float32)
    return small

################################ ##### POS PROCESSING ########################################

####### small ########

def small(pred):
    ## convert 
    pred = pred.astype('uint8') 
    pred = r_images(pred)
    ##convert on chanell
    p_gray = cv.cvtColor(pred, cv.COLOR_BGR2GRAY)
    ##convert to binary
    ret,thresh = cv.threshold(p_gray ,127,255,cv.THRESH_BINARY)
    ##find contours
    contour, h =  cv.findContours(thresh,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    c = contour[0]
    ##algorithm minim circle
    (x,y), radius  = cv.minEnclosingCircle(c) 
    center = (int(x),int(y))
    r = int(radius)
    ##Draw circle minim circle
    cv.circle(pred, center, r, (255,255,255), -1)
    return pred
    


####### Operations morphology  ########
def morOpen(img):   
    kernel = np.ones((5,5))
    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel, iterations=5)
    return img

def erosion(pred):
    kernel =np.ones((3,3))
    print(kernel)
    return pred 

 
def dilatation(pred):
    kernel = np.ones((5,5), np.uint8)
    pred = cv.dilate(pred, kernel, iterations = 2)
    return pred 
####### Remove false positives  ########

def r_area( i, pred, path):
    i_erosion = erosion(pred)
    i_dilatation = dilatation(i_erosion)
    s_image(i, i_erosion, path + "e_")
    s_image(i, i_dilatation, path + "d_")
    return  i_erosion, i_dilatation
    
def p_processing(i, pred, path):
    label = measure.label(pred)
    regions = measure.regionprops(label)

    if (len(regions) > 1):
        ##remove area
       i_erosion, i_dilatation  = r_area(i, pred, path)
        ##aplication of small
       i_small = small(i_dilatation)
       i_small = morOpen(i_small)
       path = re.sub("remove_area", "small", path)
       s_image(i, i_small, path + "a_")
    
    else:
        path = re.sub("remove_area", "small", path)
        i_small = small(pred)
        i_small = morOpen(i_small)
        s_image(i, i_small, path + "a_")

    return i_small


