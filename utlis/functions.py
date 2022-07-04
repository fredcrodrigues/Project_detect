from skimage import measure
from keras import models
import matplotlib.pyplot as plt
import pickle as pk
import math
import re
import numpy as np
import cv2 as cv


############################ save filters of convolution ############################ 

def filters(featurs):
    ##alter path
    path = '/mnt/c/Users/Fredson/Downloads/project_pupil/data/train/image/01.JPG'
    img = cv.imread(path)
    img = cv.resize(img, dsize=(224,224), interpolation=cv.INTER_CUBIC)

    img = img/255.0
    img = np.array(img)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)

    layer_names = [layer._name for layer in featurs.layers]
    layer_output = [layer.output for layer in featurs.layers]
   
    activation_model = models.Model(inputs=featurs.input, outputs=layer_output) ## after is la featurs.layer[1].output
    
    f_activations = activation_model.predict(img)
    for layer_name, feature in zip(layer_names, f_activations):
      
       if 'conv' in layer_name:
            k = feature.shape[-1]
            size = feature.shape[1]
            
            #scale = 20. / k
            print(layer_name)
        

            for i in range(k):
                feature_image =  feature[0,:,:,i]
                plt.figure(figsize=(10,12))
                plt.imshow(feature_image, cmap="gray")
                plt.title(layer_name)
                plt.grid(False)
                plt.axis("off")
                if 'block1'in layer_name:
                    plt.savefig(f"/mnt/c/Users/Fredson/Downloads/project_pupil/results/filter/block1/{layer_name}_f_{i}.png")
                if 'block2'in layer_name:
                    plt.savefig(f"/mnt/c/Users/Fredson/Downloads/project_pupil/results/filter/block2/{layer_name}_f_{i}.png")
                if 'block3'in layer_name:
                    plt.savefig(f"/mnt/c/Users/Fredson/Downloads/project_pupil/results/filter/block3/{layer_name}_f_{i}.png")
                if 'block4'in layer_name:
                    plt.savefig(f"/mnt/c/Users/Fredson/Downloads/project_pupil/results/filter/block4/{layer_name}_f_{i}.png")
                if 'block5'in layer_name:
                    plt.savefig(f"/mnt/c/Users/Fredson/Downloads/project_pupil/results/filter/block5/{layer_name}_f_{i}.png")
                else:
                    break
                print("End!")
   

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


############################### ###########Load Archive TXT ##############

def load_txt(path, i):
    i += 1
    ##ajustement in path 
    if i < 10: 
        path = path + "0" + str(i) + ".txt"
    else:
        path = path +  str(i) + ".txt"

    ## tweak to get information from txt file
    info = open( path, 'r')
    infoPos = info.read()
    infoPos = re.sub(r'[a-zA-Zç~ã,:]', "", infoPos)
    infoPos = re.split(r'\s+', infoPos)
    infoPos.pop(0)
    return infoPos[0], infoPos[1], infoPos[2], infoPos[3], infoPos[4]


def save_archive_txt(path, info):
    archive = open(path, 'w')
    archive.write(info)
    archive.close()

################ Genration archve TXT FATOR #### #########################################

def save_txt(values, o_path, t_path):
  
    o_path = re.sub(".png", ".txt", o_path)
    t_path = re.sub(".png", ".txt", t_path)
    o_archive = open(o_path, 'w' )
    t_archive = open(t_path, 'w' )
    o_archive.write(values)
    t_archive.write(values)
    o_archive.close()
    t_archive.close()

############### Erro factor #################################################

def d_factor(i_small, p_small, path, i):
   FtReal, DRealIris, DRealPupil, RRealIris, RRealPupil = load_txt(path, i)
    ## fator calculado
   Ft, Di,  Dp, ri, rp = f_dilatation(i_small, p_small)
   DFator = calc_erro(FtReal, Ft)
   DIris = calc_erro(DRealIris, Di)
   DPupil = calc_erro(DRealPupil, Dp)
    ## alter path
   nPath = re.sub("/data/test/image/", "/results/output/fator/", path)
   save_archive_txt(nPath + "FATOR_DIFERENCE_IRIS_PUPIL_IMG_" +  str(i) + ".txt" , "Value of Factor caluculate: " +  
   str(Ft) + " Value diameter iris: " +  str(Di) + "Value Diameter Pupil: " + str(Dp) + "\nValue between factors (ERROR): " +  str(DFator) + "Diference between diameter iris: " + str(DIris) + 
    "Difference Between factors: " +  str(DPupil))
   return DFator, DIris, DPupil

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


