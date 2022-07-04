from __future__ import print_function
#os.environ["KERAS_BACKEND"] = "tensorflow"
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input,concatenate,Conv2D,MaxPooling2D,AveragePooling2D,Conv2DTranspose,Layer,Cropping2D
from keras.layers import Dense,Flatten, Dropout,BatchNormalization, ZeroPadding2D,Lambda
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint ,ReduceLROnPlateau
from keras.layers.core import Activation
from keras import backend as K
from keras.applications import vgg16

from utlis.losses import *

import statistics
import pickle  as pk
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

from math import cos, sin, pi, sqrt, atan2





def dice(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    union = np.sum(y_true_f) + np.sum(y_pred_f)
    intersection = np.sum(y_true_f  * y_pred_f)
    
    return 2. * intersection / union
def coord(coord_lis):
    pi = 3.14
    for i in range(len(coord_lis)):
        x1 = coord_lis[0]
        y1 = coord_lis[1]
        rp = coord_lis[2]

        x2 = coord_lis[3]
        y2 = coord_lis[4]
        rr = coord_lis[5]

        Areap = pi*(rp*rp)
        Arear = pi*(rr*rr)

        dx , dy = x1 - x2 , y1 - y2
        d = sqrt(dx*dx+dy*dy)
        if d > rp + rr:
            print("não há interseção")
         
        iou = d/float(Areap + Areap - d)

        #print("coordenadas:", iou)
    return coord_lis

def houg(canny_pred, canny_real, norm_pred , conv_mask , conv_image , conv_image_real):

    circles_pred = cv.HoughCircles(canny_pred , cv.HOUGH_GRADIENT , 1 , 20 , param1=1.2, param2=10, minRadius=0 , maxRadius=0)
    circle_real = cv.HoughCircles(canny_real , cv.HOUGH_GRADIENT , 1 , 20 , param1=1.2, param2=10, minRadius=0 , maxRadius=0)
    
    image_real = conv_image
    image_pred = conv_image_real
    
    #cv.imshow("canny pred",canny_pred)
    #cv.imshow("canny real",canny_real)
    list_coor = []
    if circles_pred is not None: 
        circles_pred = np.uint16(np.around(circles_pred))
       
        for i in circles_pred[0,:]:
           image_pred_circle = cv.circle(image_pred, (i[0],i[1]),i[2],(0,255,0), 1)
           list_coor.append(i[0])
           list_coor.append(i[1])
           list_coor.append(i[2])
        
        
            
    if circle_real is not None: 
        circle_real = np.uint16(np.around(circle_real))
        
        for x in circle_real[0,:]:
           image_real_circle = cv.circle(image_real, (x[0],x[1]),x[2],(0,255,0), 1)
           list_coor.append(x[0])
           list_coor.append(x[1])
           list_coor.append(x[2])
   
    #coord(list_coor)
    
    #intersection_houng = np.logical_and( mage_pred_circle, image_real_circle)
    #unio_hougn = np.logical_or( image_pred_circle, image_real_circle)
    #iou = np.sum(intersection_houng)/np.sum(unio_hougn)
 
   
    return  image_pred_circle , image_real_circle

def pos_processing(pred ,image , mask ):

   
    cov_pred = pred.astype('uint8')
    conv_image_real = image.astype(np.uint8)
    cov_image = image.astype(np.uint8)
    conv_mask = mask.astype(np.uint8)
    
    
    norm_pred = np.zeros(cov_pred.shape)
    norm_pred = cv.normalize(cov_pred ,  norm_pred, 0 ,255 , cv.NORM_MINMAX)
    
    
    canny_pred = cv.Canny(norm_pred , 60 , 100)
    canny_real = cv.Canny(conv_mask , 60 , 100)
    
    houg_img ,  image_real  = houg(canny_pred , canny_real , norm_pred , conv_mask , cov_image ,conv_image_real)

    '''
    ##iou
    c = np.expand_dims(canny_real , axis=-1)
    intersection_houng = np.logical_and(houg_img, image_real )
    unio_hougn = np.logical_or( houg_img, image_real )
    iou = np.sum(intersection_houng)/np.sum(unio_hougn)
    
    print("IOU:", iou)
    cv.imshow('Detecção de Borda' , canny_pred)
    cv.imshow('Circulo detetctado' , houg_img)
    cv.imshow('Mascara Original' , conv_mask)
   
    cv.waitKey(0)
   # return houg_img
    '''
def data_array(image , mask ,  image_test , mask_test):
    
    ## UNIR MASCARAS BINARIAS 
    image_numpy = list()
    mask_numpy = list()


    image_numpy_test= list()
    mask_numpy_test = list()   


    for path_image ,  path_mask in zip(image , mask):
        
        
        image_load = cv.imread(path_image)
        mask_load = cv.imread(path_mask)
       
        

        image_train = cv.resize(image_load,dsize=(224,224), interpolation=cv.INTER_CUBIC)
        mask_train = cv.resize(mask_load,dsize=(224,224), interpolation=cv.INTER_CUBIC)


        image_train =  image_train/255.0
      

        image_numpy.append(image_train[:,:])
        mask_numpy.append(mask_train[:,:])
       

    for path_image_tes,  path_mask_tes in zip(image_test,mask_test):
    
        image_load_test = cv.imread(path_image_tes)
        mask_load_test = cv.imread(path_mask_tes)
       
        image_train_test = cv.resize(image_load_test,dsize=(224,224), interpolation=cv.INTER_CUBIC)
        mask_train_test = cv.resize(mask_load_test,dsize=(224,224), interpolation=cv.INTER_CUBIC)
       
     
        image_train_test =  image_train_test/255.0
      
        

        image_numpy_test.append(image_train_test[:,:])
        mask_numpy_test.append(mask_train_test[:,:])
    
    print("Dados Carregado ... ")

    return image_numpy , mask_numpy ,  image_numpy_test , mask_numpy_test 
   
def load_data():
    path_train = '/home/fredson/project_pupil/data/train/'
    path_test = '/home/fredson/project_pupil/data/test/'
    
    train_I , train_M = path_train + 'image/' + 'data_image.pickle' ,  path_train + 'class_train/' + 'data_mask.pickle'
    test_I , test_M =  path_test + 'image_test/' + 'data_image_test.pickle' ,  path_test + 'class_test/' + 'data_mask_test.pickle'
   
    file_load_I = open(train_I, 'rb')
    file_load_M = open(train_M , 'rb')
    
    file_load_I_t = open(test_I, 'rb')
    file_load_M_t = open(test_M, 'rb')
     
    image = pk.load(file_load_I)
    mask = pk.load(file_load_M) 
    
    image_test = pk.load(file_load_I_t)
    mask_test = pk.load(file_load_M_t)

    x_train , y_train , x_test , y_test = data_array(image,mask,image_test,mask_test)

    ##AJUSTES##
   
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    
   
    
    print('x_train:',x_train.shape)
    print('y_train:',y_train.shape)
    print('x_test:', x_test.shape)
    print('y_test:',y_test.shape)

    

    return x_train , y_train , x_test , y_test  
  

class Unet():
    def __init__(self , img_rows,img_cols,img_channels=3,batch_size=16,N_CLASSES=3 ,path_weights=None):
        self.img_rows=img_rows
        self.img_cols=img_cols
        self.img_channels=img_channels
        self.path_weights = path_weights
        self.batchsize=batch_size
        self.N_CLASSES=N_CLASSES
        self.build_model()

    
    def build_model(self):

        ###ARUQTETURA COM VGG 16 E UM OUTPUT##
        
        self.vgg = vgg16.VGG16(include_top=False , weights='imagenet', input_shape=(self.img_rows , self.img_cols , self.img_channels)) 
        
        VGG16M = self.vgg 
        
        enconder = self.vgg.output
        
        '''
        ##FREEZE LAYERS
        set_trainable = False
        for layer in self.vgg.layers:
            if layer.name in ['block1_conv1']:
                set_trainable = True
                print('Descoongelado')
            if layer.name in ['block1_pool','block2_pool','block3_pool','block4_pool','block5_pool']:
                
                layer.trainable = False
                print("CONGELAR")
        '''
        
        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(VGG16M.get_layer("block5_conv3").output), VGG16M.get_layer("block4_conv3").output], axis=3)
        conv6 = Conv2D(256, (3, 3), padding='same')(up6)
        conv6 = LeakyReLU()(conv6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Dropout(0.2)(conv6)
        conv6 = Conv2D(256, (3, 3), padding='same')(conv6)
        conv6 = LeakyReLU()(conv6)
        conv6 = BatchNormalization()(conv6)

        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), VGG16M.get_layer("block3_conv3").output], axis=3)
        conv7 = Conv2D(128, (3, 3), padding='same')(up7)
        conv7 = LeakyReLU()(conv7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Dropout(0.2)(conv7)
        conv7 = Conv2D(128, (3, 3), padding='same')(conv7)
        conv7 = LeakyReLU()(conv7)
        conv7 = BatchNormalization()(conv7)

        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), VGG16M.get_layer("block2_conv2").output], axis=3)
        conv8 = Conv2D(64, (3, 3), padding='same')(up8)
        conv8 = LeakyReLU()(conv8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Dropout(0.2)(conv8)
        conv8 = Conv2D(64, (3, 3), padding='same')(conv8)
        conv8 = LeakyReLU()(conv8)
        conv8 = BatchNormalization()(conv8)

        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), VGG16M.get_layer("block1_conv2").output], axis=3)
        conv9 = Conv2D(32, (3, 3), padding='same')(up9)
        conv9 = LeakyReLU()(conv9)
        conv9 = Dropout(0.2)(conv9)
        conv9 = Conv2D(32, (3, 3), padding='same')(conv9)
        conv9 = LeakyReLU()(conv9)

        conv10 = Conv2D(self.N_CLASSES, (1, 1) ,padding='same')(conv9)
        
        conv10 = Activation('sigmoid')(conv10)
        
        self.model = Model(VGG16M.input , conv10)
        
       
        
        otimizador = Adam(lr = 0.00001, decay = 0.000001, clipvalue = 0.5)
        #ot = Adam(lr=5*1e-4)
        self.model.compile(optimizer=otimizador , loss='binary_crossentropy' , metrics=[dice_coef,'binary_accuracy',])
    

    def load(self):
        self.model.load_weights('models/weigths/weigths.h5')
        print("Modelo Carregado!")

    def train(self , epochs , X_train , X_test , Y_train , Y_test):
        
        print(X_train.shape ,  Y_train.shape  , Y_test.shape  , Y_test.shape )
        self.model.summary()
        self.train_history=self.model.fit(X_train, Y_train,
                    validation_data=(X_test,Y_test),
                    batch_size=self.batchsize,
                    epochs=epochs,
                    verbose=1
					)
        self.model.save_weights('models/weigths/weigths.h5') 
        print("Treinando")

        plt.figure(figsize=[8,6])
        plt.plot(self.train_history.history['loss'] , label = "train loss" )
        plt.plot(self.train_history.history['val_loss'] , label = "validation loss" )
        plt.xlabel('Epochs ',fontsize=16)
        plt.ylabel('Loss',fontsize=16)
        plt.savefig('results/loss.png' , format='png') 
        
        plt.figure(figsize=[8,6])
        plt.plot(self.train_history.history['dice_coef'], label = "train dice")
        plt.plot(self.train_history.history['val_dice_coef'],  label = "validation dice")
        plt.xlabel('Epochs ',fontsize=16)
        plt.ylabel('Coeficiente Dice',fontsize=16)
        plt.savefig('results/dice_coif.png' , format='png') 

 
    def test(self , image_teste , mask_test):
        print(len(image_teste))
       
      
        list_dice = []

        for i in range(len(image_teste)):
          
        
            t = np.expand_dims(image_teste[i] , axis=0)
           
            pred = self.model.predict(t)


            pred[pred>0.5] = 1.0
            pred[pred<0.5] = 0.0

           
            new_mask = np.zeros(mask_test[i].shape)

            new_mask[np.where(mask_test[i] == 0)] = 0
            new_mask[np.where(mask_test[i] == 1)] = 255

            # SEGMENTAÇÃO FIGURA

            fig = plt.figure(figsize = (8,6))
            plt.subplot(1,4,1)
            plt.imshow(pred[0])
            plt.title("Previsão")
            plt.axis("off")
            
                
            plt.subplot(1,4,2)
            plt.imshow(new_mask)
            plt.title("Mascara Real")
            plt.axis("off")
            
            plt.subplot(1,4,3)
            plt.imshow(new_mask, alpha = 0.5 , cmap = "gray")
            plt.imshow(pred[0] ,  cmap = "gray" , alpha = 0.3)
            plt.title("Sobreposição")
            plt.axis("off")
    
            plt.subplot(1,4,4)
           
            plt.imshow(image_teste[i] , alpha = 0.3 , cmap="gray")
            plt.title("origin")
            plt.axis("off")
            plt.savefig('results/imagens_test/image' + str(i) + '.png' ,  format='png' )


            # CALCULO DO DICE
            new_mask = np.expand_dims(new_mask ,  axis=0)
            c = dice_calc(new_mask, pred)
            #print('dice' , c)

            list_dice.append(c)
           
           
            pos_processing(pred[0],image_teste[i],new_mask)

            '''
            #DESVIO PADRÃO DO CONJUNTO
            mean = np.mean(image_teste[i])
            std = np.std(image_teste[i])
            std2 = np.std(image_teste[i])
            pre_std = np.std(pred)
            print(mean)
            print(std)
            print(std2)
            print(pre_std)
            '''
          
        '''
        # MÉDIAS FINAIS
        mean_dice = np.mean(list_dice)
        desvs_dice = np.std(0.85331)
        desvs_dice_two = np.std(mean_dice)
        print("Media - " , mean_dice)
        print("Desvio Padrão Coef Dice - " ,desvs_dice)
        '''
      
