from __future__ import print_function
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input,concatenate,Conv2D,Conv2DTranspose,Dropout,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.core import Activation
from keras.applications import vgg16 
from keras.callbacks import EarlyStopping, ModelCheckpoint
from skimage import measure
from utlis.attention import attention
from utlis.losses import *
from utlis.functions import *

import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import statistics as st

cwd = os.getcwd()



class Unet():
    
    def __init__(self , img_rows,img_cols,img_channels=3,batch_size=5,path_weights=None):
        
        self.img_rows=img_rows
        self.img_cols=img_cols
        self.img_channels=img_channels
        self.path_weights = path_weights
        self.batchsize=batch_size
        self.build_model()

    def build_model(self):
        
        self.vgg = vgg16.VGG16(include_top=False , weights='imagenet', input_shape=(self.img_rows , self.img_cols , self.img_channels)) 
        
        VGG16M = self.vgg 
        
        enconder = self.vgg.output
      

        ## -- block test -- ###
        #up6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(VGG16M.get_layer("block5_conv3").output)
        #up6 = attention(VGG16M.get_layer("block4_conv3").output,VGG16M.get_layer("block5_conv3").output,256)
        #up6 =  concatenate([VGG16M.get_layer("block5_conv3").output,VGG16M.get_layer("block4_conv3").output])
        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(VGG16M.get_layer("block5_conv3").output), VGG16M.get_layer("block4_conv3").output], axis=3)
        up6 =  attention(VGG16M.get_layer("block4_conv3").output, up6, 256)
        conv6 = Conv2D(256, (3, 3), padding='same')(up6)
        conv6 = LeakyReLU()(conv6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Dropout(0.2)(conv6)
        conv6 = Conv2D(256, (3, 3), padding='same')(conv6)
        conv6 = LeakyReLU()(conv6)
        conv6 = BatchNormalization()(conv6) 

        ## -- block test -- ###
        #up7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6)
        #up7 = attention(VGG16M.get_layer("block3_conv3").output, up7, 128)
        #up7 =  concatenate(VGG16M.get_layer("block3_conv3").output, up7)
        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), VGG16M.get_layer("block3_conv3").output], axis=3)
        up7  =  attention(VGG16M.get_layer("block3_conv3").output, up7, 128)
        conv7 = Conv2D(128, (3, 3), padding='same')(up7)
        conv7 = LeakyReLU()(conv7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Dropout(0.2)(conv7)
        conv7 = Conv2D(128, (3, 3), padding='same')(conv7)
        conv7 = LeakyReLU()(conv7)
        conv7 = BatchNormalization()(conv7)


        ## -- block test -- ###
        #up8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
        #up8 = attention(VGG16M.get_layer("block2_conv2").output, up8, 64)
        #up8 =  concatenate(VGG16M.get_layer("block2_conv2").output, up8)
        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7),VGG16M.get_layer("block2_conv2").output], axis=3)
        up8 = attention(VGG16M.get_layer("block2_conv2").output, up8, 64)
        conv8 = Conv2D(64, (3, 3), padding='same')(up8)
        conv8 = LeakyReLU()(conv8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Dropout(0.2)(conv8)
        conv8 = Conv2D(64, (3, 3), padding='same')(conv8)
        conv8 = LeakyReLU()(conv8)
        conv8 = BatchNormalization()(conv8)
        
        
        ## -- block test -- ###
        #up9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8)
        #up9 = attention(VGG16M.get_layer("block1_conv2").output, up9, 32)
        #up9 =  concatenate(VGG16M.get_layer("block1_conv2").output, up9)
        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), VGG16M.get_layer("block1_conv2").output], axis=3)
        up9  = attention(VGG16M.get_layer("block1_conv2").output, up9, 32)
        conv9 = Conv2D(32, (3, 3), padding='same')(up9)
        conv9 = LeakyReLU()(conv9)
        conv9 = Dropout(0.2)(conv9)
        conv9 = Conv2D(32, (3, 3), padding='same')(conv9)
        conv9 = LeakyReLU()(conv9)
        
        conv10 = Conv2D(3 , (1, 1) ,padding='same')(conv9)
        conv11 = Conv2D(3 , (1, 1) ,padding='same')(conv9)

        conv12 = Activation('sigmoid' ,  name='conv12')(conv10)
        conv13 = Activation('sigmoid' , name='conv13')(conv11)
        
      
        self.model = Model(VGG16M.input , [conv12, conv13])
        
        otimizador = Adam(lr = 0.00001, decay = 0.000001, clipvalue = 0.5)
        self.model.compile(optimizer=otimizador, loss='binary_crossentropy' , metrics=[dice_coef, 'binary_accuracy'])
       

        ## laod weigths
    def load(self):
        self.model.load_weights('models/weigths/utiris/pesos.h5')
        print("Modelo Carregado!")

    
   # train
    def train(self , epochs ,X_train, X_val, Y_train , Y_val, Z_train , Z_val ):  
     
        datagem = dict(
            #brightness_range=[1.0,1.1],
            horizontal_flip = True,
            vertical_flip = True,
            width_shift_range=0.2,
            zoom_range=0.3,
            rotation_range=90,
            fill_mode='nearest'
            )
        datagem_mask = dict(
           
            horizontal_flip = True,
            vertical_flip = True,
            width_shift_range=0.2,
            zoom_range=0.3,
            rotation_range=90,
            fill_mode='nearest'
        )

        seed =1 

        datagem_image = ImageDataGenerator(**datagem)
        datagem_mask_iris = ImageDataGenerator(**datagem_mask)
        datagem_mask_pupil = ImageDataGenerator(**datagem_mask)

        datagem_image_test = ImageDataGenerator(**datagem)
        datagem_mask_iris_test = ImageDataGenerator(**datagem_mask)
        datagem_mask_pupil_test = ImageDataGenerator(**datagem_mask)



        datagem_image.fit(X_train , seed=seed)
        datagem_mask_iris.fit(Y_train , seed=seed)
        datagem_mask_pupil.fit(Z_train , seed=seed)

        datagem_image_test.fit(X_val , seed=seed)
        datagem_mask_iris_test.fit(Y_val , seed=seed)
        datagem_mask_pupil_test.fit(Z_val , seed=seed)


        data_image = datagem_image.flow(X_train ,[1]*X_train.shape[0], batch_size = X_train.shape[0] , seed = seed)
        data_mask_iris = datagem_mask_iris.flow(Y_train ,[1]*Y_train.shape[0] ,  batch_size = Y_train.shape[0] , seed = seed)
        data_mask_pupil = datagem_mask_pupil.flow(Z_train ,[1]*Z_train.shape[0],batch_size = Z_train.shape[0] , seed = seed)

        data_image_val = datagem_image_test.flow(X_val ,[1]*X_val.shape[0], batch_size =X_val.shape[0], seed = seed)
        data_mask_iris_val  = datagem_mask_iris_test.flow(Y_val ,[1]*Y_val.shape[0], batch_size =  Y_val.shape[0] , seed = seed)
        data_mask_pupil_val = datagem_mask_pupil_test.flow(Z_val ,[1]*Z_val.shape[0], batch_size =  Z_val.shape[0] , seed = seed)



        train_generator = zip(data_image ,  data_mask_iris ,data_mask_pupil)
        val_generator = zip(data_image_val , data_mask_iris_val , data_mask_pupil_val)


        for (X,Y,Z) in train_generator:
            X_train = X[0]
            Y_train = Y[0]
            Z_train = Z[0]
        
            break
        for (X,Y,Z) in val_generator:
            X_val = X[0]
            Y_val = Y[0]
            Z_val = Z[0]
            break
        ## print shape image ##
        print(X_train.shape)
        print(Y_train.shape)
        print(Z_train.shape)
        print(X_val.shape)
        print(Y_val.shape)
        print(Z_val.shape)

        es = EarlyStopping(
            monitor='val_loss',
            patience=5,
            mode='min',
            verbose=1

        )

        mc = ModelCheckpoint('models/weigths/steps_utiris/model{epoch:001d}.h5' , monitor='val_loss', mode='min', verbose=1, save_best_only=True, save_freq='epoch')
        
       
        print("Teste Model")
        filters(self.model)

        self.model.summary()
        self.train_history=self.model.fit(X_train,(Y_train,Z_train),
                    validation_data=(X_val, (Y_val,Z_val)),
                    batch_size=self.batchsize,
                    verbose=1,
                    epochs=epochs,
                    callbacks=[es,mc]
                    )
      
        self.model.save_weights('models/weigths/utiris/utiris_attention_iris.h5')
        print("Train...") 
       
        plt.figure(figsize=[8,6])
        plt.plot(self.train_history.history['loss'] , label = "train loss" )
        plt.plot(self.train_history.history['val_loss'] , label = "validation loss" )
        plt.xlabel('Epochs ',fontsize=16)
        plt.ylabel('Loss',fontsize=16)
        plt.savefig('results/utiris/loss.png' , format='png') 
        
        plt.figure(figsize=[8,6])
        plt.plot(self.train_history.history['conv12_dice_coef'] , label = "Train Dice" )
        plt.plot(self.train_history.history['val_conv12_dice_coef'] , label = "Validation Dice" )
        plt.xlabel('Epochs ',fontsize=16)
        plt.ylabel('Dice',fontsize=16)
        plt.savefig('results/utiris/Dice_iris.png' , format='png') 

        plt.figure(figsize=[8,6])
        plt.plot(self.train_history.history['conv13_dice_coef'] , label = "train Dice" )
        plt.plot(self.train_history.history['val_conv13_dice_coef'] , label = "validation Dice" )
        plt.xlabel('Epochs ',fontsize=16)
        plt.ylabel('Dice',fontsize=16)
        plt.savefig('results/utiris/DicePupil.png' , format='png') 
       
    def test(self , image_teste , mask_iris_test, mask_pupil_test):
        list_dice_iris = []
        list_dice_pupil = []
        
        for i in range(len(image_teste)):
             
            image_test = np.expand_dims(image_teste[i] , axis=0)

            pred1 , pred2  = self.model.predict(image_test)
          
            pred1[pred1>0.5] = 1.0
            pred1[pred1<0.5] = 0.0

            pred2[pred2>0.5] = 1.0
            pred2[pred2<0.5] = 0.0
        

            pred1 = pred1[0]
            pred2 = pred2[0]


            ##### save image of segemntation ####
            s_image(i, pred1, cwd + '/results/output/iris/')
            s_image(i, pred2, cwd + '/results/output/pupil/')

            ## pos processign of image ######
            i_small =  p_processing(i, pred1, cwd + '/results/pos_processing/remove_area/iris/')
            p_small  = p_processing(i, pred2, cwd + '/results/pos_processing/remove_area/pupil/')
            
            i_small = morOpen(i_small)
            p_small = morOpen(p_small)

            ##### save  with small segemntation ####
            s_image(i, i_small, cwd + '/results/pos_processing/small/iris/b_')
            s_image(i, p_small, cwd + '/results/pos_processing/small/pupil/b_')


            ##### Ajust Small ####
            i_small = ajustSmall(i_small)
            p_small = ajustSmall(p_small)
           
         
            ## calculate dice
        
            diceIris = dice_coef_test(mask_iris_test[i], i_small)
            dicePupil = dice_coef_test(mask_pupil_test[i], p_small)

            print("Dice Iris",  diceIris)
            print("Dice Pupil",  dicePupil)

            ### round dice and add in list
            diceIris = round(diceIris,3)
            dicePupil =  round(diceIris, 3)

            list_dice_iris.append(diceIris)
            list_dice_pupil.append(dicePupil)


            ###########
            ##Save dice for image

            save_archive_txt( cwd + '/results/output/dice/IMG_IRIS_PUPIL_DICE' + str(i) + ".txt", 
                " Value  Dice for Iris: " + str(diceIris) + " Value  Dice for Pupil: " + str(dicePupil))

            ## calculate fator
            DFator, DIris, DPupil = d_factor(i_small, p_small, cwd + '/data/test/Fator/', i)

        
        mean_dice_iris = np.mean(list_dice_iris)   
        desvs_dice_iris = st.stdev(mean_dice_iris)

        mean_dice_pupil = np.mean(list_dice_pupil)
        desvs_dice_pupil = st.stdev(mean_dice_pupil)
        
        save_archive_txt(cwd + '/results/output/MEAN_END_.txt' , 
            " Value mean Dice Iris: " + str(mean_dice_iris) + " Standard Deviation Iris:  " + str(desvs_dice_iris)
            + " Value mean Dice Pupil: " + str(mean_dice_pupil)  + "Standard Deviation Pupil: " + str(desvs_dice_pupil))
        