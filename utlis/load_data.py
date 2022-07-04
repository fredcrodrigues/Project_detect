from ast import Return
import cv2 as cv
import pickle as pk
import numpy as np
import os


cwd = os.getcwd()

def data_ajust(p_image, p_iris, p_pupil):
    
    file_load_IM= open(p_image, 'rb')
    file_load_I= open(p_iris , 'rb')
    file_load_P = open(p_pupil, 'rb')

    image = pk.load(file_load_IM)
    mask_iris = pk.load( file_load_I) 
    mask_pupil= pk.load(file_load_P) 

    print("Open pickle...")
    return image, mask_iris, mask_pupil

def data_array(image , mask_iris, mask_pupila):
    
    image_numpy = list()
    mask_iris_numpy = list() 
    mask_pupil_numpy = list()

    for path_image ,  path_mask_iris , path_mask_pupil in zip(image , mask_iris, mask_pupila):
     
        image_load = cv.imread(path_image)

        mask_iris_load = cv.imread(path_mask_iris)
        mask_pupil_load = cv.imread(path_mask_pupil)


        s_image = cv.resize(image_load,dsize=(224,224), interpolation=cv.INTER_CUBIC)
        s_mask_iris= cv.resize( mask_iris_load ,dsize=(224,224), interpolation=cv.INTER_CUBIC)
        s_mask_pupil = cv.resize(mask_pupil_load,dsize=(224,224), interpolation=cv.INTER_CUBIC)

        s_image =  s_image/255.0
        s_mask_iris = s_mask_iris/255.0
        s_mask_pupil = s_mask_pupil/255.0
        
        image_numpy.append(s_image[:,:])
        mask_iris_numpy.append(s_mask_iris[:,:])
        mask_pupil_numpy.append(s_mask_pupil[:,:])
       
        print("Adjustemeted image...")
  
    print("Load data ... ")

    return  image_numpy, mask_iris_numpy, mask_pupil_numpy 

def ajust_image(image , mask_iris, mask_pupila):
    
    x_train = np.array(image)
    y_train = np.array(mask_iris)
    z_train = np.array(mask_pupila)
    
    x_train = x_train.astype(np.float32)
    y_train  = y_train.astype(np.float32)
    z_train =  z_train.astype(np.float32)

    print("Preparing train and test...")
    return x_train,  y_train, z_train 

def load_data_test():
     path_test = cwd  + '/data/test/'

     x_test, y_test, z_test = load_data(path_test)

     return  x_test, y_test, z_test


def load_data_train():
     path_train = cwd  + '/data/train/train2/'

     x_train, y_train, z_train,  x_val, y_val, z_val = load_data(path_train)

     return  x_train, y_train, z_train, x_val, y_val, z_val

def load_data(path):
    print('Loop', path)
    if(path == cwd  + '/data/test/' ):
        print('Test', path)
        pathImage = path + 'image/' + 'data_image.pickle'
        pathMaskIris = path +  'mask_iris/' + 'data_mask_iris.pickle'
        pathMaskPupil =  path  + 'mask_pupil/' + 'data_mask_pupil.pickle'
        

        image, iris, pupil  = data_ajust( pathImage, pathMaskIris, pathMaskPupil)
        d_image,d_iris, d_pupil =  data_array(image, iris, pupil)
        x_test, y_test, z_test = ajust_image(d_image,d_iris, d_pupil)

        print('x_test:',x_test.shape)
        print('y_test:',y_test.shape)
        print('z_test:',z_test.shape)

        return  x_test, y_test, z_test
    else:

        pathTtrain = cwd  + '/data/train/train2/'
        pathTval = cwd  + '/data/val/val2/'

        pathTrainImage = pathTtrain + 'image/' + 'data_image.pickle'
        pathTrainIris = pathTtrain +  'mask_iris/' + 'data_mask_iris.pickle' 
        pathTrainPupil =  pathTtrain + 'mask_pupil/' + 'data_mask_pupil.pickle'
        
        pathValImage = pathTval + 'image/' + 'data_image.pickle'
        pathValIris  = pathTval + 'mask_iris/' + 'data_mask_iris.pickle'
        pathValPupil = pathTval  + 'mask_pupil/' + 'data_mask_pupil.pickle'
        
        ### Data Train
        t_image, t_iris, _t_pupil  = data_ajust( pathTrainImage,  pathTrainIris, pathTrainPupil )
        dt_image,dt_iris, dt_pupil =  data_array(t_image, t_iris, _t_pupil)
        x_train, y_train, z_train = ajust_image(dt_image,dt_iris, dt_pupil)
        
       
       
        ### Data Val
        v_image,t_iris, v_pupil  = data_ajust(  pathValImage, pathValIris  ,pathValPupil )
        dv_image, dv_iris, dv_pupil =  data_array(v_image, t_iris,  v_pupil)
        x_val, y_val, z_val = ajust_image(dv_image,dv_iris, dv_pupil)

        ## print shape
        print('x_train:', x_train.shape)
        print('y_train:', y_train.shape)
        print('z_train:', z_train.shape)

        print('x_val:', x_val.shape)
        print('y_val:', y_val.shape)
        print('z_val:', z_val.shape)

        return   x_train, y_train, z_train, x_val, y_val, z_val


  

    

    