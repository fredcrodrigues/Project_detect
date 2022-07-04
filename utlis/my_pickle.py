import os 
import numpy as np 
import pickle  as pk


### generation arcive pickle ##
cwd = os.getcwd()

path_data = cwd + "/data/test/"

list_mask_iris = []
list_pupil = []

list_image= []
list_mask = []


for name_dir in os.listdir(path_data): 
    print(name_dir)
   
    if name_dir == 'image': ## ALTERAR DIRETORIO 
        path_image_mask = path_data + 'image/'
        archiveImagMask = open(path_image_mask + "data_image.pickle" , 'wb')
        for name_image_mask in sorted(os.listdir(path_image_mask)):
            if name_image_mask != 'data_image.pickle':
                print(path_image_mask + name_image_mask )
                list_mask.append(path_image_mask + name_image_mask )
        pk.dump(list_mask, archiveImagMask)
        archiveImagMask.close()    
  
    if name_dir == 'mask_pupil': ## ALTERAR DIRETORIO
        path_image = path_data + 'mask_pupil/'
        
        archiveImage = open(path_image  + "data_mask_pupil.pickle" , 'wb')
        for name_image in sorted(os.listdir(path_image)):
            print(name_image)
            if name_image != 'data_mask_pupil.pickle':
               print(path_image + name_image)
               list_image.append(path_image + name_image)
       
        pk.dump(list_image, archiveImage )
        archiveImage.close()    




