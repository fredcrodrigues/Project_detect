from models.segmentation import *
from utlis.load_data import *
import argparse 

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog="main" , description='-- Sgementation Pupil -- U-NET')
    parser.add_argument('--mode', '-m', type=str , help='Erro mode')
    
    
    args = parser.parse_args()
    mode = args.mode

    batch_size = 5
	
    Model = Unet(224,224,img_channels=3,batch_size=batch_size)
    
    if(mode=='train'):

        X_train , Y_train  , Z_train , X_val , Y_val , Z_val = load_data_train()
        args = parser.parse_args()
        arguments = args.__dir__
        epoch = 100
        Model.train(epoch, X_train, X_val ,Y_train , Y_val , Z_train , Z_val)


    if(mode=='test'):

        X_test , Y_test ,Z_test =  load_data_test()
        Model.load()
        Model.test(X_test, Y_test , Z_test)

   