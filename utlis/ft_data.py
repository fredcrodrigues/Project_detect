from functions import *
from load_data import *
import os

cwd = os.getcwd()

## calculate factor of date ###

def calcFactor(onePath, twoPath):
    for onePath, twoPath in zip(onePath, twoPath):
       
        print('Caminho Iris: ', onePath)
        print('Caminho Pupil: ', twoPath)

        oneImage = cv.imread(onePath)
        twoImage = cv.imread(twoPath)
    
        Ft, Di,  Dp, ri, rp = f_dilatation(oneImage, twoImage)
        values  = "Fator de dilatação: " + str(Ft) + "," + " Diametro Iris: " + str(Di) + "," + " Diametro pupila: " + str(Dp) + "," + " Raio iris: " + str(ri) + "," + " Raio Pupila: " + str(rp)
        save_txt(values, onePath, twoPath)



def verifyPast(pathOriginal, past):
    print("Path", pathOriginal)
    print("Path", past)
  
    pathImage = pathOriginal + past + "/image/data_image.pickle"
    pathIris = pathOriginal + past + "/mask_iris/data_mask_iris.pickle"
    pathPupil = pathOriginal + past + "/mask_pupil/data_mask_pupil.pickle"
    
    print("Path fim ", pathIris)
    print("Path fim ", pathPupil)
    
    
    return  pathImage,  pathIris,  pathPupil 
 
   
if __name__ == "__main__":
   
    path= cwd  + '/data/'
    for past in os.listdir(path):
 
        if(past == "test"):
           pathTest, IrisTest,  PupilTest = verifyPast(path, past )
           print("Path Picle", pathTest)

           imageTest, irisTest, pupilTest = data_ajust(pathTest, IrisTest,  PupilTest)
           calcFactor(irisTest,  pupilTest)
        elif(past == "train"):
           pathTrain, IrisTrain,  PupilTrain = verifyPast(path, past)
           print("Path Picle", pathTrain)

           image, irisTrain, pupilTrain = data_ajust(pathTrain, IrisTrain,  PupilTrain)
           calcFactor(irisTrain, pupilTrain)

        elif(past == "val"):
           pathVal, IrisVal,  PupilVal = verifyPast(path, past)
           print("Path Picle", pathVal)

           imageVal, irisVal, pupilVal = data_ajust( pathVal, IrisVal,  PupilVal)
           calcFactor(irisVal,  pupilVal)
        else:
            break
           
    

     
      
  