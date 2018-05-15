import cv2
import os
from scipy import misc
import ipdb

def imshow(image, name=0):
  import cv2 as cv
  if type(image)==str: image = cv.imread(image)
  cv.startWindowThread()
  cv.namedWindow(str(name), cv.WINDOW_NORMAL)
  cv.imshow(str(name),image)  

def resizeImage(img, xSize, ySize):
  tUpleSize = (xSize, ySize)
  return misc.imresize(img, tUpleSize)

def cropAndResize(anImage, finalSize):
  imageShape = anImage.shape
  if imageShape[0] < imageShape[1]:
    cut1Cols = round((imageShape[1]-imageShape[0]) / 2)
    cut2Cols = imageShape[1]-round((imageShape[1]-imageShape[0])  / 2)
    cropedImg = anImage[0: imageShape[0]-1, cut1Cols:cut2Cols]
    return resizeImage(cropedImg, finalSize, finalSize)
  else:
    
    cut1Rows = int(round((imageShape[0]-imageShape[1]) / 2))
    cut2Rows = int(imageShape[0]-round((imageShape[0]-imageShape[1])  / 2))
    cropedImg = anImage[cut1Rows:cut2Rows, 0:imageShape[1]-1]
    # ipdb.set_trace()
    return resizeImage(cropedImg, finalSize, finalSize)
  # return resizeImage(anImage, finalSize, finalSize)

def transformImagesResize(source, targetDir,size):
  if os.path.isfile(targetDir + '/0001.jpg') or  os.path.isfile(targetDir + '/001.jpg') or  os.path.isfile(targetDir + '/0001.jpg'):#frames are there already
    print ('Frames for  ', source, ' already resized')
    return
  
  dirs = os.listdir(source)
  print('Process Dir', source, ' imgs ', len(dirs))   
  
  for aFile in dirs:   
    fileSourcePath = os.path.join(source, aFile)
    print(fileSourcePath)

    img = cv2.imread(fileSourcePath, cv2.IMREAD_COLOR)
    # ipdb.set_trace()
    r, g, b = cv2.split(img)
    
    #Sometimes you want to switch channels thats why this is here
    rescaledR = cropAndResize(r, size)
    rescaledG = cropAndResize(g, size)
    rescaledB = cropAndResize(b, size)

    rescaled = cv2.merge((rescaledR, rescaledG, rescaledB))

    targetImg = os.path.join(targetDir, aFile)
    if not os.path.exists(targetDir):
      os.makedirs(targetDir)
      print ('created ', targetDir)

    cv2.imwrite(targetImg, rescaled)
    
def handleTasksForPerson(aPersonDir, targetPerson,size):
  tasks = os.listdir(aPersonDir)

  for aTask in tasks:
    transformImagesResize(os.path.join(aPersonDir, aTask),  os.path.join(targetPerson, aTask),size)
    
def handleSubjects(rootPath, targetRoot,size): 
  persons = os.listdir(rootPath)
  print('len(dirs)', len(persons))
  for aPerson in persons:
    print('process person', aPerson)
    handleTasksForPerson(os.path.join(rootPath, aPerson), os.path.join(targetRoot, aPerson),size)

#EXEC CODE
size=400
root = '/home/afromero/datos/Databases/BP4D/Sequences'
targetRoot = '/home/afromero/datos/Databases/BP4D/Sequences_400'

handleSubjects(root, targetRoot,size)
