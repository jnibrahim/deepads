import os
import numpy as np
import urllib as urllib
import cv2
import argparse

def store_raw_images(folder, link):
	pic_num = 1
	print("folder: ",folder)
	if not os.path.exists(folder):
		os.makedirs(folder)
	print("link: ",link)
	image_urls = str(urllib.urlopen(link).read())
	url_list = image_urls.split('\r\n')
	url_count = len(url_list)
	loop_counter = 0
	max_count = 500
	if url_count < max_count:
		max_count = url_count
	print("url_list count: ",url_count)
	
	while pic_num < 500:
		i = url_list[loop_counter]
		print("i in image_urls: ",i)
		print("Downloading image "+str(loop_counter)+" of "+str(url_count))
		try:
			urllib.urlretrieve(i, folder+"/"+str(pic_num)+".jpg")
			img = cv2.imread(folder+"/"+str(pic_num)+".jpg")
			
			if img is not None:
				cv2.imwrite(folder+"/"+str(pic_num)+".jpg", img)
				pic_num += 1
				
		except Exception as e:
			print(str(e))
		
		loop_counter += 1
		
		
	
	removeInvalid(folder)
				

def removeInvalid(dirPath):
	print("dirPath: ",dirPath)
	for img in os.listdir(dirPath):
		print("img: ",img)
		try:
			current_image_path = str(dirPath)+'/'+str(img)
			invalid = cv2.imread('invalid.jpg')
			question = cv2.imread(current_image_path)
			print("question.shape: ",question.shape)
			print("invalid.shape: ",invalid.shape)
			if invalid.shape == question.shape and not(np.bitwise_xor(invalid,question).all()) and current_image_path != 'invalid.jpg':
				print("this is an invalid image, it should be deleted")
				os.remove(current_image_path)


		except Exception as e:
			print(str(e))
					

def rotateImg(img, angle):
	(rows, cols, ch) = img.shape
	M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
	return cv2.warpAffine(img, M, (cols,rows))


def loadBlurImg(path, imgSize):
	img = cv2.imread(path)
	angle = np.random.randint(0, 360)
	img = rotateImg(img, angle)
	img = cv2.blur(img(5,5))
	img = cv2.resize(img, imgSize)
	return img
	
def loadImgClass(classPath, classLabel, classSize, imgSize):
	x = []
	y = []
	
	for path in classPath:
		img = loadBlurImg(path, imgSize)
		x.append(img)
		y.append(classLabel)
		
	while len(x) < classSize:
		randIdx = np.random.randint(0, len(classPath))
		img = loadBlurImg(classPath[randIdx], imgSize)
		x.append(img)
		y.append(classLabel)
		
	return x, y
	
def toGray(images):
	images = 0.2989*images[:,:,:,0] + 0.5870*images[:,:,:,1] + 0.1140*images[:,:,:,2]
	return images
	
def normalizeImages(images):
	images = (images / 255.).astype(np.float32)
	
	for i in range(images.shape[0]):
		images[i] = exposure.equalize_hist(images[i])
	
	images = images.reshape(images.shape + (1,))
	return images
	
def preprocessData(images):
	grayImages = toGray(images)
	return normalizeImages(grayImages)
	


if __name__ =="__main__":
	a = argparse.ArgumentParser()
	a.add_argument("--imagesURL")
	a.add_argument("--classLabel")
	
	args = a.parse_args()
	
	store_raw_images("training/images/"+args.classLabel, args.imagesURL)
