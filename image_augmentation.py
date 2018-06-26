from scipy import ndarray
import os
import io
from scipy.misc import imsave
from scipy.misc import imread
import numpy as np
from skimage import exposure
from skimage import transform
import random
import skimage as sk

def random_rotation(image_array, image_name, path, count):
    	random_degree = random.uniform(-45, 45)
    	rot_image = sk.transform.rotate(image_array, random_degree)
    	rot_name = '%s_rotated(%d).jpeg' %(image_name, count)
    	save_path = os.path.join(path, rot_name)
    	imsave(save_path, rot_image)

def contrast_streching(img, image_name, path, count):
	A = random.randrange(2,10)
	B = random.randrange(90, 98)
	pA, pB = np.percentile(img, (A, B))
	img_rescale = exposure.rescale_intensity(img, in_range=(pA, pB))
	contrast_img_name = '%s_contrast(%d).jpeg' %(image_name, count)
	save_path = os.path.join(path, contrast_img_name)
	imsave(save_path, img_rescale)

def hist_equalizator(img, image_name, path, count):
	img_eq = exposure.equalize_hist(img)
	img_eq_name = '%s_equalized(%d).jpeg' %(image_name, count)
	save_path = os.path.join(path, img_eq_name)
	imsave(save_path, img_eq)
	
def adap_eq(img, image_name, path, count):
	clip_rand = random.random()
	img_adapteq = exposure.equalize_adapthist(img, clip_limit=clip_rand)
	img_adap_eq_name = '%s_adap_equalized(%d).jpeg' %(image_name, count)
	save_path = os.path.join(path, img_adap_eq_name)
	imsave(save_path, img_adapteq)

data_type = ['val', 'train', 'test']
for folder in data_type:
	path_normal = '/home/Documents/assigment_05/chest_xray/%s/NORMAL' %folder
	path_pneumonia = '/home/Documents/assigment_05/chest_xray/%s/PNEUMONIA' %folder
	save_path_normal = '/home/Documents/assigment_05/chest_xray/%s/NORMAL_AUGMENTED' %folder
	if not os.path.exists(save_path_normal):
    		os.makedirs(save_path_normal)
	save_path_pneumonia = '/home/ajpeter/Documents/assigment_05/chest_xray/%s/PNEUMONIA_AUGMENTED' %folder
	if not os.path.exists(save_path_pneumonia):
    		os.makedirs(save_path_pneumonia)

	dirs_normal = os.listdir(path_normal)
	image_dirs_normal = []

	for item in dirs_normal:
		if ('.jpeg' in item) or ('.JPEG' in item):
	                 image_dirs_normal.append(item)
	
	count = 0
	for image in image_dirs_normal:
		count = count + 1
		print("PROCESSING NORMAL CLASS FROM %s: %d of %d" %(folder, count, 5000))
		image_path = os.path.join(path_normal,image)
		image_name = image[:-5]
		image_array = imread(image_path)
		random_rotation(image_array, image_name, save_path_normal, count)
		count = count + 1
		print("PROCESSING NORMAL CLASS FROM %s: %d of %d" %(folder, count, 5000))
		contrast_streching(image_array, image_name, save_path_normal, count)
		count = count + 1
		print("PROCESSING NORMAL CLASS FROM %s: %d of %d" %(folder, count, 5000))
		hist_equalizator(image_array, image_name, save_path_normal, count)
		count = count + 1
		print("PROCESSING NORMAL CLASS FROM %s: %d of %d" %(folder, count, 5000))
		adap_eq(image_array, image_name, save_path_normal, count)

	
	path_normal = save_path_normal
	dirs_normal = os.listdir(path_normal)
	image_dirs_normal = []
	for item in dirs_normal:
		if ('.jpeg' in item) or ('.JPEG' in item):
	                 image_dirs_normal.append(item)	
	while(count<=5000):
		for image in image_dirs_normal:
			count = count + 1
			print("PROCESSING NORMAL CLASS FROM %s: %d of %d" %(folder, count, 5000))
			image_path = os.path.join(path_normal,image)
			image_name = image[:-5]
			image_array = imread(image_path)
			random_rotation(image_array, image_name, save_path_normal, count)
			count = count + 1
			print("PROCESSING NORMAL CLASS FROM %s: %d of %d" %(folder, count, 5000))
			contrast_streching(image_array, image_name, save_path_normal, count)
			count = count + 1
			print("PROCESSING NORMAL CLASS FROM %s: %d of %d" %(folder, count, 5000))
			hist_equalizator(image_array, image_name, save_path_normal, count)
			count = count + 1
			print("PROCESSING NORMAL CLASS FROM %s: %d of %d" %(folder, count, 5000))
			adap_eq(image_array, image_name, save_path_normal, count)
			
	

	dirs_pneumonia = os.listdir(path_pneumonia)
	image_dirs_pneumonia = []	
	for item in dirs_pneumonia:
		if ('.jpeg' in item) or ('.JPEG' in item):
			image_dirs_pneumonia.append(item)
	
	count = 0
	for image in image_dirs_pneumonia:
		count = count + 1
		print("PROCESSING PNEUMONIA CLASS FROM %s: %d of %d" %(folder, count, 5000))
		image_path = os.path.join(path_pneumonia,image)
		image_name = image[:-5]
		image_array = imread(image_path)
		random_rotation(image_array, image_name, save_path_pneumonia, count)
		count = count + 1
		print("PROCESSING PNEUMONIA CLASS FROM %s: %d of %d" %(folder, count, 5000))
		contrast_streching(image_array, image_name, save_path_pneumonia, count)
		count = count + 1
		print("PROCESSING PNEUMONIA CLASS FROM %s: %d of %d" %(folder, count, 5000))
		hist_equalizator(image_array, image_name, save_path_pneumonia, count)
		count = count + 1
		print("PROCESSING PNEUMONIA CLASS FROM %s: %d of %d" %(folder, count, 5000))
		adap_eq(image_array, image_name, save_path_pneumonia, count)
	
	path_pneumonia = save_path_pneumonia
	dirs_pneumonia = os.listdir(path_pneumonia)
	image_dirs_pneumonia = []
	for item in dirs_pneumonia:
		if ('.jpeg' in item) or ('.JPEG' in item):
			image_dirs_pneumonia.append(item)
	while(count<=5000):
		for image in image_dirs_pneumonia:
			count = count + 1
			path_pneumonia = save_path_pneumonia
			print("PROCESSING PNEUMONIA CLASS FROM %s: %d of %d" %(folder, count, 5000))
			image_path = os.path.join(path_pneumonia,image)
			image_name = image[:-5]
			image_array = imread(image_path)
			random_rotation(image_array, image_name, save_path_pneumonia, count)
			
			count = count + 1
			print("PROCESSING PNEUMONIA CLASS FROM %s: %d of %d" %(folder, count, 5000))
			contrast_streching(image_array, image_name, save_path_pneumonia, count)
			count = count + 1
			print("PROCESSING PNEUMONIA CLASS FROM %s: %d of %d" %(folder, count, 5000))
			hist_equalizator(image_array, image_name, save_path_pneumonia, count)
			count = count + 1
			print("PROCESSING PNEUMONIA CLASS FROM %s: %d of %d" %(folder, count, 5000))				
			adap_eq(image_array, image_name, save_path_pneumonia, count)
			
