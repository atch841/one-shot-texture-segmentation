import cv2
import numpy as np
import os

def generate_texture(img_folder_p):
	# Read images from dtd dataset and crop them into 256x256.
	# Images that are smaller than 256x256 are dropped.
	all_texture = []
	for img_folder_c in os.listdir(img_folder_p):
	    img_folder = img_folder_p + img_folder_c + '/'
	    for img_name in os.listdir(img_folder):
	        img = cv2.imread(img_folder + img_name)
	        if type(img) == type(None):
	            print(img_folder + img_name, 'is not a image.')
	            continue
	        if img.shape[0] < 256 or img.shape[1] < 256:
	            print(img_folder + img_name, 'is too small.')
	            continue
	        mid_0 = int(img.shape[0] / 2)
	        mid_1 = int(img.shape[1] / 2)
	        img = img[mid_0-128:mid_0+128,mid_1-128:mid_1+128,:]
	        all_texture.append(img)
	all_texture = np.array(all_texture)
	return all_texture

if __name__ == '__main__':
	img_folder_p = '/fast_data/texture/dtd/dtd-r1.0.1/dtd/images/'
	textures = generate_texture(img_folder_p)
	print('full textures shape:', textures.shape)
	np.random.shuffle(textures)
	train_texture = textures[:-200]
	val_texture = textures[-200:]
	print('train, val split:', train_texture.shape, val_texture.shape)
	np.save('train_texture.npy', train_texture)
	np.save('val_texture.npy', val_texture)