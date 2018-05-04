import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm

def process(base_dir):
	print(base_dir)
	#print(os.popen('ls .').readlines())
	subdirs = [base_dir + d[:-1] + '/' for d in os.popen('ls ' + base_dir).readlines()]
	print(subdirs)
	img_list = []
	label_list = []
	for subdir in subdirs:
		print(subdir)
		img_files = [subdir + '/' + d[:-1] for d in os.popen('ls ' + subdir).readlines()]
		#print(img_files)
		for img_file in tqdm(img_files):
			#print(img_file)
			img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
			img = cv2.resize(img, (48, 48))
			#print(img.shape)
			img_list.append(img)
			label = int(subdir.split('/')[-2])
			#print(label)
			label_list.append(label)
	imgs = np.asarray(img_list)

	#print(imgs.shape)
	imgs = imgs[:, np.newaxis, :, :]
	#print(imgs.shape)
	labels = np.asarray(label_list)
	print(labels[:100])
	#print(labels.shape)
	return imgs, labels

if __name__ == '__main__':
	train_img, train_label = process('./train/')
	print('shape of train:', train_img.shape, train_label.shape)
	with open('./train.pkl', 'wb') as f:
		pickle.dump([train_img, train_label], f)
	valid_img, valid_label = process('./val/')
	print('shape of valid:', valid_img.shape, valid_label.shape)
	with open('./valid.pkl', 'wb') as f:
		pickle.dump([valid_img, valid_label], f)
	test_img, test_label = process('./test/')
	print('shape of test:', test_img.shape, test_label.shape)
	with open('./test.pkl', 'wb') as f:
		pickle.dump([test_img, test_label], f)
