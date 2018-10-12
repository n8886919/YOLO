from __future__ import print_function
import scipy.io as sio 
import os
import numpy as np
from PIL import Image
import xml.etree.cElementTree as ET

def rand_select_bg():	
	sun_img_path = os.path.join(bg_root, 'JPEGImages')
	sun_anno_path = os.path.join(bg_root, 'Annotations')
	while True:
		detected = False
		img = np.random.choice(os.listdir(sun_img_path))
		img_name = (img.split('.')[0]).split('/')[-1]
		#print(img)
		#print(bg_name)
		try:
			img_xml_path = os.path.join(sun_anno_path, (img_name+'.xml'))
			img_xml = ET.ElementTree(file=img_xml_path)
			root = img_xml.getroot()
			for child in root:
				if child.tag == 'object':
					for sub_child in child:
						if sub_child.tag == 'name':
							text = sub_child.text
							if ('car' in text or 'van' in text or 'truck' in text):
								detected = True
								break	
							'''
							if sub_child.text[1:-1] in delete_key_worlds: 
								detected = True
								break
							'''
					if detected: break
		except:
			detected = True
		if not detected:
			selected_bg = Image.open(os.path.join(sun_img_path ,img))
			break

	return selected_bg
bg_root = 'sun2012pascalformat'
data_root = 'PASCAL3D+_release1.1/Annotations/car_imagenet'
img_root = 'PASCAL3D+_release1.1/Images/car_imagenet'
train_root = 'train'
valid_root = 'valid'
syn_size = [480., 270.]
for fi,f in enumerate(os.listdir(data_root)):
	train = True if np.random.sample()<0.95 else False
	#f = 'n03770679_713.mat'
	mat = sio.loadmat(os.path.join(data_root, f))
	mat = mat['record'][0][0] # 0~9
	img_dir = os.path.join(img_root, f.split('.')[0]+'.JPEG')
	#print(f)
	for mi, m in enumerate(mat):
		if mi == 1:
			label = [[],[]]
			for ni, n in enumerate(m[0]):
				for pi, p in enumerate(n):		
					if pi == 1: 
						box = [int(i) for i in p[0]] 
						label[0] += [box]	
						#print('\t\t\t{}\t{}'.format(p[0][2]-p[0][0],p[0][3]-p[0][1]), end='')
						#print('\t\t\t{}\t{}\t{}\t{}'.format(*p[0]), end='')
					if pi == 3: 
						for qi, q in enumerate(p[0][0]):
							if qi == 2:
								img_cls = int((360-q[0][0])/45)
								#print('\t{}'.format(img_cls))
								label[1] += [img_cls]							
	
	img = Image.open(img_dir)
	r = [img.size[0]/syn_size[0], img.size[1]/syn_size[1]]
	max_r = max(r)	#smax_index = r.index(max_value)
	img = img.resize((int(img.size[0]/max_r), int(img.size[1]/max_r)), Image.BILINEAR)

	label[0] = np.array(label[0])/max_r	
	
	for j in range(5):
		############ img ############
		r2 = 0.5 + np.random.rand()*0.5
		img2 = img.resize((int(r2*img.size[0]), int(r2*img.size[1])), Image.BILINEAR)
		syn_img = rand_select_bg().resize((int(syn_size[0]), int(syn_size[1]))) 
		left = int((syn_size[0]-img2.size[0])*np.random.rand())
		top = int((syn_size[1]-img2.size[1])*np.random.rand())
		syn_img.paste(img2, (left, top))
		if train:
			save_name = os.path.join(train_root, f.split('.')[0] + '_%d'%j)
		else:
			save_name = os.path.join(valid_root, f.split('.')[0] + '_%d'%j)
		syn_img.save(save_name +'.jpg')
		############ label ############
		label2 = label[0]*r2
		label2[:, 0] = label2[:, 0] + left
		label2[:, 1] = label2[:, 1] + top
		label2[:, 2] = label2[:, 2] + left
		label2[:, 3] = label2[:, 3] + top
		label_num = len(label2)
		label3 = np.zeros((label_num, 5))
		for j1 in range(label_num):
			label3[j1, 0] = label[1][j1]
			for j2 in range(4):
				label3[j1, j2+1] = label2[j1][j2]
		#print(label3)
		np.savetxt(save_name + '.txt', label3, fmt='%d %.6f %.6f %.6f %.6f')		
		img2.close()
		syn_img.close()
	img.close()	
	#break
	#if fi > 100: break
