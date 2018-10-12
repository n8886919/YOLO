import numpy as np
import time
from mxnet import nd, gpu, cpu

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

def main(data, k, iters=10):
	plt.ion()
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_xlim(0, 1)
	ax.set_ylim(0, 1)
	show_nd_data_2d(ax, data)
	raw_input('Press any key to continue')

	means = []
	for r in np.random.randint(len(data), size=k):
		means.append(data[r])

	for iter in range(iters):
		
		tmp = []
		for mean in means:
			tmp.append(get_iou(data, mean))
			#tmp.append(get_distance(data, mean))

		best = nd.concat(*tmp, dim=-1).argmax(axis=1)
		#best = nd.concat(*tmp, dim=-1).argmin(axis=1)

		splited_data = []
		for _ in range(k):
			splited_data.append([])

		for b,d in zip(best.asnumpy(), data):
			splited_data[int(b)].append(d.reshape((1,2)))

		for i, d in enumerate(splited_data):
			d = nd.concat(*d, dim=0)
			means[i][0]=nd.sum(d, axis=0)[0]/len(d)
			means[i][1]=nd.sum(d, axis=0)[1]/len(d)
			splited_data[i] = d
		
		ax.clear()
		for d in splited_data:
			show_nd_data_2d(ax, d)
		plt.pause(0.01)
		
	return 	means

def get_iou(data, mean):
	# data = bs*(w, h) ndarray
	# mean = 1*(w, h) ndarray
	#|--------|-----|
	#| inters |     |
	#|--------|     |  h
	#|              |
	#|--------------|
	#        w   
	data_w, data_h = data.split(num_outputs=2, axis=-1)
	mean_w, mean_h = mean

	inters_w = nd.minimum(data_w, mean_w)
	inters_h = nd.minimum(data_h, mean_h)
	inters = inters_w * inters_h

	data_area = data_w * data_h
	mean_area = mean_w * mean_h
	ious = inters/(data_area + mean_area - inters)
	return ious 

def get_distance(data, mean):
	vec = data-mean
	distance = nd.norm(vec, ord=2, axis=-1).reshape((-1,1))
	return distance

def show_nd_data_2d(ax, data):
	dx, dy = data.split(num_outputs=2, axis=-1)
	ax.scatter(dx.asnumpy(),dy.asnumpy(), s=0.5)
	

if  __name__ == '__main__':
	'''
	loc = nd.array([[0.2,0.2],[0.2,0.8],[0.8,0.8],[0.8,0.2]])
	scale = nd.array([[0.3,0.3],[0.3,0.3],[0.3,0.3],[0.3,0.3]])
	data = nd.random.normal(loc, scale, shape=10000, ctx=gpu(0))
	data = data.transpose(axes=(2,0,1)).reshape((-1,2))
	'''
	data = nd.random.uniform(low=0, high=1, shape=((10000,2)))
	main(data, 9)


