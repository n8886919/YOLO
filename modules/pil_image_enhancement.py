import numpy as np
import PIL

class PILImageEnhance():
	def __init__(self, M=0, N=0, R=0, G=1, noise_var=50):
		self.M = M
		self.N = N
		self.R = R
		self.G = G
		self.noise_var = noise_var

	def __call__(self, img, **kwargs):
		r = 0
		if self.M>0 or self.N>0 or ('M' in kwargs and kwargs['M'] != 0) or \
			('N' in kwargs and kwargs['N'] != 0):
			img = self.random_shearing(img, **kwargs)
		if self.R != 0 or ('R' in kwargs and kwargs['R'] != 0):
			img, r = self.random_rotate(img, **kwargs)
		if self.G != 0 or ('G' in kwargs and kwargs['G'] != 0):
			img = self.random_blur(img, **kwargs)
		if self.noise_var != 0 or \
			('self.noise_var' in kwargs and kwargs['self.noise_var'] != 0):
			img = self.random_noise(img, **kwargs)
		return img, r

	def random_shearing(self, img, **kwargs):
		#https://stackoverflow.com/questions/14177744/
		#how-does-perspective-transformation-work-in-pil
		M = kwargs['M'] if 'M' in kwargs else self.M
		N = kwargs['N'] if 'N' in kwargs else self.N
		w, h = img.size

		m, n = np.random.random()*M*2-M, np.random.random()*N*2-N # +-M or N
		xshift, yshift = abs(m)*h, abs(n)*w

		w, h = w + int(round(xshift)), h + int(round(yshift))
		img = img.transform((w, h), PIL.Image.AFFINE, 
			(1, m, -xshift if m > 0 else 0, n, 1, -yshift if n > 0 else 0), 
			PIL.Image.BILINEAR)

		return img

	def random_noise(self, img, **kwargs):
		noise_var = kwargs['noise_var'] if 'noise_var' in kwargs else self.noise_var
		np_img = np.array(img)
		noise = np.random.normal(0., noise_var, np_img.shape)
		np_img = np_img + noise 
		np_img = np.clip(np_img, 0, 255)
		img = PIL.Image.fromarray(np.uint8(np_img))
		return img

	def random_rotate(self, img, **kwargs):
		R = kwargs['R'] if 'R' in kwargs else self.R

		r = np.random.uniform(low=-R, high=R) 
		img = img.rotate(r, PIL.Image.BILINEAR, expand=1)
		r = float(r*np.pi)/180
		return img, r
	
	def random_blur(self, img,  **kwargs):
		G = kwargs['G'] if 'G' in kwargs else self.G
		img = img.filter(PIL.ImageFilter.GaussianBlur(radius=np.random.rand()*G))
		return img