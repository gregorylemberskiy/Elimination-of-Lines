"""
* -----------------------------------------------------------------------------
* MODULE: elmpy
*
* Author: Gregory Lemberskiy
* -----------------------------------------------------------------------------
"""
from Readfits  import *
from Highpass  import *
from hough     import *
from Optim     import *
from gl_imshow import *

if __name__ == "__main__":
	
	image, invar = Readfits()

	#Passing the image through a Highpass filter
	highpass_image = Highpass(image)
	
	"""
	---------------------------------------------------------------
	If the image is complicated: Stellar objects are brighter than
	the trail include True as the second paramater for hough, this 
	processes the image using  histeq(). Histeq returns an nd.array
	with lowered intensities. 
	
	---------------------------------------------------------------
	
	For images in which the trail is the brightest object, include 
	False as the second parameter.
	
	---------------------------------------------------------------
	
	Histeq is necessary for the default image. 
	"""
	himage, Offset, Maxbindex, Angle, bins = hough(highpass_image,True)

	# [Offset, Angle, Sky, Thickness, Normalization, Left Endpoint,
	#  Right Endpoint]
	par_guess = [50., -30., 1.0, .005, 1.0, 0.5, 0.5]
	model = Optim(par_guess, image)
	
	fig = plt.figure()
	ax1 = fig.add_subplot('221')
	ax2 = fig.add_subplot('222')
	ax3 = fig.add_subplot('212')
	
	diff = image - model
	
	vmin = np.percentile(image, 1.)
	vmax = np.percentile(image,99.)
	
	plt.gray()

	gl_imshow(image.T, ax=ax1, origin='image',vmin=vmin, vmax=vmax, 
		   interpolation='nearest')
	gl_imshow(model.T, ax=ax2, origin='image',vmin=vmin, vmax=vmax, 
		   interpolation='nearest')
	gl_imshow(diff .T, ax=ax3, origin='image',vmin=vmin, vmax=vmax, 
		   interpolation='nearest')
	
	ax1.set_title('data')
	ax2.set_title('model')
	ax3.set_title('diff')

	plt.show()
