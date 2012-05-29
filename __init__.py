"""
MODULE: elmpy
Eliminates astronomical trails from astronomical images
Copyright (C) 2012  Gregory Lemberskiy

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA
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

	gl_imshow(image, ax=ax1, origin='image',vmin=vmin, vmax=vmax, 
		   interpolation='nearest')
	gl_imshow(model, ax=ax2, origin='image',vmin=vmin, vmax=vmax, 
		   interpolation='nearest')
	gl_imshow(diff , ax=ax3, origin='image',vmin=vmin, vmax=vmax, 
		   interpolation='nearest')
	
	ax1.set_title('data')
	ax2.set_title('model')
	ax3.set_title('diff')

	plt.show()
