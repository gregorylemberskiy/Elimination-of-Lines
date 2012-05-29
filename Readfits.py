import numpy as np
import pyfits as fits
import optparse
from matplotlib import pyplot as plt

"""
Running this file in the command line as:
python Readfits.py 
Will result analyze an image with endpoints by default. For additional
options:
python Readfits.py --f directory/filename.gz
"""

def Readfits():
    import sys
    """
    Returns: nd.array containing image data and nd.array containing
    invariance data. The optparse module is applied to process pyfits
    files from the command line.
    ----------------------------------------------------------------------
    Parameters: None
    """
    p = optparse.OptionParser()
    p.add_option('--f', '--filename', default=
                 'NGC_3521_UGC_6150-r.fits.gz')
    options, arguments = p.parse_args()
    fname = options.f
    hdulist         = fits.open(str(fname))    
    image, invar = hdulist[0].data, hdulist[1].data

    return image, invar

def Main():
    """
    Runs the hough transform on a generated line. 
    """

    image, invar = Readfits()

    Nx, Ny = np.shape(image)

    fig1 = plt.figure(figsize=(8,5))
    ax1  = fig1.add_subplot(121)
    ax2  = fig1.add_subplot(122)

    # Image
    ax1.set_title('Image')
    ax1.imshow(image.T)
    ax1.set_xlim(0,Nx)
    ax1.set_ylim(0,Ny)
  
    # Invariance
    ax2.set_title('Invariance')
    ax2.imshow(invar.T)
    ax2.set_xlim(0,Nx)
    ax2.set_ylim(0,Ny)

    plt.show()

if __name__ == "__main__":
    Main()
