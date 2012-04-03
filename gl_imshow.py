import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import median_filter

def gl_imshow(image,vmin=None, vmax=None, ax=None,freeze=None,**kwargs):
    """
    Returns: pyplot.imshow() with vmin/vmax set to the appropriate quantile.
    Uses ax parameter to determine whether this function is a subplot. 
    ----------------------------------------------------------------------
    Parameters: image
    """
    if ax is None:
        canvas = plt
    else:
        canvas = ax
    if freeze == None:
        vmin = np.percentile(image,1.)
        vmax = np.percentile(image,99.)
    else:
        vmin = np.percentile(freeze,1.)
        vmax = np.percentile(freeze,99.)
    return canvas.imshow(image.T,vmin=vmin,vmax=vmax,**kwargs)

def main():
    
    d = np.arange(500).reshape(25,20)
    
    A = np.exp(-d**2 / (2*60**2))+50.
    B = np.exp(-d**2 / (2*20**2))-50.

    fig1 = plt.figure(figsize=(8,5))
    ax1  = fig1.add_subplot(121)
    ax2  = fig1.add_subplot(122)

    ax1.set_title('Subplot 1')
    gl_imshow(A,ax=ax1, cmap=plt.cm.gray, interpolation= 'nearest')
    
    ax2.set_title('Subplot 2')
    gl_imshow(B,ax=ax2,cmap=plt.cm.gray, interpolation='nearest',origin='image')

    return plt.show()

if __name__ == "__main__":
    main()
