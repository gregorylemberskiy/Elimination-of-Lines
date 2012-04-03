import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import median_filter

def Highpass(image):
    """
    Returns: image after passing it through a highpass filter. 
    ----------------------------------------------------------------------
    Parameters: image
    """
    return image - median_filter(image,size=9)

def main():

    d = np.arange(500).reshape(25,20)
    A = np.exp(-d**2 / (2*40**2))+50.

    fig  = plt.figure()
    ax1 = fig.add_subplot('121')
    ax2 = fig.add_subplot('122')

    plt.gray()
    ax1.imshow(A ,  origin='image', interpolation='nearest')
    ax2.imshow(highpass(A),origin='image', interpolation='nearest')
    
    ax1.set_title('Original')
    ax2.set_title('Histeq')
    plt.show()

if __name__ == "__main__":
    main()
