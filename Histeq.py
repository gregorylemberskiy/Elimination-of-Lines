import numpy as np
from matplotlib import pyplot as plt

def Histeq(image):
    """
    Returns: nd.array with lowered intensities. This function serves to
    optimize the performance of the hough transform especially in the
    presence of bright stellar objects.
    ----------------------------------------------------------------------
    Parameters: image 
    """
    foo = np.zeros(image.size).astype(float)
    foo[np.argsort(image.flatten())] = np.linspace(-1.,1.,image.size)
    return foo.reshape(image.shape)

def main():

    d = np.arange(500).reshape(25,20)
    A = np.exp(-d**2 / (2*40**2))+50.

    fig  = plt.figure()
    ax1 = fig.add_subplot('121')
    ax2 = fig.add_subplot('122')

    plt.gray()
    ax1.imshow(A ,  origin='image', interpolation='nearest')
    ax2.imshow(Histeq(A),origin='image', interpolation='nearest')
    
    ax1.set_title('Original')
    ax2.set_title('Histeq')
    plt.show()

if __name__ == "__main__":
    main()
