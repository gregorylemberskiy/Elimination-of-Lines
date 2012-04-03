import numpy as np
from matplotlib import pyplot as plt

def LineModel(pt1, pt2, Nx=100, Ny=100, h=1.0, sig=0.625):
    """
    Generates an image (2d numpy array) of a smeared out line connecting two
    points.

    Parameters:
    -------------------------------------------------------------------------

    pt1, pt2 ... (x1, y1), (x2, y2) points through whigh the line passes
    Nx, Ny ... resolution in x and y direction
    h ... line intensity
    sig ... line width
    
    The input coordinates lie in the square [0,1]^2, although the image
    may have different resolutions Nx, Ny.

    Parameters:
    ----------------------------------------------------------------------------

    pt1, pt2 ... (x1, y1), (x2, y2) points through whigh the line passes
    Nx, Ny ... resolution in x and y direction
    h ... line intensity
    sig ... line width

    The input coordinates lie in the square [0,1]^2.
    """

    x1, y1 = pt1
    x2, y2 = pt2
    
    X, Y = np.mgrid[0:1:Nx*1j,0:1:Ny*1j]
    X = X*Nx
    Y = Y*Ny
    
    m = (y2 - y1) / (x2 - x1)
    d = abs((x2-x1)*(y1-Y) - (y2-y1)*(x1-X)) / np.sqrt((x2-x1)**2 + (y2-y1)**2)
    
    A = h / np.sqrt(2*np.pi*sig**2) * np.exp(-d**2 / (2*sig**2))

    A[y1 - (X-x1)/m > Y] = 0.0
    A[y2 - (X-x2)/m < Y] = 0.0
    return A

def hough(image):
    """
    Returns: 
    himage - The nd.array of the hough image.
    Offset - The line's offset from the center of the image.
    Maxbindex - The integer index of the maximum bin.
    Maxangleindex - The integer index of the maximum angle.
    Bins - Returns 1d.array containing bin centers. 
    ----------------------------------------------------------------------
    Parameters: image
    """
## HOUGH TRANSFORM, FUNCTION THAT FINDS LINES IN THE IMAGE
    Nx, Ny = image.shape

    Angle  = 180.
    AStep  = 180. # number of angle steps
    BStep  = 1.   # size of b steps
    x, y   = np.mgrid[:Nx,:Ny]
    x     -= 0.5 * Nx
    y     -= 0.5 * Ny
    bins   = np.arange(-0.5 * Ny, 0.5 * Ny, BStep)
    Theta  = np.deg2rad(np.linspace(0., Angle, AStep))
#    histeqdata = histeq(image) #For Complicated Images
    histeqdata = image          #For Simple      Images
    himage = []
    for t in Theta:
        X  = np.sin(t) * x + np.cos(t) * y
        N, bins = np.histogram(X, bins=bins, weights=histeqdata)
        himage.append(N)
    himage = np.array(himage)
    bins   = 0.5*(bins[1:] + bins[:-1])
    indx   = np.argmax(himage)
    Maxangleindex = indx / len(bins)
    Maxbindex = indx % len(bins)
    Maxangle  = Maxangleindex*Angle/AStep
    Offset = bins[Maxbindex]
    """
    Stretching Image so that it scale to Bins. Useful when interpreting the
    hough image to reproduce the line.
    """
    z = np.zeros([himage.shape[0], himage.shape[1]*2])
    z[:,0::2] = himage
    z[:,1::2] = himage
    himage = z
    return himage, Offset, Maxbindex, Maxangleindex, bins

def Main():
    """
    Runs the hough transform on a generated line. 
    """

    Nx, Ny = [100,100]

    data = LineModel([20.0,20.0],[90.0,40.0])
  
    himage, Offset, Maxbindex, Angle, bins = hough(data)
    
    fig1 = plt.figure(figsize=(8,5))
    ax1  = fig1.add_subplot(121)
    ax2  = fig1.add_subplot(122)

# Regular Image
    ax1.set_title('Regular Image')
    ax1.imshow(data.T, cmap=plt.cm.gray, interpolation= 'nearest')
    
    def Line_Pars(himage,Offset,Angle,Nx=Nx,Ny=Ny):
        m = -np.tan(np.deg2rad(Angle))
        b = Offset/np.cos(np.deg2rad(Angle))+0.5*Ny-0.5*Nx*m
        y = m*np.arange(Nx)+b
        return y

    line = Line_Pars(himage, Offset, Angle)

    ax1.plot(np.arange(Nx),line,'--r')
    ax1.set_xlim(0,Nx)
    ax1.set_ylim(0,Ny)
  
# Hough Image
    ax2.set_title('Hough Image')
    ax2.imshow(himage, cmap=plt.cm.gray, interpolation='nearest',origin='image'
              ,extent=[bins[0],bins[-1],0,180],aspect=len(bins)/180.)
    ax2.plot(Offset,Angle,marker='x',color='r')
    ax2.set_xlabel('Bins')
    ax2.set_ylabel('Angles')
    ax2.axis([bins[0],bins[-1],0,180])

    plt.show()

if __name__ == "__main__":
    Main()
