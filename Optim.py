import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import binary_closing, grey_closing

def LineModel(pars, Nx=100, Ny=100):
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

    The input coordinates lie in the square [0,1]^2, although the image may have
    different resolutions Nx, Ny.
    """
    Offset, Angle, Sky, Sig, h, x1, x2= pars

    m = -np.tan(np.deg2rad(Angle))
    b = 0.2

    y1 = m * x1 + b
    y2 = m * x2 + b

    m      = (y2 - y1) / (x2 - x1)

    X, Y = np.mgrid[0:1:Nx*1j,0:1:Ny*1j]
  
    d = abs((x2-x1)*(y1-Y) - (y2-y1)*(x1-X)) / np.sqrt((x2-x1)**2 + (y2-y1)**2)
    A = h / np.sqrt(2*np.pi*Sig**2) * np.exp(-d**2 / (2*Sig**2))
    A[y1 - (X-x1)/m > Y] = 0.0
    A[y2 - (X-x2)/m < Y] = 0.0
    return A

def Optim(pars, data):
    """
    Tests a line-model fit with endpoints using the scipy.fmin. Optimizes over
    all parameters of the line. 
    """
    from scipy.optimize import leastsq, fmin

    def cost(pars):
        print "called with pars", pars
        return ((data - LineModel(pars))**2).sum()

    v = fmin(cost, pars)
    return v 

def main():
    """
    Tests a line-model fit over the line endpoints using the scipy.fmin
    routine. Intensity and width are kept fixed.
    """
    Nx, Ny = [100,100]
    
    v_real = [50., -30., 1.0, .005, 1.0, 0.2, 0.8]

    data = LineModel(v_real)

    v_guess = [50., -30., 1.0, .005, 1.0, 0.5, 0.5]
    
    v = Optim(v_guess,data)
    model = LineModel(v)

    fig = plt.figure()
    ax1 = fig.add_subplot('221')
    ax2 = fig.add_subplot('222')
    ax3 = fig.add_subplot('212')

    diff = data - model

    vmin = np.percentile(data,1.)
    vmax = np.percentile(data,99.)
    
    plt.gray()
    ax1.imshow(data .T, origin='image',vmin=vmin, vmax=vmax, interpolation='nearest')
    ax2.imshow(model.T, origin='image',vmin=vmin, vmax=vmax, interpolation='nearest')
    ax3.imshow(diff .T, origin='image',vmin=vmin, vmax=vmax, interpolation='nearest')

    ax1.set_title('data')
    ax2.set_title('model')
    ax3.set_title('diff')

    plt.show()
    
if __name__ == "__main__":
    main()
