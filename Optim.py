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
    -------------------------------------------------------------------------

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

def GenerateInfo(image):
    Nx, Ny = image.shape
    xi = np.linspace(1, Nx, Nx)
    yi = np.linspace(1, Ny, Ny)
    xj = xi[:,np.newaxis]
    yj = yi[np.newaxis,:]
    X, Y = np.mgrid[0:1:Nx*1j,0:1:Ny*1j]        
    X = X*Nx
    Y = Y*Ny
    info  = Nx, Ny, xj, yj, X, Y
    return info

def Model(pars, info):
    """   
    Returns: model image of line. Generates image using 7 parameters:
    offset, angle, thickness, normalization, sky, and left/right endpoints 
    along x coordinate. The line is a gaussian with given thickness
    normalization. 
    """
    Offset, Angle, sky, sig, Norm, x1, x2 = pars
    Nx, Ny, xj, yj, X, Y = info
    m = -np.tan(np.deg2rad(Angle))
    b = Offset/np.cos(np.deg2rad(Angle))+0.5*Ny-.5*Nx*m
    xp = (xj + m*(yj - b)) / (m**2 + 1)
    yp = m*xp + b
    
    x1 = Nx*x1
    x2 = Nx*x2
    y1 = m*x1+b
    y2 = m*x2+b

    r  = np.sqrt((yp - yj)**2 + (xp - xj)**2)

    gs = Norm * np.exp(-0.5 * r**2 / sig**2)
    gs[y1 - (X - x1)/abs(m) > Y]  = 0.0 
    gs[y2 - (X - x2)/abs(m) < Y]  = 0.0
    gs += sky
    return gs


def Optim(pars, data):
    """
    Tests a line-model fit with endpoints using the scipy.fmin. Optimizes over
    all parameters of the line. 
    """
    from scipy.optimize import leastsq, fmin

    info = GenerateInfo(data)

    def cost(pars, data, info):
        #print "called with pars", pars
        return ((data - Model(pars,info))**2).sum()

    v = fmin(cost, pars, args = (data, info))
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
    
    info = GenerateInfo(data)
    
    v = Optim(v_guess,data)
    model = Model(v,info)

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
