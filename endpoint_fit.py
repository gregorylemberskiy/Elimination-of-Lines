import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import leastsq, fmin
from scipy.ndimage import binary_closing, grey_closing


def LineModel(pt1, pt2, Nx=100, Ny=100, h=1.0, sig=0.025):
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
    """

    x1, y1 = pt1
    x2, y2 = pt2
    m      = (y2 - y1) / (x2 - x1)

    X, Y = np.mgrid[0:1:Nx*1j,0:1:Ny*1j]
  
    d = abs((x2-x1)*(y1-Y) - (y2-y1)*(x1-X)) / np.sqrt((x2-x1)**2 + (y2-y1)**2)
    A = h / np.sqrt(2*np.pi*sig**2) * np.exp(-d**2 / (2*sig**2))
    A[y1 - (X-x1)/m > Y] = 0.0
    A[y2 - (X-x2)/m < Y] = 0.0
    return A


def LineModel_test():
    im = LineModel([0.2, 0.2], [0.8, 0.4], sig=0.025)
    plt.imshow(im.T, origin='image', interpolation='nearest')
    plt.show()


def FitLine_test0():
    """
    Tests a line-model fit over the intensity and width using
    scipy.fmin. Endpoints are left fixed.
    """

    pt1, pt2 = [0.2, 0.2], [0.8, 0.4]
    v_real = [1.0, 0.025] # h, sig

    mod = lambda v: LineModel(pt1, pt2, h=v[0], sig=v[1]).flatten()
    data = mod(v_real).flatten()

    def cost(pars):
        print "called with pars", pars
        return data - mod(pars)

    v_guess = [0.8, 0.1] # h, sig
    v = fmin(cost, v_guess)
    model = mod(v)

    fig = plt.figure()
    ax1 = fig.add_subplot('121')
    ax2 = fig.add_subplot('122')
    ax3 = fig.add_subplot('113')

    ax1.imshow(data .reshape(100,100).T, origin='image', interpolation='nearest')
    ax2.imshow(model.reshape(100,100).T, origin='image', interpolation='nearest')
    ax1.set_title('data')
    ax2.set_title('model')
    ax3.set_title('diff')

    plt.show()


def FitLine_test1():
    """
Tests a line-model fit over the intensity and width using
scipy.leastsq. Endpoints are left fixed.
"""

    pt1, pt2 = [0.2, 0.2], [0.8, 0.4]
    v_real = [1.0, 0.025] # h, sig

    mod = lambda v: LineModel(pt1, pt2, h=v[0], sig=v[1]).flatten()
    data = mod(v_real).flatten()

    def cost(pars):
        print "called with pars", pars
        return data - mod(pars)

    v_guess = [0.8, 0.1] # h, sig
    v, success = leastsq(cost, v_guess)
    model = mod(v)

    fig = plt.figure()
    ax1 = fig.add_subplot('121')
    ax2 = fig.add_subplot('122')

    ax1.imshow(data .reshape(100,100).T, origin='image', interpolation='nearest')
    ax2.imshow(model.reshape(100,100).T, origin='image', interpolation='nearest')

    ax1.set_title('data')
    ax2.set_title('model')

    plt.show()


def FitLine_test2():
    """
Tests a line-model fit over the line endpoints using the scipy.fmin
routine. Intensity and width are kept fixed.
"""
#    Nx = 200
#    Ny = 200
    Nx, Ny = [200,200]

    pt1, pt2 = [0.2, 0.2], [0.8, 0.4]
    v_real = [0.2, 0.8] # h, sig
    h = 1.0
    sig = 0.025

    mod = lambda v: LineModel([v[0],pt1[1]], [v[1],pt2[1]], h=h, sig=sig)#.flatten()
    data = mod(v_real)#.flatten()

    def cost(pars):
        print "called with pars", pars
        return (data - mod(pars)).ravel()

    v_guess = [0.5, 0.5] # h, sig
    v, success = leastsq(cost, v_guess)
    model = mod(v)

    """
    pt1, pt2 = [0.2, 0.2], [0.8, 0.4]

    # These are the parameters given by Hough
    m = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
    b = 0.1

    # The actual endpoints of the segment
    v_real = [0.2, 0.8]

    def mod(v):
        x1, x2 = v[0], v[1]
        y1 = m*x1 + b
        y2 = m*x2 + b
        return LineModel([x1, y1], [x2, y2], Nx=Nx, Ny=Ny, h=1.0, sig=0.025).flatten()

    data = mod(v_real).flatten()

    def cost(pars):
        print "Called with Pars", pars
        return abs(data - mod(pars)).sum()

    v_guess = [0.50, 0.50]
#     v,suc = leastsq(cost, v_guess)#, ftol=1e-8)
    v, suc  = leastsq(cost, v_guess, ftol=1e-12, xtol=1e-12, maxfev=1000)
    model = mod(v)
    """

    fig = plt.figure()
    ax1 = fig.add_subplot('221')
    ax2 = fig.add_subplot('222')
    ax3 = fig.add_subplot('212')

    ax1.imshow(data .T, origin= 'image', cmap=plt.cm.gray,interpolation='nearest')
    ax2.imshow(model.T, origin='image', cmap=plt.cm.gray,interpolation='nearest')
    diff = data - model

    ax3.imshow(diff .T, origin= 'image', cmap=plt.cm.gray,interpolation='nearest')

    ax1.set_title('data')
    ax2.set_title('model')
    ax3.set_title('diff')

    plt.show()

if __name__ == "__main__":
    FitLine_test2()
