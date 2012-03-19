#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import leastsq
from scipy.ndimage import binary_closing, grey_closing


def LineModel(pt1, pt2, Nx=100, Ny=100, h=1.0, sig=0.025):
    """
Generates an image (2d numpy array) of a smeared out line connecting two
points.

Parameters:
----------------------------------------------------------------------------

pt1, pt2 ... (x1, y1), (x2, y2) points through whigh the line passes
Nx, Ny ... resolution in x and y direction
h ... line intensity
sig ... line width

The input coordinates lie in the square [0,1]^2, although the image may have
different resolutions Nx, Ny.
"""

    x1, y1 = pt1
    x2, y2 = pt2
    m = (y2 - y1) / (x2 - x1)

    X, Y = np.mgrid[0:1:Nx*1j,0:1:Ny*1j]

    # --------------------------------------------------------------------------
    # Perpendicular distance to the line from a point:
    #
    # http://mathworld.wolfram.com/Point-LineDistance2-Dimensional.html
    #
    # --------------------------------------------------------------------------
    d = abs((x2-x1)*(y1-Y) - (y2-y1)*(x1-X)) / np.sqrt((x2-x1)**2 + (y2-y1)**2)

    A = h / np.sqrt(2*np.pi*sig**2) * np.exp(-d**2 / (2*sig**2))
    A[y1 - (X-x1)/m > Y] = 0.0
    A[y2 - (X-x2)/m < Y] = 0.0

    return A


def LineModel_test():
    im = LineModel([0.2, 0.2], [0.8, 0.4], sig=0.025)
    plt.imshow(im.T, origin='image', interpolation='nearest')
    plt.show()


def FitLine_test1():
    """
Tests a line-model fit over the intensity and width. Endpoints are left
fixed.
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
Tests a line-model fit over the line endpoints. Intensity and width are kept
fixed.

*** Presently, this test fails miserably ***
"""

    h, sig = 1.0, 0.025
    v_real = [0.2, 0.2, 0.8, 0.4] # x1, y1, x2, y2

    data = LineModel([v_real[0], v_real[1]], [v_real[2], v_real[3]], h=h, sig=sig).flatten()

    mod1 = lambda v: LineModel([v[0], v[1]], [v_real[2], v_real[3]]).flatten()

    def cost1(pars):
        print "First Endpoint Pars:", pars
        return data - mod1(pars)
    
    def cost2(pars):
        print "Second Endpoint Pars:", pars
        return data - mod2(pars)

    v_guess1 = [0.0, 0.0] # x1, y1
    v1,suc1 = leastsq(cost1, v_guess1, ftol=1e-12, xtol=1e-12, maxfev = 5000)
#    model1 = mod(v1)

    vg2 = [1.0, 1.0]
    mod2 = lambda v: LineModel([v1[0], v1[1]], [v[0],v[1]]).flatten()

    v2,suc2 = leastsq(cost2, vg2, ftol=1e-12, xtol=1e-12, maxfev = 5000)
    model = mod2(v2)
    print "--------------------------------------------"
    print "--First  Endpoint:",v1
    print "--Second Endpoint:",v2

#    print "success:", success

    fig = plt.figure()
    ax1 = fig.add_subplot('121')
    ax2 = fig.add_subplot('122')

    ax1.imshow(data .reshape(100,100).T, origin='image', interpolation='nearest')
    ax2.imshow(model.reshape(100,100).T, origin='image', interpolation='nearest')

    ax1.set_title('data')
    ax2.set_title('model')

    plt.show()


def Closing_test():
    """
This doesn't really work ;(
"""
    model = 1.0 - LineModel([0.2, 0.2], [0.8, 0.4], h=1.0, sig=0.0025)
    close = grey_closing(model, size=(4,4))

    fig = plt.figure()
    ax1 = fig.add_subplot('121')
    ax2 = fig.add_subplot('122')

    ax1.imshow(model.T, origin='image', interpolation='nearest')
    ax2.imshow(close.T, origin='image', interpolation='nearest')
    plt.show()


if __name__ == "__main__":
    FitLine_test2()
