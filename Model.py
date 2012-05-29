"""
Model.py is part of elmpy, a module that eliminates astronomical trails. 
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
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""

import numpy as np
from matplotlib import pyplot as plt
import Histeq

def GenerateData(pt1, pt2, Nx=100, Ny=100, h=1.0, sig=0.625):
    """
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

def GInfo(image):
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


def Main():
    # Creating Image to Model
    Nx, Ny = [100,100]
    data = LineModel([20.0,20.0],[90.0,40.0])

    #Modeling image
    pars  = [15., 160.0, 0.0, 1.0, 30.0, 0.19, 0.90]
    info = GInfo(data)
    Model_data = Model(pars,info)

    """
    Runs the hough transform on a generated line. 
    """
  
    fig1 = plt.figure(figsize=(8,4))

    ax1  = fig1.add_subplot(121)
    ax2  = fig1.add_subplot(122)

# Regular Image
    ax1.set_title('Generated Data')
    ax1.imshow(data.T, cmap=plt.cm.gray, interpolation= 'nearest')
    ax1.set_xlim(0,Nx)
    ax1.set_ylim(0,Ny)

    ax2.set_title('Unoptimized Model')
    ax2.imshow(Model_data.T, cmap=plt.cm.gray, interpolation= 'nearest')
    ax2.set_xlim(0,Nx)
    ax2.set_ylim(0,Ny)
      
    plt.show()

if __name__ == "__main__":
    Main()
