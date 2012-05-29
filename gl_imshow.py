"""
gl_imshow.py is part of elmpy, a module that eliminates astronomical trails. 
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
    
    A = np.exp(-d**2 / (2*60**2))+500.

    fig1 = plt.figure(figsize=(8,5))
    ax1  = fig1.add_subplot(121)
    ax2  = fig1.add_subplot(122)

    ax1.set_title('Gaussian [imshow()]')
    ax1.imshow(A.T, cmap=plt.cm.gray, interpolation= 'nearest')
    
    ax2.set_title('Gaussian [gl_imshow()]')
    gl_imshow(A,ax=ax2,cmap=plt.cm.gray, interpolation='nearest',origin='image')

    return plt.show()

if __name__ == "__main__":
    main()
