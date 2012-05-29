"""
Histeq.py is part of elmpy, a module that eliminates astronomical trails. 
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
