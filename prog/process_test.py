import numpy as np
import matplotlib.pyplot as pl
import imageio as io
from scipy import ndimage as nd
import os

from copy import deepcopy
from math import atan2

os.chdir("prog/")

### Filtres
gaussian3 = np.array([[1/16, 1/8, 1/16],
                      [1/8, 1/4, 1/8],
                      [1/16, 1/8, 1/16]])

gaussian5 = np.array(
            [[1/273,4/273,7/273,4/273,1/273],\
             [4/273,16/273,26/273,16/273,4/273],\
             [7/273,26/273,41/273,26/273,7/273],\
             [4/273,16/273,26/273,16/273,4/273],\
             [1/273,4/273,7/273,4/273,1/273]]
             )

# Filtres de SOBEL
#sob_v = [[1/4,0,-1/4],[1/2,0,-1/2],[1/4,0,-1/4]]
#sob_h = [[1/4,1/2,1/4],[0,0,0],[-1/4,-1/2,-1/4]]
sob_v = np.array([[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]])
sob_h = np.array([[1, 2, 1],
                  [0, 0, 0],
                  [-1, -2, -1]])

# Filtres de PREWITT
prew_h = np.array([[1/3, 0, -1/3],
                   [1/3, 0, -1/3],
                   [1/3, 0, -1/3]])
prew_v = np.array([[1/3, 1/3, 1/3],
                   [0, 0, 0],
                   [-1/3, -1/3, -1/3]])

# Autres
sharp = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
lap = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])


### Initialisation des images
#cur_im = io.imread("no_kiha.png")
#cur_im = io.imread("dummy2.png") # HD
cur_im = np.array(io.imread("nokiha_lr.png"))
imem = deepcopy(cur_im)


def render(im):
    """Affichage d'une image avec Matplotlib"""
    pl.imshow(im)
    pl.show()

def twodim(im):
    """Retourne l'array 2D correspondant aux nuances de gris de l'image passée en argument"""
    hl = [[0 for j in range(len(im[0]-1))] for i in range(len(im)-1)]
    for li in range(len(im[0])-1):
        for col in range(len(im)-1):
            ab = []
            coord = im[col][li]
            p = 0.227*coord[0] + 0.587*coord[1] + 0.114*coord[2]
            ab.append(p)
        hl.append(np.array(ab))
    return(np.array([[0.227*im[i][j][0] + 0.587*im[i][j][1] + 0.114*im[i][j][2] for j in range(len(im[0])-1)] for i in range(len(im)-1)]))

def threedim(im):
    """Revient à un array à trois dimensions afin d'afficher les nuances de gris"""
    return(np.array([[[int(im[i][j]) for _ in range(3)] for j in range(len(im[0])-1)] for i in range(len(im)-1)]))

def new_mean(im,var,thr = 128):
    "modifie la luminosité de l'image"
    for li in range(len(im[0])-1):
        for col in range(len(im)-1):
            coord = im[col][li]
            if coord[0] > thr:
                is_bright = 1
            else:
                is_bright = -1
            coord[:3] = coord[:3] + var*(is_bright)

def conv(im,mat):
    "applique une matrice de convolution à une image"
    imbis = deepcopy(im) # Copie pour préserver les données
    maxi = -1
    center = len(mat) // 2
    size = center
    for li in range(len(im)-center):
        for col in range(len(im[0])-center):
            coord = im[li][col]
            coef = coord[0]
            mat_flat = [row[k] for row in mat for k in range(len(row))]
            # on aplatit la matrice de convolution
            prox = [imbis[j+li][i+col][0]\
                    for i in range(-center, center+1)\
                    for j in range(-center, center+1)]
            hl = sum([prox[i]*mat_flat[i] for i in range(len(prox))])
            if hl > maxi:
                maxi = hl
            if hl < 0:
                hl = 0
            hl = int(255*hl/maxi)
            #normalized[li][col] = int(255*hl/maxi)
            im[li][col][:3] = hl

def sconv(im, filter):
    """Convolution avec SciPy"""
    nd.convolve(im, filter)

def sobcontour(im):
    "technique de détection des contours en utilisant les filtres de SOBEL"
    imv = deepcopy(im) # Copie de l'image pour les filtres de dérivation
    print(imv)
    imh = deepcopy(im)

    direc = [[None for col in range(len(im[0])-1)]for li in range(len(im)-1)]
    # Création d'un tableau vide de même taille que l'image
    sconv(imh,sob_h)
    sconv(imv,sob_v)

    for li in range(len(im[0])-1):
        for col in range(len(im)-1):
            dx = imv[col][li]
            dy = imh[col][li]
            im[col][li] = (dx**2 + dy**2)**0.5

def rebalance(im):
    """Modifie l'histogramme de l'image pour que son maximum soit mis au blanc"""
    maxi = -1
    for li in range(len(im)):
        for col in range(len(im[0])):
            if im[li][col] > maxi:
                maxi = im[li][col]
    for li in range(len(im)):
        for col in range(len(im[0])):
            im[li][col] = int(255*im[li][col] / maxi)
    
def laplacian(im):
    "technique de détection en utilisant le laplacien"
    conv(im,gaussian5)
    conv(im,lap)

def lowbound(im,thr=50):
    "élimine les nuances de gris trop proches du blanc"
    for li in range(len(im[0])-1):
        for col in range(len(im)-1):
            coord = im[col][li]
            if coord[0] > thr:
                coord[:3] = 255

def mainloop():
    """Boucle principale du programme"""
    print(cur_im)
    gsim = twodim(cur_im) # on travaille avec une seule coordonnée
    print(gsim)

    sobcontour(gsim)
    rebalance(gsim)
    render(threedim(gsim))

mainloop()