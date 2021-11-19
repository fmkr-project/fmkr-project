import numpy as np
import imageio as io
import matplotlib.pyplot as pl
from copy import deepcopy
from math import atan2

#cur_im = io.imread("no_kiha.png")
#cur_im = io.imread("dummy2.png") # HD
cur_im = io.imread("nokiha_lr.png")

imem = deepcopy(cur_im)


    

def render():
    "affiche l'image en cours de traitement avec plt"
    pl.imshow(cur_im)
    pl.show()

def normal():
    "affiche l'image d'origine"
    pl.imshow(imem)
    pl.show()

def grayscale(im):
    "passe une image en nuances de gris"
    for li in range(len(im[0])-1):
        for col in range(len(im)-1):
            coord = im[col][li]
            hl = 0.227*coord[0] + 0.587*coord[1] + 0.114*coord[2]
            coord[:3] = hl

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

def histeq(im):
    "égalise l'histogramme de l'image"
    




grayscale(cur_im) # on travaille avec une image en NB pour la suite

gaussian3 = [[1/16,1/8,1/16],[1/8,1/4,1/8],[1/16,1/8,1/16]]

gaussian5 = [[1/273,4/273,7/273,4/273,1/273],\
             [4/273,16/273,26/273,16/273,4/273],\
             [7/273,26/273,41/273,26/273,7/273],\
             [4/273,16/273,26/273,16/273,4/273],\
             [1/273,4/273,7/273,4/273,1/273]]


# Filtres de SOBEL
#sob_v = [[1/4,0,-1/4],[1/2,0,-1/2],[1/4,0,-1/4]]
#sob_h = [[1/4,1/2,1/4],[0,0,0],[-1/4,-1/2,-1/4]]
sob_v = [[-1,0,1],[-2,0,2],[-1,0,1]]
sob_h = [[1,2,1],[0,0,0],[-1,-2,-1]]

# Filtres de PREWITT
prew_h = [[1/3,0,-1/3],[1/3,0,-1/3],[1/3,0,-1/3]]
prew_v = [[1/3,1/3,1/3],[0,0,0],[-1/3,-1/3,-1/3]]

# Autres
sharp = [[0,-1,0],[-1,5,-1],[0,-1,0]]
lap = [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]



def conv(im,mat):
    "applique une matrice de convolution à une image"
    imbis = deepcopy(im) # Copie pour préserver les données
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
            #print(prox)
            #try:
            #except:
            #    hl = 0
            #if hl < 0:
            #    im[col][li][:3] = 0
            #elif hl > 255:
            #    im[col][li][:3] = 255
            #else:
            im[li][col][:3] = int(hl)


def sobcontour(im):
    "technique de détection des contours en utilisant les filtres de SOBEL"
    #conv(im,gaussian5)
    conv(im,gaussian5) # -> floutage plus poussé ?
    
    imv = deepcopy(im) # Copie de l'image pour les filtres de dérivation
    imh = deepcopy(im)
    direc = [[None for col in range(len(im[0])-1)]for li in range(len(im)-1)]
    # Création d'un tableau vide de même taille que l'image
    conv(imh,sob_h)
    conv(imv,sob_v)
    
    
    for li in range(len(im[0])-1):
        for col in range(len(im)-1):
            coord = im[col][li]
            dx = imv[col][li][0]
            dy = imh[col][li][0]
            coord[:3] = (dx**2 + dy**2)**0.5
            #direc[col][li] = int((atan2(dy,dx)/(2*np.pi))*360)
    
def laplacian(im):
    "technique de détection en utilisant le laplacien"
    conv(im,gaussian5)
    conv(im,gaussian5)
    conv(im,gaussian5)
    conv(im,lap)


def duochrome(im):
    "transforme une image nuance de gris en image NB"
    for li in range(len(im[0])-1):
        for col in range(len(im)-1):
            coord = im[col][li]
            if coord[0] != coord[1] and coord[0] != coord[2]:
                coord[:3] = 0
            elif coord[0] < 128:
                coord[:3] = 0
            else:
                coord[:3] = 255


def lowbound(im,thr=50):
    "élimine les nuances de gris trop proches du blanc"
    for li in range(len(im[0])-1):
        for col in range(len(im)-1):
            coord = im[col][li]
            if coord[0] > thr:
                coord[:3] = 255

