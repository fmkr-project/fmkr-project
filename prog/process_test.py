import imageio as io
import matplotlib.pyplot as pl
from copy import copy

cur_im = io.imread("dummy.png")
imem = copy(cur_im)

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

gaussian3 = [[1/9 for i in range(3)] for i in range(3)]

sob_v = [[1/4,0,-1/4],[1/2,0,-1/2],[1/4,0,-1/4]]
sob_h = [[1/4,1/2,1/4],[0,0,0],[-1/4,-1/2,-1/4]]
prew_h = [[1/3,0,-1/3],[1/3,0,-1/3],[1/3,0,-1/3]]
prew_v = [[1/3,1/3,1/3],[0,0,0],[-1/3,-1/3,-1/3]]
pb1 = [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]

def conv(im,mat):
    "applique une matrice de convolution à une image"
    center = len(mat) // 2
    size = center
    for li in range(len(im[0])-1):
        for col in range(len(im)-1):
            coord = im[col][li]
            coef = coord[0]
            try:
                hl = sum([im[j+col][i+li][0]*mat[i+center][j+center]\
                        for i in range(-center, center+1)\
                        for j in range(-center, center+1)])
            except:
                hl = 0
            if hl < 0:
                im[col][li][:3] = 0
            elif hl > 255:
                im[col][li][:3] = 255
            else:
                im[col][li][:3] = int(hl)


def prew_cont(im):
    "technique de détection des contours en utilisant les filtres de PREWITT"
    grayscale(im)
    conv(im,gaussian3)
    #conv(im,gaussian3) -> floutage plus poussé ?
    
    imv = copy(im) # Copie de l'image pour les filtres de dérivation
    imh = copy(im)
    conv(imh,sob_h)
    conv(imv,sob_v)

    for li in range(len(im[0])-1):
        for col in range(len(im)-1):
            coord = im[col][li]
            dx = imv[col][li][0]
            dy = imh[col][li][0]
            coord[:3] = (dx**2 + dy**2)**0.5








