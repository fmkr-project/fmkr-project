import numpy as np
import matplotlib.pyplot as plt
import imageio as io
from scipy import ndimage as nd

from copy import deepcopy
from math import atan2

class Kernel():
    """Classe des matrices de convolution"""
    def __init__(self, contents):
        self.contents = np.array(contents)

class Image():
    """Classe des images à traiter"""
    IMAGES_PATH = "res/"

    def __init__(self, path):
        self.rgb = np.array(io.imread(f"{self.IMAGES_PATH}{path}.png"))
        self.gs = np.array([[[int(0.227*self.rgb[i][j][0] + 0.587*self.rgb[i][j][1] + 0.114*self.rgb[i][j][2]) for _ in range(3)]
                            for j in range(len(self.rgb[0])-1)] for i in range(len(self.rgb)-1)])       # Array à 3 dimensions
        self.monodim = np.array([[int(0.227*self.rgb[i][j][0] + 0.587*self.rgb[i][j][1] + 0.114*self.rgb[i][j][2])
                            for j in range(len(self.rgb[0])-1)] for i in range(len(self.rgb)-1)])       # Array à une dimension
    
    def sconv(self, ker):
        """Convolution avec SciPy"""
        self.acc = nd.convolve(self.monodim, ker.contents)       # Le résultat est stocké dans un accumulateur propre à l'image
    
    
    def canny(self):
        """Application du filtre de CANNY sur l'image"""
        image_dx = deepcopy(self)
        image_dy = deepcopy(self)
        image_dx.sconv(sob_h)
        image_dy.sconv(sob_v)
        self.acc = (image_dx.acc**2 + image_dy.acc**2) ** 0.5
        self.acc = self.acc * (255/self.acc.max())      # Normalisation sur [0, 255]
        self.acc = threedim(self.acc)


def render(im):
    """Affichage d'une image avec Matplotlib"""
    plt.imshow(im)
    plt.show()

gaussian3 = Kernel([[1/16, 1/8, 1/16],
                    [1/8, 1/4, 1/8],
                    [1/16, 1/8, 1/16]])

gaussian5 = Kernel([[1/273,4/273,7/273,4/273,1/273],\
                    [4/273,16/273,26/273,16/273,4/273],\
                    [7/273,26/273,41/273,26/273,7/273],\
                    [4/273,16/273,26/273,16/273,4/273],\
                    [1/273,4/273,7/273,4/273,1/273]]
            )

# Filtres de SOBEL
sob_v = Kernel([[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]])
sob_h = Kernel([[1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1]])

# Filtres de PREWITT
prew_h = Kernel([[1/3, 0, -1/3],
                [1/3, 0, -1/3],
                [1/3, 0, -1/3]])
prew_v = Kernel([[1/3, 1/3, 1/3],
                [0, 0, 0],
                [-1/3, -1/3, -1/3]])

# Autres
sharp = Kernel([[0,-1,0],[-1,5,-1],[0,-1,0]])
lap = Kernel([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])


def threedim(im):
    """Revient à un array à trois dimensions afin d'afficher les nuances de gris"""
    return(np.array([[[int(im[i][j]) for _ in range(3)] for j in range(len(im[0])-1)] for i in range(len(im)-1)]))

def mainloop():
    """Boucle principale du programme"""
    ### Initialisation des images
    #cur_im = io.imread("no_kiha.png")
    #cur_im = io.imread("dummy2.png") # HD
    cur_im = Image("nokiha_lr")

    cur_im.canny()
    render(cur_im.acc)


mainloop()
