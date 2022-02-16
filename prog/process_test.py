import numpy as np
import matplotlib.pyplot as plt
import imageio as io
from scipy import ndimage as nd

from copy import deepcopy

class Kernel():
    """Classe des matrices de convolution"""
    def __init__(self, contents):
        self.contents = np.array(contents)

class Image():
    """Classe des images à traiter"""
    IMAGES_PATH = "res/"

    # Paramètres de l'hystérésis
    UPPER_COLOR = 255
    LOWER_COLOR = 45
    UPPER_RATIO = 0.33
    LOWER_RATIO = 0.12

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
        def nms(im, grad):
            """Non-Maximum Suppression, conservation des maxima locaux de l'image uniquement"""
            x,y = im.shape
            corrected = np.zeros((x,y))

            for i in range(x):
                for j in range(y):
                    try:
                        # Obtention de la valeur des pixels voisins selon la direction
                        pix1 = 255
                        pix2 = 255
                        current_grad = grad[i][j]
                        if np.abs(current_grad) <= np.pi/8 or np.abs(current_grad) >= 7*np.pi/8:               # horizontal
                            pix1 = im[i][j+1]
                            pix2 = im[i][j-1]
                        elif np.pi/8 <= current_grad <= 3*np.pi/8 or -7*np.pi/8 <= current_grad <= -5*np.pi/8:       # oblique, haut à droite
                            pix1 = im[i+1][j-1]
                            pix2 = im[i-1][j+1]
                        elif 3*np.pi/8 <= current_grad <= 5*np.pi/8 or -5*np.pi/8 <= current_grad <= -3*np.pi/8:     # vertical
                            pix1 = im[i+1][j]
                            pix2 = im[i-1][j]
                        elif 5*np.pi/8 <= current_grad <= 7*np.pi/8 or -7*np.pi/8 <= current_grad <= -5*np.pi/8:       # oblique, haut à gauche
                            pix1 = im[i+1][j+1]
                            pix2 = im[i-1][j-1]
                        
                        # Mise à jour de la valeur du pixel courant
                        if im[i][j] >= pix1 and im[i][j] >= pix2:
                            corrected[i][j] = im[i][j]
                        else:
                            corrected[i][j] = 0
                    except:     # On ne traite pas les bords
                        pass
            return(corrected)
        
        def cat(im):
            """Sépare les contours en 3 catégories : contours forts, contours faibles et contours peu intéressants"""
            bound_high = im.max() * self.UPPER_RATIO
            bound_low = self.LOWER_RATIO * bound_high
            x,y = im.shape
            catted = np.zeros((x,y))

            # Détermination des coordonnées des 3 types de contours
            high_x, high_y = np.where(im >= bound_high)
            low_x, low_y = np.where((bound_low <= im) & (im < bound_high))      # On utilise & car im est un array, donc a 2 dimensions

            # Mise à jour des valeurs
            catted[high_x, high_y] = self.UPPER_COLOR
            catted[low_x, low_y] = self.LOWER_COLOR

            return(catted)
        
        def hyster(im):
            """Fonction d'hystérésis qui transforme les pixels faibles en pixels forts s'ils côtoient un pixel fort"""
            x,y = im.shape
            upper = self.UPPER_COLOR
            lower = self.LOWER_COLOR
            for i in range(x):
                for j in range(y):
                    if im[i][j] == lower:
                        try:
                            if im[i+1][j-1] == upper or im[i+1][j] == upper or im[i+1][j+1] == upper or im[i][j-1] == upper or im[i][j+1] == upper or im[i-1][j-1] == upper or im[i-1][j] == upper or im[i-1][j+1] == upper:
                                im[i][j] = upper
                            else:
                                im[i][j] = 0
                        except:
                            pass
            return(im)

        # Filtrage de SOBEL
        image_dx = deepcopy(self)       # Dérivée horizontale
        image_dy = deepcopy(self)       # Dérivée verticale
        image_dx.sconv(sob_h)
        image_dy.sconv(sob_v)
        self.grad = np.arctan2(image_dy.acc, image_dx.acc)     # Image du gradient d'intensité
        self.acc = (image_dx.acc**2 + image_dy.acc**2) ** 0.5
        self.acc = self.acc * (255/self.acc.max())      # Normalisation sur [0, 255]

        # Suppression des non-maxima locaux
        self.acc = nms(self.acc, self.grad)

        # Catégorisation en contours faibles et contours forts
        self.acc = cat(self.acc)

        # Hystérésis
        self.acc = hyster(self.acc)

        # Passage en trois dimensions pour l'affichage
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
sob_h = Kernel([[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]])
sob_v = Kernel([[1, 2, 1],
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
    print("Processing...")

    ### Initialisation des images
    #cur_im = io.imread("no_kiha.png")
    #cur_im = io.imread("dummy2.png") # HD
    cur_im = Image("nokiha_lr")
    cur_im.canny()
    render(cur_im.acc)


mainloop()
