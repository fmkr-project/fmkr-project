import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
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

    # Paramètres de l'espace de HOUGH
    THETA_RES = 200      # Nombre de valeurs de θ possibles
    RHO_RES = 200        # Nombre de valeurs de ρ possibles
    INTERSEC_TH = 300    # Nombre minimal de sinusoïdales se coupant en un point pour la recherche des maxima

    def __init__(self, path):
        self.rgb = np.array(io.imread(f"{self.IMAGES_PATH}{path}.png"))
        print("Converting image to GrayScale...")
        self.gs = np.array([[[int(0.227*self.rgb[i][j][0] + 0.587*self.rgb[i][j][1] + 0.114*self.rgb[i][j][2]) for _ in range(3)]
                            for j in range(len(self.rgb[0])-1)] for i in range(len(self.rgb)-1)])       # Array à 3 dimensions
        self.monodim = np.array([[int(0.227*self.rgb[i][j][0] + 0.587*self.rgb[i][j][1] + 0.114*self.rgb[i][j][2])
                            for j in range(len(self.rgb[0])-1)] for i in range(len(self.rgb)-1)])       # Array à une dimension
        self.med = np.mean(self.monodim)
        self.max = self.monodim.max()
        self.shape = self.monodim.shape

    def sconv(self, ker):
        """Convolution avec SciPy"""
        self.acc = nd.convolve(self.monodim, ker.contents)       # Le résultat est stocké dans un accumulateur propre à l'image
    
    def canny(self):
        """Application du filtre de CANNY sur l'image"""
        print("Applying Canny Filter...")
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
            bound_high = im.max() * self.upper_ratio
            bound_low = self.lower_ratio * bound_high
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
                            if im[i+1][j-1] == upper or im[i+1][j] == upper or im[i+1][j+1] == upper or im[i][j-1] == upper\
                            or im[i][j+1] == upper or im[i-1][j-1] == upper or im[i-1][j] == upper or im[i-1][j+1] == upper:
                                im[i][j] = upper
                            else:
                                im[i][j] = 0
                        except:
                            pass
            return(im)

        
        # Ratios pour la catégorisation des contours
        # Moyen d'obtention : on se place autour de la médiane (facteur sigma)
        self.sigma = 0.01
        self.upper_ratio = ((1 + self.sigma) * self.med) / self.max
        self.lower_ratio = ((1 - self.sigma) * self.med) / self.max
        print(self.med, self.upper_ratio, self.lower_ratio)

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
        self.mono_borders = deepcopy(self.acc)
        self.acc = threedim(self.acc)
        self.borders = deepcopy(self.acc)
    
    def hough(self):
        """Transformée de HOUGH classique pour la détection des lignes droites"""
        print("Calculating Hough space...")
        # Création de l'espace de HOUGH
        thetas = np.linspace(-np.pi/2, np.pi/2, self.THETA_RES)     # Liste des θ possibles
        self.rhomax = np.ceil(np.sqrt(self.shape[0] ** 2 + self.shape[1] ** 2))      # Longueur maximale d'une ligne : diagonale de l'image
        rhos = np.linspace(-self.rhomax, self.rhomax, self.RHO_RES)   # Liste des ρ possibles
        self.hspace = np.zeros((self.THETA_RES, self.RHO_RES))   # Espace de HOUGH
        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                if self.mono_borders[y][x] != 0:
                    coords = (y - self.shape[0]/2, x - self.shape[1]/2)       # Coordonnées cartésiennes du bord en cours de traitement
                    for th in range(len(thetas)):
                        rho = coords[1]*np.cos(thetas[th]) + coords[0]*np.sin(thetas[th])   # ρ = xcos(θ) + ysin(θ) (LEAVERS, 1992)
                        rh = np.argmin(np.abs(rhos - rho))      # Recherche de la valeur de ρ appropriée
                        self.hspace[rh][th] += 1
        
        # Recherche des maxima
        self.lines = []     # Liste des lignes trouvées (tuple de deux points)
        for y in range(self.RHO_RES):
            for x in range(self.THETA_RES):
                if self.hspace[y][x] >= self.INTERSEC_TH:
                    coords = (rhos[y], thetas[x])       # Coordonnées du maximum dans l'espace de HOUGH
                    a, b = np.cos(coords[1]), np.sin(coords[1])     # Pente et ordonnée à l'origine de la ligne équivalente (LEAVERS, 1992)
                    x0, y0 = a*coords[0] + self.shape[1]/2, b*coords[0] + self.shape[0]/2     # Coordonnées du point de la ligne le plus proche de l'origine
                    y1, x1 = y0-300, 300*np.sin(coords[1]) + x0        # Coordonnées d'un point quelconque appartenant à la ligne
                    f = lambda t : (y1-y0) / (x1-x0) * t + (y0 - (y1-y0)/(x1-x0)*x0)       # Fonction associée à la ligne
                    self.lines.append(((x0, y0), (x1, y1), f))
    
    def align(self):
        """DEBUG. Affichage de l'image originale, de l'image cannyfiée, de l'espace de HOUGH, et de l'image avec les lignes détectées"""
        fig = plt.figure(figsize = (12, 12))
        orig = fig.add_subplot(141)
        orig.axis('off')
        orig.imshow(self.rgb)
        orig.title.set_text("Image originale")
        cannyed = fig.add_subplot(142)
        cannyed.axis('off')
        cannyed.imshow(self.borders)
        cannyed.title.set_text("Filtre de CANNY")
        houghed = fig.add_subplot(143)
        houghed.axis('off')
        houghed.imshow(self.hspace)
        houghed.title.set_text("Espace de HOUGH")
        final = fig.add_subplot(144)
        final.axis('off')
        final.imshow(self.rgb)
        for line in self.lines:
            final.add_line(lines.Line2D((line[0][0], line[1][0]), (line[0][1], line[1][1])))
        final.title.set_text("Lignes détectées")
        plt.show()

def debug(im):
    plt.axis('off')
    plt.imshow(im.rgb)
    for line in im.lines:
        plt.axline(line[0], line[1], color = 'yellow')
    plt.show()

def render(im, axis):
    """Affichage d'une image avec Matplotlib"""
    if not axis:
        plt.axis('off')
    plt.imshow(im.hspace)#, extent = (0, np.pi, -im.rhomax, im.rhomax))
    plt.savefig("res.png", bbox_inches = 'tight')
    plt.show()


# Filtres gaussiens

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


def threedim(im):
    """Revient à un array à trois dimensions afin d'afficher les nuances de gris"""
    return(np.array([[[int(im[i][j]) for _ in range(3)] for j in range(len(im[0])-1)] for i in range(len(im)-1)]))

def mainloop():
    """Boucle principale du programme"""
    print("Processing...")

    ### Initialisation des images
    #cur_im = io.imread("no_kiha.png")
    #cur_im = io.imread("dummy2.png") # HD
    cur_im = Image("tree")
    cur_im.canny()
    cur_im.hough()
    debug(cur_im)


mainloop()
#ratios_influence("sta1_echigo", 3, 3, 0.1)