import sys
import cv2
import time
import numpy as np
from numba import jit
import os
import shutil
def openImages(cheminimage, cheminmasque,  tau):
    image = cv2.imread(cheminimage, 1)
    masque = cv2.imread(cheminmasque, 0)

    xsize, ysize, _ = image.shape
    x, y = masque.shape

    if x != xsize or y != ysize:
        print("La taille de l'image et du filtre doivent être les mêmes")
        exit()

    confiance = masque.copy()

    # Vectorisation pour la manipulation du masque
    masque[masque < tau] = 1
    masque[masque >= tau] = 0
    confiance[masque == 1] = 0.
    confiance[masque == 0] = 1.

    return image, masque, confiance, xsize, ysize


@jit(nopython=True)
def createMask(image, masque, taillecadre, xsize, ysize, confiance):
    sourcePatch = []
    # Création de l'image avec le masque
    for x in range(xsize):
        for y in range(ysize):
            v = masque[x, y]
            if v == 1:
                image[x, y] = [255, 255, 255]
            if ((x <= (xsize - taillecadre - 1)) and (y <= (ysize - taillecadre - 1))):
                if patch_complet(x, y, taillecadre + 1, taillecadre + 1, confiance):
                    sourcePatch.append((x, y))
    bool_val = True
    source = confiance.copy()
    d = 0  # variable pour compter le nombre d'images
    minx = miny = 0

    while bool_val:
        d += 1
        print(d)
        dOmega, minx, miny = fillfront(masque, minx, miny)
        # print('enter1')
        pointPatch = (0, 0)
        mini = minvar = sys.maxsize
        patch = Patch(dOmega, 5)
        x1, y1 = patch[0]
        x2, y2 = patch[1]

        compteur, cibles, ciblem = crible(y2 - y1 + 1, x2 - x1 + 1, x1, y1, masque)
        # print('enter2')

        for (y, x) in sourcePatch:
            R = V = B = ssd = 0
            for (i, j) in cibles:

                ima = image[y + i, x + j]
                omega = image[y1 + i, x1 + j]
                for k in range(3):
                    difference = float(ima[k]) - float(omega[k])
                    ssd += difference ** 2
                R += ima[0]
                V += ima[1]
                B += ima[2]
            ssd /= compteur
            if ssd < mini:
                mini = ssd
                pointPatch = (x, y)

                # variation = 0
                # for (i, j) in ciblem:
                #     ima = image[y + i, x + j]
                #     differenceR = ima[0] - R / compteur
                #     differenceV = ima[1] - V / compteur
                #     differenceB = ima[2] - B / compteur
                #     variation += differenceR ** 2 + differenceV ** 2 + differenceB ** 2
                # if ssd < mini or variation < minvar:
                #     minvar = variation
                #     mini = ssd
                #     pointPatch = (x, y)

        image, masque, confiance = update(dOmega, image, source, pointPatch, ciblem, masque)

        # Vérification de la condition de fin
        bool_val = np.any(confiance == 0)
        # cv2.imwrite("detectall/res{}.jpg".format(k), image)
    return image


@jit(nopython=True)
def fillfront(masque, minx, miny):
    dOmega = (0, 0)

    found = False
    for x in range(minx, xsize):
        if found:
            break
        for y in range(miny, ysize):
            miny = 0
            if (masque[x, y] == 1):
                dOmega = (y - 1, x - 1)
                found = True
                minx = x
                miny = y
                break
    return dOmega, minx, miny


@jit(nopython=True)
def crible(Xsize, Ysize, x1, y1, masque):
    compteur = 0
    cibles = []
    ciblem = []
    for i in range(Xsize):
        for j in range(Ysize):
            if masque[y1 + i, x1 + j] == 0:
                compteur += 1
                cibles.append((i, j))
            else:
                ciblem.append((i, j))
    return compteur, cibles, ciblem


@jit(nopython=True)
def patch_complet(x, y, Xsize, Ysize, original):
    for i in range(Xsize):
        for j in range(Ysize):
            if original[x + i, y + j] == 0:
                return False
    return True


@jit(nopython=True)
def update(dOmega, image, confiance, point, list, masque):
    global minx, miny
    p = dOmega
    px, py = p
    patch = Patch(p, 5)
    x1, y1 = patch[0]
    px, py = point
    for (i, j) in list:
        image[(y1 + i, x1 + j)] = image[(py + i, px + j)]
        confiance[y1 + i, x1 + j] = 1
        masque[y1 + i, x1 + j] = 0
    return (image, masque, confiance)




@jit(nopython=True)
def Patch(point, taillecadre):
    px, py = point
    x4 = min(px + taillecadre, ysize - 1)
    y4 = min(py + taillecadre, xsize - 1)
    return ((px, py), (x4, y4))

imgpath='./images'
maskpath='./mask'
ind=0
num_img=len(os.listdir(imgpath))
if os.path.exists('./result'):
    shutil.rmtree('./result')
os.makedirs('./result')
while ind<num_img:

    programme_debute = time.time()
    img_courant=os.path.join(imgpath,os.listdir(imgpath)[ind])
    mask_courant =os.path.join(maskpath,os.listdir(maskpath)[ind])
    image, masque, confiance, xsize, ysize = openImages(img_courant, mask_courant,170)
    image = createMask(image, masque, 5, xsize, ysize, confiance)
    filename=os.listdir(imgpath)[ind].split('.')[0]
    cv2.imwrite("result/{}.jpg".format(filename.split('.')[0]), image)
    print("Exécution des itérations de l'image {} en {} secondes".format(img_courant,time.time() - programme_debute))
    ind+=1





