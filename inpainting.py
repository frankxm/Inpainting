#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import cv2
import numpy as np
import fillfront
import priorities
import bestpatch
import update
import time
import shutil
import os
import multiprocessing

def processpatch(im,taillecadre,masque,source,confiance,data,original,gradientX,gradientY):
    s_t = time.time()
    # 通过滤波获取需要先处理的边缘像素domega，同时保存像素点的梯度向量和梯度向量的法向量normale
    # domega保存坐标形式为[y,x]。 图像坐标中：左上角为原点，向右为x，向下为y
    dOmega, normale = fillfront.IdentifyTheFillFront(masque, source)
    print("函数1所用时间为{}秒".format(time.time() - s_t))

    s_t = time.time()
    # 计算所有边缘值的优先值，分别算Confiance和Data，最后找到最优的点先处理
    confiance, data, index = priorities.calculPriority(im, taillecadre, masque, dOmega, normale, data, gradientX,
                                                       gradientY, confiance)
    print("函数2所用时间为{}秒".format(time.time() - s_t))

    s_t = time.time()
    list, pp = bestpatch.calculPatch(dOmega, index, im, original, masque, taillecadre)
    print("函数3所用时间为{}秒".format(time.time() - s_t))

    s_t = time.time()
    im, gradientX, gradientY, confiance, source, masque = update.update(im, gradientX, gradientY, confiance, source,
                                                                        masque, dOmega, pp, list, index,
                                                                        taillecadre)
    print("函数4所用时间为{}秒".format(time.time() - s_t))
    return im
if __name__ == '__main__':
    inpainting_imgpath='./images'
    inpainting_maskpath=r'./mask'
    ind = 0
    num_img = len(os.listdir(inpainting_imgpath))
    while ind < num_img:
        programme_debute = time.time()
        img_courant = os.path.join(inpainting_imgpath, os.listdir(inpainting_imgpath)[ind])
        mask_courant = os.path.join(inpainting_maskpath, os.listdir(inpainting_maskpath)[ind])


        taillecadre = 10
        programme_debute = time.time()
        # 三通道 高,宽,通道
        image = cv2.imread(img_courant, 1)
        # 双通道
        masque = cv2.imread(mask_courant, 0)

        xsize, ysize, channels = image.shape  # meme taille pour filtre et image

        # on verifie les tailles

        x, y = masque.shape

        if x != xsize or y != ysize:
            print("La taille de l'image et du filtre doivent être les même")
            exit()

        tau = 175  # valeur pour séparer les valeurs du masque
        # 记录目标区域像素点
        omega = []
        confiance = np.copy(masque)
        confiance = confiance.astype(np.float32)
        # 此时masque遮挡处为0其余为255


        # 此时masque遮挡处为1，其余为0 confiance相反 图像上遮挡处为255

        if os.path.exists('./detectall'):
            shutil.rmtree('./detectall')
        os.makedirs('./detectall')

        mask_condition = masque < tau
        image[mask_condition] = [255, 255, 255]
        masque[mask_condition] = 1
        confiance[mask_condition] = 0.0
        masque[~mask_condition] = 0
        confiance[~mask_condition] = 1.0

        source = np.copy(confiance)
        original = np.copy(confiance)
        dOmega = []
        normale = []
        im = np.copy(image)
        result = np.ndarray(shape=image.shape)
        data = np.ndarray(shape=image.shape[:2])
        bool = True  # pour le while
        print("Algorithme en fonctionnement")
        k = 0


        while bool:
            start_time = time.time()
            k += 1
            print(k)
            xsize, ysize = source.shape
            niveau_de_gris = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            gradientX = np.float32(cv2.convertScaleAbs(cv2.Scharr(niveau_de_gris, cv2.CV_32F, 1, 0)))
            gradientY = np.float32(cv2.convertScaleAbs(cv2.Scharr(niveau_de_gris, cv2.CV_32F, 0, 1)))
            gradientX[masque == 1] = 0
            gradientY[masque == 1] = 0

            # # 将数值转换到0到1之间正常显示
            gradientX, gradientY = gradientX / 255, gradientY / 255
            im=processpatch(im, taillecadre, masque, source, confiance, data, original, gradientX, gradientY)

            bool = np.any(source == 0)

            cv2.imwrite(img_courant[:-4] + "_resultat.jpg", im)
            cv2.imwrite("detectall/res{}.jpg".format(k), im)
            print("迭代第{}轮所用时间为{}秒".format(k, time.time() - start_time))
        print("执行完{}轮所用时间为{}秒".format(k, time.time() - programme_debute))


        ind += 1






#
#
#
#
#
# import cv2
# import numpy as np
# import fillfront
# import priorities
# import bestpatch
# import update
# import time
# import shutil
# import os
# import multiprocessing
#
#
# def processpatch_worker(args):
#     im, taillecadre, masque, source, confiance, data, original, gradientX, gradientY, terminate_flag = args
#     return processpatch(im, taillecadre, masque, source, confiance, data, original, gradientX, gradientY,
#                         terminate_flag)
#
#
# def processpatch(im, taillecadre, masque, source, confiance, data, original, gradientX, gradientY, terminate_flag):
#     s_t = time.time()
#     # 通过滤波获取需要先处理的边缘像素domega，同时保存像素点的梯度向量和梯度向量的法向量normale
#     # domega保存坐标形式为[y,x]。 图像坐标中：左上角为原点，向右为x，向下为y
#     dOmega, normale = fillfront.IdentifyTheFillFront(masque, source)
#     print("函数1所用时间为{}秒".format(time.time() - s_t))
#
#     s_t = time.time()
#     # 计算所有边缘值的优先值，分别算Confiance和Data，最后找到最优的点先处理
#     confiance, data, index = priorities.calculPriority(im, taillecadre, masque, dOmega, normale, data, gradientX,
#                                                        gradientY, confiance)
#     print("函数2所用时间为{}秒".format(time.time() - s_t))
#
#     s_t = time.time()
#     list, pp = bestpatch.calculPatch(dOmega, index, im, original, masque, taillecadre)
#     print("函数3所用时间为{}秒".format(time.time() - s_t))
#
#     s_t = time.time()
#     im, gradientX, gradientY, confiance, source, masque = update.update(im, gradientX, gradientY, confiance, source,
#                                                                         masque, dOmega, pp, list, index,
#                                                                         taillecadre)
#     print("函数4所用时间为{}秒".format(time.time() - s_t))
#
#     # Check the termination condition
#     terminate_flag.value = not np.any(source == 0)
#     return source
#
#
#
#
# if __name__ == '__main__':
#     cheminimage = "original.jpg"
#     cheminmasque = "mask.jpg"
#     taillecadre = 15
#     programme_debute = time.time()
#     # 三通道 高,宽,通道
#     image = cv2.imread(cheminimage, 1)
#     # 双通道
#     masque = cv2.imread(cheminmasque, 0)
#     xsize, ysize, channels = image.shape  # meme taille pour filtre et image
#
#     # on verifie les tailles
#     x, y = masque.shape
#     if x != xsize or y != ysize:
#         print("La taille de l'image et du filtre doivent être les même")
#         exit()
#
#     tau = 175  # valeur pour séparer les valeurs du masque
#     # 记录目标区域像素点
#     omega = []
#     confiance = np.copy(masque)
#     confiance = confiance.astype(np.float32)
#
#     if os.path.exists('./detectall'):
#         shutil.rmtree('./detectall')
#     os.makedirs('./detectall')
#
#     mask_condition = masque < tau
#     image[mask_condition] = [255, 255, 255]
#     masque[mask_condition] = 1
#     confiance[mask_condition] = 0.0
#     masque[~mask_condition] = 0
#     confiance[~mask_condition] = 1.0
#
#     source = np.copy(confiance)
#     original = np.copy(confiance)
#     dOmega = []
#     normale = []
#
#     im = np.copy(image)
#     result = np.ndarray(shape=image.shape)
#
#     data = np.ndarray(shape=image.shape[:2])
#
#     print("Algorithme en fonctionnement")
#     niveau_de_gris = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
#     gradientX = np.float32(cv2.convertScaleAbs(cv2.Scharr(niveau_de_gris, cv2.CV_32F, 1, 0)))
#     gradientY = np.float32(cv2.convertScaleAbs(cv2.Scharr(niveau_de_gris, cv2.CV_32F, 0, 1)))
#
#     gradientX[masque == 1] = 0
#     gradientY[masque == 1] = 0
#
#     gradientX, gradientY = gradientX / 255, gradientY / 255
#     # Set the number of worker processes
#     num_processes = 2
#     # Create a multiprocessing.Manager to manage shared objects
#     manager = multiprocessing.Manager()
#     # Create a shared Value for termination condition
#     terminate_flag = manager.Value('b', False)
#     k=0
#     while not terminate_flag.value:
#         k+=1
#         print(k)
#         # Split the image into chunks for parallel processing
#         chunks = [(im_chunk, taillecadre, masque_chunk, source_chunk, confiance_chunk, data_chunk, original_chunk,
#                    gradientX_chunk, gradientY_chunk, terminate_flag) for im_chunk,masque_chunk, source_chunk, confiance_chunk,
#                                                                          data_chunk, original_chunk, gradientX_chunk,
#                                                                          gradientY_chunk in
#                   zip(np.array_split(im, num_processes, axis=0),
#                       np.array_split(masque, num_processes, axis=0),
#                       np.array_split(source, num_processes, axis=0),
#                       np.array_split(confiance, num_processes, axis=0),
#                       np.array_split(data, num_processes, axis=0),
#                       np.array_split(original, num_processes, axis=0),
#                       np.array_split(gradientX, num_processes, axis=0),
#                       np.array_split(gradientY, num_processes, axis=0))]
#
#         pool = multiprocessing.Pool(processes=num_processes)
#         results = pool.map(processpatch_worker, chunks)
#         pool.close()
#         pool.join()
#
#         # Combine the results back into the original arrays
#         im = np.vstack(results)
#         cv2.imwrite(cheminimage[:-4] + "_resultat.jpg", im)
#         cv2.imwrite("detectall/res{}.jpg".format(k), im)
#
#     print("Termination condition met. Exiting...")