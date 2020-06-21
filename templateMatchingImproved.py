import cv2 as cv
import os
import numpy as np
import imutils

# Data Sets
DATASET_PATH = "data/"
# DATASET_PATH = "data/TinyTLP/"
DATA_SETS = ["CarChase1", "CarChase2",  # 0,1
             'ISS', 'Boat', 'KinBall3',  # 2,3,4 [Occlusion: Kinball3]
             'DriftCar1', 'Drone1', 'Boxing1', 'Bike',  # 5,6,7,8 [Occlusion: Boxing1]
             'MotorcycleChase', 'Elephants']  # 9, 10 [Occlusion: Elephants]
DATASET_NUMBER = 0

# Constants
DATASET_NAME = DATA_SETS[DATASET_NUMBER]
DATASET_FOLDER = DATASET_PATH + DATASET_NAME + "/img"
GROUND_TRUTH_PATH = DATASET_PATH + DATASET_NAME + "/groundtruth_rect.txt"
OUTPUT_PATH = "output/templateMatching/" + DATASET_NAME + "/"

# different methods of template matching, however I got good results using 1st one
METHODS_TEMP = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',  # 0, 1, 2
                'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']  # 3, 4, 5
METHOD_NUMBER = 0
METHOD = eval(METHODS_TEMP[METHOD_NUMBER])

# Save Images or Not
SAVE_IMAGES = False


def markObject(stPoint, enPoint, image, count):
    markedImage = cv.rectangle(image, stPoint, enPoint, (255, 0, 0), 2)
    cv.imshow('Marked Image ' + str(count), markedImage)
    cv.waitKey(0)


def cropObject(stPoint, enPoint, image):
    # clone = image.copy()
    crop_img = image[stPoint[1]:enPoint[1], stPoint[0]:enPoint[0]]
    return crop_img


if __name__ == "__main__":
    file_name = "00001.jpg"
    file_path = os.path.join(DATASET_FOLDER, file_name)
    # print(file_path)
    image = cv.imread(file_path, 0)
    heightImage, widthImage = image.shape

    f = open(GROUND_TRUTH_PATH)
    line = f.readline()
    lineList = line.split(",")
    tempXSt = int(lineList[1])
    tempXEn = int(lineList[1]) + int(lineList[3])
    tempYSt = int(lineList[2])
    tempYEn = int(lineList[2]) + int(lineList[4])

    markObject((tempXSt, tempYSt), (tempXEn, tempYEn), image, 0)
    template = cropObject((tempXSt, tempYSt), (tempXEn, tempYEn), image)
    h, w = template.shape

    # defining an area based on intitial location of object to be searched
    stXSearch = int(lineList[1]) - int(lineList[3])
    stXSearch = max(0, stXSearch)
    enXSearch = int(lineList[1]) + 2 * int(lineList[3])
    enXSearch = min(widthImage, enXSearch)
    stYSearch = int(lineList[2]) - int(lineList[4])
    stYSearch = max(0, stYSearch)
    enYSearch = int(lineList[2]) + 2 * int(lineList[4])
    enYSearch = min(heightImage, enYSearch)
    # print(stXSearch, enXSearch, stYSearch, enYSearch)

    i = 0
    while i < 600:
        i = i + 1
        file_no = str(i)
        zeros = 5 - len(file_no)
        file_name = file_no.rjust(zeros + len(file_no), '0')
        file_name = file_name + ".jpg"
        file_path = os.path.join(DATASET_FOLDER, file_name)
        # print(file_path)
        colored_image = cv.imread(file_path)
        image = cv.imread(file_path, 0)
        found = None

        # handling the case of different sized objects by resizing the image to match with template
        for scale in np.linspace(0.8, 1.2, 10):
            resized = imutils.resize(image, width=int(image.shape[1] * scale))
            r = image.shape[1] / float(resized.shape[1])

            imagePart = resized[int(stYSearch / r):int(enYSearch / r), int(stXSearch / r):int(enXSearch / r)]
            # cv.imshow("image part"+str(i), imagePart)
            # cv.waitKey(0)
            if imagePart.shape[0] >= h and imagePart.shape[1] >= w:
                result = cv.matchTemplate(imagePart, template, METHOD)
                (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(result)
                # if found is None or maxVal > found[0]:
                if METHOD in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
                    if found is None or minVal < found[0]:
                        found = (minVal, minLoc, r)
                else:
                    if found is None or maxVal > found[0]:
                        found = (maxVal, maxLoc, r)

        (val, loc, r) = found
        (startX, startY) = (int(loc[0] * r) + stXSearch, int(loc[1] * r) + stYSearch)
        (endX, endY) = (int((loc[0] + w) * r) + stXSearch, int((loc[1] + h) * r) + stYSearch)
        markedImage = cv.rectangle(colored_image, (startX, startY), (endX, endY), (255, 0, 0), 2)
        cv.imshow('Track Window', markedImage)
        cv.waitKey(50)

        # updating the search area
        stXSearch = int(2 * startX - endX)
        stXSearch = max(0, stXSearch)
        enXSearch = int(2 * endX - startX)
        enXSearch = min(widthImage, enXSearch)
        stYSearch = int(2 * startY - endY)
        stYSearch = max(0, stYSearch)
        enYSearch = int(2 * endY - startY)
        enYSearch = min(heightImage, enYSearch)

        # template = cropObject((startX, startY), (endX, endY), image)
        # h, w = template.shape
        # cv.imshow('New Template' + str(i), template)
        # cv.waitKey(0)
