import cv2 as cv
import numpy as np
import os
import imutils

# name of the folder containing the frames
DATASET_PATH = "data/"
DATA_SETS = ["CarChase1", "CarChase2", 'KinBall3',  # 0,1,2 [Occlusion: Kinball3]
             'DriftCar1', 'Boxing1', 'Elephants']  # 3,4,5 [Occlusion: Boxing1, Elephants]
DATASET_NUMBER = 3

DATASET_NAME = DATA_SETS[DATASET_NUMBER]
DATASET_FOLDER = DATASET_PATH + DATASET_NAME + "/img"
GROUND_TRUTH_PATH = DATASET_PATH + DATASET_NAME + "/groundtruth_rect.txt"
OUTPUT_PATH = "output/kalman/" + DATASET_NAME + "/"
SAVE_IMAGES = True


def cropObject(stPoint, enPoint, image):
    crop_img = image[stPoint[1]:enPoint[1], stPoint[0]:enPoint[0]]
    # cv.imshow("crop_img", crop_img)
    # cv.waitKey(0)
    return crop_img


def markObject(stPoint, enPoint, image, count, imageName):
    # clone_image = image.copy()
    line_color = None
    # different colors for predicted, detected and corrected measures
    if imageName == "Predicted":
        line_color = (255, 0, 0)
    elif imageName == "Detected":
        line_color = (0, 255, 0)
    else:
        line_color = (0, 0, 255)
    markedImage = cv.rectangle(image, stPoint, enPoint, line_color, 2)
    cv.imshow(imageName, markedImage)
    cv.waitKey(5)


if __name__ == "__main__":

    # setup output folder
    if SAVE_IMAGES:
        if not os.path.exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)

    kf = cv.KalmanFilter(6, 4, 0)
    dt = float(20 / 600)
    state = None
    meas = None
    kf.transitionMatrix = np.array([[1, 0, dt, 0, 0, 0],
                                    [0, 1, 0, dt, 0, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1]], np.float32)
    # print(type(kf.transitionMatrix))
    kf.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 0, 1]], np.float32)

    kf.processNoiseCov = np.array([[1e-2, 0, 0, 0, 0, 0],
                                   [0, 1e-2, 0, 0, 0, 0],
                                   [0, 0, 5, 0, 0, 0],
                                   [0, 0, 0, 5, 0, 0],
                                   [0, 0, 0, 0, 1e-2, 0],
                                   [0, 0, 0, 0, 0, 1e-2]], np.float32)
    # print(kf.processNoiseCov)
    kf.measurementNoiseCov = np.array([[1e-1, 0, 0, 0],
                                       [0, 1e-1, 0, 0],
                                       [0, 0, 1e-1, 0],
                                       [0, 0, 0, 1e-1]], np.float32)

    # extraction of template from 1st image
    file_name = "00001.jpg"
    file_path = os.path.join(DATASET_FOLDER, file_name)
    image = cv.imread(file_path, 0)
    # cv.imshow('Image', image)
    # cv.waitKey()
    f = open(GROUND_TRUTH_PATH)
    line = f.readline()
    lineList = line.split(",")
    # print(lineList)
    tempXSt = int(lineList[1])
    tempXEn = int(lineList[1]) + int(lineList[3])
    tempYSt = int(lineList[2])
    tempYEn = int(lineList[2]) + int(lineList[4])
    # print(tempXEn)
    # markObject((tempXSt, tempYSt), (tempXEn, tempYEn), image, 0, "template")
    template = cropObject((tempXSt, tempYSt), (tempXEn, tempYEn), image)
    h, w = template.shape

    i = 0
    first_detect = True
    files = os.listdir(DATASET_FOLDER)

    while i < 600:
        print("frame" + str(i))
        i = i + 1
        file_no = str(i)
        zeros = 5 - len(file_no)
        file_name = file_no.rjust(zeros + len(file_no), '0')
        file_name = file_name + ".jpg"
        # print(file_name)
        file_path = os.path.join(DATASET_FOLDER, file_name)
        colored_image = cv.imread(file_path)
        image = cv.imread(file_path, 0)
        # cv.imshow('check', image)
        # cv.waitKey(0)

        # -------------Prediction--------------------
        if not first_detect:
            state = kf.predict()
            print("predicted state ", state)
            widthPred = state[4, 0]
            heightPred = state[5, 0]
            startXPred = state[0, 0] - float(widthPred / 2.0)
            endXPred = state[0, 0] + float(widthPred / 2.0)
            startYPred = state[1, 0] - float(heightPred / 2.0)
            endYPred = state[1, 0] + float(heightPred / 2.0)
            # markObject((int(startXPred), int(startYPred)),
            #                (int(endXPred), int(endYPred)), colored_image, i, "Predicted")

        # -------------Template Matching (Measurement)--------------------
        # template matching by changing the frame ratio
        found = None
        for scale in np.linspace(0.4, 1.6, 20):
            resized = imutils.resize(image, width=int(image.shape[1] * scale))
            r = image.shape[1] / float(resized.shape[1])
            if resized.shape[0] < h or resized.shape[1] < w:
                break
            result = cv.matchTemplate(resized, template, cv.TM_CCOEFF)
            (_, maxVal, _, maxLoc) = cv.minMaxLoc(result)
            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)
        (_, maxLoc, r) = found
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + w) * r), int((maxLoc[1] + h) * r))
        # markObject((startX, startY), (endX, endY), colored_image, i, "Detected")

        centerX = float((float(startX) + float(endX)) / 2.0)
        centerY = float((float(startY) + float(endY)) / 2.0)
        width = float(float(endX) - float(startX))
        height = float(float(endY) - float(startY))
        meas = np.array([[centerX],
                         [centerY],
                         [width],
                         [height]], np.float32)
        # print("measurement ", meas)

        # -------------Correction--------------------
        if first_detect:
            kf.errorCovPre = np.array([[1, 0, 0, 0, 0, 0],
                                       [0, 1, 0, 0, 0, 0],
                                       [0, 0, 1, 0, 0, 0],
                                       [0, 0, 0, 1, 0, 0],
                                       [0, 0, 0, 0, 1, 0],
                                       [0, 0, 0, 0, 0, 1]], np.float32)
            state = np.array([[meas[0, 0]],
                              [meas[1, 0]],
                              [0],
                              [0],
                              [meas[2, 0]],
                              [meas[3, 0]]], np.float32)
            # print(state)
            kf.statePost = state
            first_detect = False
        else:
            kf.correct(meas)
            # print("Corrected State ", kf.statePost)
            widthCorr = kf.statePost[4, 0]
            heightCorr = kf.statePost[5, 0]
            startXCorr = kf.statePost[0, 0] - float(widthCorr / 2.0)
            endXCorr = kf.statePost[0, 0] + float(widthCorr / 2.0)
            startYCorr = kf.statePost[1, 0] - float(heightCorr / 2.0)
            endYCorr = kf.statePost[1, 0] + float(heightCorr / 2.0)
            stPoint = (int(startXCorr), int(startYCorr))
            enPoint = (int(endXCorr), int(endYCorr))
            markObject(stPoint, enPoint, colored_image, i, "Corrected")

            if SAVE_IMAGES:
                line_color = (0, 0, 255)
                markedImage = cv.rectangle(colored_image, stPoint, enPoint, line_color, 2)
                output_file = OUTPUT_PATH + "{0:0=5d}".format(i) + '.jpg'
                print(output_file)
                cv.imwrite(output_file, markedImage)
