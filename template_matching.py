import cv2 as cv
import numpy as np
import imutils

METHODS_TEMP = [cv.TM_CCOEFF, cv.TM_CCOEFF_NORMED, cv.TM_CCORR,
                cv.TM_CCORR_NORMED, cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]


def markObject(stPoint, enPoint, image, count):
    stPoint = tuple((int(x) for x in stPoint))
    enPoint = tuple((int(x) for x in enPoint))
    markedImage = cv.rectangle(image, stPoint, enPoint, (255, 0, 0), 2)


def cropObject(stPoint, enPoint, image):
    crop_img = image[stPoint[1]:enPoint[1], stPoint[0]:enPoint[0]]
    return crop_img


def find_object(image, template, method):

    h, w, _ = template.shape
    found = None
    if method not in METHODS_TEMP:
        print("Error: Invalid Template Matching Method.")
        exit(0)

    # handling the case of different sized objects by resizing the image to match with template
    for scale in np.linspace(0.2, 1.8, 40):
        resized = imutils.resize(image, width=int(image.shape[1] * scale))
        r = image.shape[1] / float(resized.shape[1])
        if resized.shape[0] < h or resized.shape[1] < w:
            continue
        result = cv.matchTemplate(resized, template, method)
        (_, maxVal, _, maxLoc) = cv.minMaxLoc(result)
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)

    if found is None:
        print("No Object Found")
        exit(0)
    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + w) * r), int((maxLoc[1] + h) * r))
    return [(startX, startY), (endX, endY)]
