import cv2 as cv
import os
import numpy as np
import imutils
# Abhi ISS, Jet3, Boat
# Ye 3 thore simple hai, similar to CarChase1
# Kinball 3: Random Motion plus Occulsion
# ZebraFish: Partial Occultion and Random Motion
# Billiard2: Sudden Motion

# the images are present in this folder
DATASET_NAME = "CarChase1"
PREFIX = "data/" + DATASET_NAME + "/"
DATASET_FOLDER = PREFIX + "/img"

# different methods of template matching, however I got good results using 1st one
METHODS_TEMP = [cv.TM_CCOEFF, cv.TM_CCOEFF_NORMED, cv.TM_CCORR,
                cv.TM_CCORR_NORMED, cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]


def markObject(stPoint, enPoint, image, count):
    stPoint = tuple((int(x) for x in stPoint))
    enPoint = tuple((int(x) for x in enPoint))
    markedImage = cv.rectangle(image, stPoint, enPoint, (255, 0, 0), 2)
    # cv.imshow('Marked Image ' + str(count), markedImage)
    # cv.waitKey(10)


def cropObject(stPoint, enPoint, image):
    # clone = image.copy()
    crop_img = image[stPoint[1]:enPoint[1], stPoint[0]:enPoint[0]]
    # cv.imshow("crop_img", crop_img)
    # cv.waitKey(0)
    return crop_img


# def templateMatching(image, template):
#     method = eval(METHODS_TEMP[0])
#     h, w = template.shape
#     print("height" + h + " " + "width" + w)
#     res = cv.matchTemplate(image, template, method)
#     min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
#     if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
#         top_left = min_loc
#     else:
#         top_left = max_loc
#     bottom_right = (top_left[0] + w, top_left[1] + h)
#     return top_left, bottom_right

def find_object(image, template, method):
    # handling the case of different sized objects by resizing the image to match with template
    h, w, _ = template.shape
    found = None
    if method not in METHODS_TEMP:
        print("Error: Invalid Template Matching Method.")
        exit(0)

    for scale in np.linspace(0.2, 1.8, 40):
        resized = imutils.resize(image, width=int(image.shape[1] * scale))
        r = image.shape[1] / float(resized.shape[1])
        if resized.shape[0] < h or resized.shape[1] < w:
            break
        # resized = cv.Canny(resized, 50, 200)
        # det_coord = templateMatching(resized, template)
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


def hsv_hist_for_win(img, wind, th=0.2):
    c, r, w, h = wind
    # th = 0.2
    # print(wind)
    # print([(h*th),(h*(1-th)), (w*th),(w*(1-th))])
    roi = img[r+round(h*th):r+round(h*(1-th)), c+round(w*th):c+round(w*(1-th))]
    # cv.imshow('roi', roi)
    # cv.waitKey(5000)
    plotHist(roi)
    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    # mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv.calcHist([hsv_roi], [0], None, [180], [0, 180])
    cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
    print(roi_hist)
    # cv.imshow('roi_hist', roi_hist)
    # cv.waitKey(0)
    # exit(0)
    return roi_hist


def plotHist(image):
    from matplotlib import pyplot as plt
    chans = cv.split(image)
    colors = ("b", "g", "r")
    plt.figure()
    plt.title("'Flattened' Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    features = []
    # loop over the image channels
    for (chan, color) in zip(chans, colors):
        # create a histogram for the current channel and
        # concatenate the resulting histograms for each
        # channel
        hist = cv.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)
        # plot the histogram
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()


def evalPart(backproj, particles):
    p = [int(x) for x in particles[0]]
    q = [int(x) for x in particles[1]]
    return backproj[q, p]


if __name__ == "__main__":
    file_name = "00001.jpg"
    file_path = os.path.join(DATASET_FOLDER, file_name)
    # print(file_path)
    img = cv.imread(file_path, 0)
    # cv.imshow('Image', image)
    # cv.waitKey()

    # I have manually put the coordinates of the object to be detected from the file along with images
    # It can be read from the file programmatically
    markObject((413, 346), (585, 453), img, 0)
    tmpl = cropObject((413, 346), (585, 453), img)

    # print(template.shape)
    # template = cv.Canny(template, 50, 200)
    # cv.imshow('Template', template)
    # cv.waitKey()
    files = os.listdir(DATASET_FOLDER)
    i = 0
    for file_name in files:
        # i = i + 1
        # if i == 50:
        # 	break
        file_path = os.path.join(DATASET_FOLDER, file_name)
        # print(file_path)
        img = cv.imread(file_path, 0)
        start, end = find_object(img, tmpl)
        markObject(start, end, img, i)
