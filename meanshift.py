import cv2
import os
import numpy as np

DATASET_PATH = "data/"
DATA_SETS = ["CarChase1", "CarChase2"]
DATASET_NUMBER = 1
DATASET_NAME = DATA_SETS[DATASET_NUMBER]
DATASET_FOLDER = DATASET_PATH + DATASET_NAME + "/img"
GROUND_TRUTH_PATH = DATASET_PATH + DATASET_NAME + "/groundtruth_rect.txt"
OUTPUT_PATH = "output/meanshift/" + DATASET_NAME + "/"
SAVE_IMAGES = True

if __name__ == '__main__':

    bounding_box = []
    with open(GROUND_TRUTH_PATH, 'r') as f:
        for line in f:
            line = line.split(',')
            line = [int(x) for x in line]
            box = tuple(line[1:5])
            bounding_box.append(box)

    images = os.listdir(DATASET_FOLDER)
    images.sort()
    frames = []
    total_frames = 0

    for img_name in images:
        path = os.path.join(DATASET_FOLDER, img_name)
        frame = cv2.imread(path)
        frames.append(frame)
        total_frames += 1

    init_bbox = bounding_box[0]
    first_frame = frames[0]

    # setup initial location of window
    x, y, w, h = init_bbox[0], init_bbox[1], init_bbox[2], init_bbox[3]
    track_window = (x, y, w, h)

    # set up the ROI for tracking
    # roi = first_frame[y:y+h, x:x+w]
    roi = first_frame[int(y + h/4):int(y+h*3/4), int(x+w/4):int(x+w*3/4)]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = None
    hist_channels = [0, 1]
    hist_range = [0, 255, 60, 255]
    roi_hist = cv2.calcHist([hsv_roi], hist_channels, mask, [12, 12], hist_range)
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 1)

    for i in range(total_frames):
        frame = frames[i]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], hist_channels, roi_hist, hist_range, 1)
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        # Draw it on image
        x, y, w, h = track_window
        output_img = cv2.rectangle(frame, (x, y), (x+w, y+h), 255, 2)
        cv2.imshow('img2', output_img)
        k = cv2.waitKey(30)
        if SAVE_IMAGES:
            output_file = OUTPUT_PATH + "{0:0=5d}".format(i) + '.jpg'
            print(output_file)
            cv2.imwrite(output_file, output_img)


