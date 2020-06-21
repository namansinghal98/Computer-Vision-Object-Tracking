import cv2
import os
import numpy as np

# Data Sets
DATASET_PATH = "data/"
# DATASET_PATH = "data/TinyTLP/"
DATA_SETS = ["CarChase1", "CarChase2",  # 0,1
             'ISS', 'Boat', 'KinBall3',  # 2,3,4 [Occlusion: Kinball3]
             'DriftCar1', 'Drone1', 'Boxing1', 'Bike',  # 5,6,7,8 [Occlusion: Boxing1]
             'MotorcycleChase', 'Elephants']  # 9, 10 [Occlusion: Elephants]
DATASET_NUMBER = 7

# Constants
DATASET_NAME = DATA_SETS[DATASET_NUMBER]
DATASET_FOLDER = DATASET_PATH + DATASET_NAME + "/img"
GROUND_TRUTH_PATH = DATASET_PATH + DATASET_NAME + "/groundtruth_rect.txt"
OUTPUT_PATH = "output/meanshift/" + DATASET_NAME + "/"

# Save Images or Not
SAVE_IMAGES = False

# Debugging mode for extra outputs
DEBUG = False

# Dataset Parameter
PARAMS = [1, 1,
          2, 2, 2,
          3, 3, 3, 3,
          3, 1]
PARAM_NUMBER = PARAMS[DATASET_NUMBER]

if __name__ == '__main__':

    # Define Kalman Filter
    kf = cv2.KalmanFilter(6, 4, 0)
    dt = float(20 / 600)
    state = None
    meas = None
    first_detect = True
    not_found = 0

    kf.transitionMatrix = np.array([[1, 0, dt, 0, 0, 0],
                                    [0, 1, 0, dt, 0, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1]], np.float32)

    kf.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 0, 1]], np.float32)

    kf.processNoiseCov = np.array([[1e-1, 0, 0, 0, 0, 0],
                                   [0, 1e-1, 0, 0, 0, 0],
                                   [0, 0, 1e-1, 0, 0, 0],
                                   [0, 0, 0, 1e-1, 0, 0],
                                   [0, 0, 0, 0, 1e-1, 0],
                                   [0, 0, 0, 0, 0, 1e-1]], np.float32)

    kf.measurementNoiseCov = np.array([[1e-5, 0, 0, 0],
                                       [0, 1e-5, 0, 0],
                                       [0, 0, 1e-5, 0],
                                       [0, 0, 0, 1e-5]], np.float32)

    if DATASET_NUMBER == 10:
        kf.measurementNoiseCov = np.array([[1, 0, 0, 0],
                                           [0, 1, 0, 0],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]], np.float32)

    if DATASET_NUMBER == 4:
        kf.processNoiseCov = np.array([[1e-1, 0, 0, 0, 0, 0],
                                       [0, 1e-1, 0, 0, 0, 0],
                                       [0, 0, 10, 0, 0, 0],
                                       [0, 0, 0, 10, 0, 0],
                                       [0, 0, 0, 0, 1e-1, 0],
                                       [0, 0, 0, 0, 0, 1e-1]], np.float32)
    # Get the ground truth information
    bounding_box = []
    with open(GROUND_TRUTH_PATH, 'r') as f:
        for line in f:
            line = line.split(',')
            line = [int(x) for x in line]
            box = tuple(line[1:5])
            bounding_box.append(box)

    # Read the Images
    images = os.listdir(DATASET_FOLDER)
    images.sort()
    frames = []
    total_frames = 0

    for img_name in images:
        path = os.path.join(DATASET_FOLDER, img_name)
        frame = cv2.imread(path)
        frames.append(frame)
        total_frames += 1

    # setup initial frame and track window
    init_bbox = bounding_box[0]
    first_frame = frames[0]
    x, y, w, h = init_bbox[0], init_bbox[1], init_bbox[2], init_bbox[3]
    track_window = (x, y, w, h)

    # setup output folder
    if SAVE_IMAGES:
        if not os.path.exists(OUTPUT_PATH):
            os.mkdir(OUTPUT_PATH)

    # 1) Set parameters according to dataset
    if PARAM_NUMBER == 1:
        red_factor = 0.7
        mask = None
        hist_range = [0, 255, 60, 255]
        hist_size = [12, 12]
    elif PARAM_NUMBER == 2:
        red_factor = 0.5
        mask = None
        hist_range = [0, 255, 0, 255]
        hist_size = [5, 5]
    elif PARAM_NUMBER == 3:
        red_factor = 0.5
        mask = None
        hist_range = [0, 255, 60, 255]
        hist_size = [5, 5]
    else:
        red_factor = 1
        mask = None
        # mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        hist_range = [0, 255, 60, 255]
        hist_size = [12, 12]

    # 2) Set up the ROI (Region of Interest) for tracking
    roi = first_frame[int(y + h * red_factor / 2):int(y + h * (2 - red_factor) / 2),
          int(x + w * red_factor / 2):int(x + w * (2 - red_factor) / 2)]

    if DEBUG:
        cv2.imshow('box1', roi)
        print(np.shape(first_frame))

    # 3) Convert ROI to HSV color space
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 4) Using the Histogram parameters, calc the histogram of ROI
    hist_channels = [0, 1]
    roi_hist = cv2.calcHist([hsv_roi], hist_channels, mask, hist_size, hist_range)

    # 5) Normalize the histogram
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # 6) Setup the termination criteria, either 10 iteration or move by at least 1 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 1)

    # 7) Perform for each frame
    for i in range(total_frames):
        frame = frames[i]

        # -------------Prediction--------------------
        if not first_detect:
            state = kf.predict()
            print("predicted state "+str(i), state)
            widthPred = state[4, 0]
            heightPred = state[5, 0]
            startXPred = state[0, 0] - float(widthPred / 2.0)
            endXPred = state[0, 0] + float(widthPred / 2.0)
            startYPred = state[1, 0] - float(heightPred / 2.0)
            endYPred = state[1, 0] + float(heightPred / 2.0)

            # rectangle around predicted result (Green)
            # print("predicted ", startXPred, startYPred, widthPred, heightPred)
            output_img = cv2.rectangle(frame, (int(startXPred), int(startYPred)), (int(endXPred), int(endYPred)),
                                       (0, 255, 0), 2)

        # 8) Convert frame to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 9) Calculate the BackProject of frame with the histogram
        dst = cv2.calcBackProject([hsv], hist_channels, roi_hist, hist_range, 1)

        # 10) Perform meanshift tracking
        # print(track_window, i + 1, " tracking")
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # 11) Draw the resultant box on image (Blue for measurement)
        x, y, w, h = track_window
        # if first_detect:
        output_img = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)

        if DEBUG:
            cv2.imshow("backprop", dst)

        # 12) Display the output
        # cv2.imshow('tracking_output', output_img)
        # k = cv2.waitKey(30)

        # -----------------------Update the measurement--------------
        centerX = float((float(x) + float(x + w)) / 2.0)
        centerY = float((float(y) + float(y + h)) / 2.0)
        width = float(w)
        height = float(h)
        meas = np.array([[centerX],
                         [centerY],
                         [width],
                         [height]], np.float32)
        # print("measurement " + str(i + 1), meas)

        # -------------Correction--------------------
        # if 140 < i < 170 and 383 < i < 430:            # for dataset 10
        if 355 < i < 372: #these are the frames where mean shift is not giving the result due to occlusion for dataset 7
            not_found = not_found + 1
            if not_found > 100:
                first_detect = True
        else:
            not_found = 0
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
                print("Corrected State " + str(i), kf.statePost)
                widthCorr = kf.statePost[4, 0]
                heightCorr = kf.statePost[5, 0]
                startXCorr = kf.statePost[0, 0] - float(widthCorr / 2.0)
                endXCorr = kf.statePost[0, 0] + float(widthCorr / 2.0)
                startYCorr = kf.statePost[1, 0] - float(heightCorr / 2.0)
                endYCorr = kf.statePost[1, 0] + float(heightCorr / 2.0)
                track_window = (int(startXCorr), int(startYCorr), int(widthCorr), int(heightCorr))

                # rectangle around corrected result (Red)
                output_img = cv2.rectangle(frame, (int(startXCorr), int(startYCorr)), (int(endXCorr), int(endYCorr)),
                                           (0, 0, 255), 2)


        # print("Final State " + str(i), kf.statePost)
        cv2.imshow('tracking_output', output_img)
        k = cv2.waitKey(50)
        # cv2.destroyWindow('tracking_output' + str(i))

        # Print the output images
        if SAVE_IMAGES:
            output_file = OUTPUT_PATH + "{0:0=5d}".format(i) + '.jpg'
            print(output_file)
            cv2.imwrite(output_file, output_img)