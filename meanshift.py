import cv2
import os
import numpy as np
import particle_filter as pf

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
OUTPUT_PATH = "output/particle/" + DATASET_NAME + "/"

# Save Images or Not
SAVE_IMAGES = True

# Debugging mode for extra outputs
DEBUG = True

# Dataset Parameter
PARAMS = [1, 1,
          2, 2, 2,
          3, 3, 3, 3,
          3, 1]
PARAM_NUMBER = PARAMS[DATASET_NUMBER]

TRACK_NAME = ["Meanshift", "Particle", "Kalman"]
TRACK_NUMBER = 1
TRACK_METHOD = TRACK_NAME[TRACK_NUMBER]

# Particle Filter Parameters
STD = 10
NUM_PARTICLES = 200
PF_RESAMPLE_THRESH = 0.5
PF_RESAMPLE_METHOD = 2
GOOD_THRESH = 150
T_COUNT = 60


if __name__ == '__main__':

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
            os.makedirs(OUTPUT_PATH)

        # Use this for Linux
        # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
        # Make sure size matches output frame i.e. y, x
        out = cv2.VideoWriter(OUTPUT_PATH + 'video.avi', fourcc, 30, (first_frame.shape[1], first_frame.shape[0]))

    # Set up tracking methods
    # particles = []
    # weights = []
    if TRACK_NUMBER == 0:
        # Only Meanshift
        pass
    elif TRACK_NUMBER == 1:  # Particle
        particles = pf.create_gaussian_particles(mean=(x + w / 2, y + h / 2), std=(STD, STD), N=NUM_PARTICLES)
        weights = np.ones(NUM_PARTICLES) / NUM_PARTICLES
        x_pos = []
        y_pos = []
        # Use Particle Filter also
    elif TRACK_NUMBER == 2:  # Kalman
        pass

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
    roi = first_frame[int(y + h*red_factor/2):int(y+h*(2-red_factor)/2), int(x+w*red_factor/2):int(x+w*(2-red_factor)/2)]

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

        # 8) Convert frame to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 9) Calculate the BackProject of frame with the histogram
        dst = cv2.calcBackProject([hsv], hist_channels, roi_hist, hist_range, 1)

        # 10 a) Perform meanshift tracking
        ret, track_window_obs = cv2.meanShift(dst, track_window, term_crit)
        if TRACK_NUMBER == 0:
            track_window = track_window_obs
        # 10 b) Use Particle Filter to improve observation accuracy
        elif TRACK_NUMBER == 1:
            # TODO add velocity model
            # use last XX frames to extrapolate motion using np.polyfit
            # Will help estimate recent velocity of motion
            # Problem: estimated centroid is very random, can lead to high variance of movement
            # Solution: Can consider using better weighting model, or less randomness
            # Otherwise just try assuming a linear model and see how it works

            # i) Estimate current velocity of object using linear regression
            vel = pf.estVelocity(x_pos, y_pos, t_count=T_COUNT) # Use 3 with KinBall3
            # ii) Predict position of object
            pf.predict(particles, vel=vel, std=STD)

            centre_obs = pf.getCentreFromWindow(track_window_obs)
            noise = np.random.uniform(0.0, STD)
            centre_obs = tuple(p + noise for p in centre_obs)

            # iii) Update weights of particles based on dist from observation
            pf.update(particles, weights, 'gaussian', centre_obs)

            # iv) Esimate new position based on particles and weights
            centre_est, _ = pf.estimate(particles, weights)

            x_pos.append(centre_est[0])
            y_pos.append(centre_est[1])
            track_window = pf.getTrackWindow(centre_est, track_window_obs)

            # TODO Use evalPart to estimate how good or bad the particle is rather than distance from centre
            # evalPart checks how well each particle fits the histogram obtained using backprop
            particles = particles.clip(np.zeros(2), np.array((frame.shape[1], frame.shape[0]))-1)

            # v) Evaluate particles based on backprop
            f = pf.evalParticle(dst, particles.T)
            good = particles[f >= GOOD_THRESH]
            bad = particles[f < GOOD_THRESH]

            # vi) Resample particles if required
            if good.size < PF_RESAMPLE_THRESH * NUM_PARTICLES:
                pf.resample(particles, weights, 1 * NUM_PARTICLES, PF_RESAMPLE_METHOD)
            # pf.resample(particles, weights, PF_RESAMPLE_THRESH * NUM_PARTICLES, PF_RESAMPLE_METHOD)
        elif TRACK_NUMBER == 2:
            pass

        # 11) Draw the resultant box on image
        x, y, w, h = track_window
        output_img = cv2.rectangle(frame, (x, y), (x+w, y+h), 255, 2)

        if DEBUG:
            cv2.imshow("backprop", dst)
            if TRACK_NUMBER == 1:
                pf.draw_particles(frame, good, (0, 255, 0))
                pf.draw_particles(frame, bad, (0, 255, 0))
                # frame = cv2.circle(frame, (int(centre_est[0]), int(centre_est[1])), 2, (0, 255, 0), -1)
                # frame = cv2.circle(frame, (int(centre_obs[0]), int(centre_obs[1])), 2, (255, 0, 0), -1)

        # 12) Display the output
        cv2.imshow('tracking_output', output_img)
        k = cv2.waitKey(30)

        # Print the output images
        if SAVE_IMAGES:
            output_file = OUTPUT_PATH + "{0:0=5d}".format(i) + '.jpg'
            print(output_file)
            cv2.imwrite(output_file, output_img)
            out.write(frame)

    if SAVE_IMAGES:
        out.release()
