import particle_filter as pf
import template_matching as tm
import os
import cv2
import numpy as np

DATASET = ["CarChase1", "CarChase2", "DriftCar1", "Boxing1", "KinBall3", "Elephants"]
NUM_SET = 0
DATASET_NAME = DATASET[NUM_SET]
PREFIX = "data/" + DATASET_NAME + "/"
DATASET_PATH = PREFIX + "img/%05d.jpg"
GROUNDTRUTH_PATH = PREFIX + "groundtruth_rect.txt"
OUTPUT_PATH = "output/particle/" + DATASET_NAME + "/"
SAVE_IMAGES = True

# Particle filter parameters
NUM_PARTICLES = 500
MAX_ITERS = 50000
std = 15
maxNoise = 5
RESAMPLE_METHOD = 2
RESAMPLE_THRESHOLD = .5

# Template Matching parameters
METHODS_TEMP = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
                cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
TM_METHOD = 1

if __name__ == '__main__':
    # setup output folder
    if SAVE_IMAGES:
        if not os.path.exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)

    img_path = DATASET_PATH
    gt_path = GROUNDTRUTH_PATH

    cap = cv2.VideoCapture(img_path)
    ret, img = cap.read()

    if not ret:
        print("Error: Unable to open file.")
        exit(0)

    xlim = (0, np.size(img, 0))
    ylim = (0, np.size(img, 1))

    # Get bounding box for first frame from ground truth file
    fid = open(gt_path, 'r+')
    gt = [int(x) for x in fid.readline().split(',')]
    fid.close()

    gt = gt[1:-1]
    start_point = (gt[0], gt[1])
    end_point = (gt[0] + gt[2], gt[1] + gt[3])

    # Get the object template to be matched
    tm.markObject(start_point, end_point, img, 0)
    template = tm.cropObject(start_point, end_point, img)

    # init filter as a normal distribution near the initial pos of the points
    particles_start = pf.create_gaussian_particles(mean=start_point, std=(std, std), N=NUM_PARTICLES)
    particles_end = pf.create_gaussian_particles(mean=end_point, std=(std, std), N=NUM_PARTICLES)

    weights_start = np.ones(NUM_PARTICLES) / NUM_PARTICLES
    weights_end = np.ones(NUM_PARTICLES) / NUM_PARTICLES

    sp = start_point
    ep = end_point

    i = 0
    while cap.isOpened() and i < MAX_ITERS:

        i += 1
        ret, img = cap.read()
        if not ret:
            print("Error: Unable to open file.")
            exit(0)

        # Move particles based on estimated velocity
        pf.predict(particles_start, None, std=std)
        pf.predict(particles_end, None, std=std)
        print("&&")

        # Use template matching to get location of object
        start_point, end_point = tm.find_object(img, template, METHODS_TEMP[TM_METHOD])

        # Measure the particle w/ some noise in the sensor
        noise = np.random.uniform(0.0, maxNoise)

        sp_m = tuple(p + noise for p in start_point)
        ep_m = tuple(p + noise for p in end_point)

        # Update particle weight based on measurement
        pf.update(particles_start, weights_start, 'gaussian', sp_m)
        pf.update(particles_end, weights_end, 'gaussian', ep_m)

        # Get estimates of bounding box corners based on weighted average of particles
        sp_mu, _ = pf.estimate(particles_start, weights_start)
        ep_mu, _ = pf.estimate(particles_end, weights_end)
        sp = tuple(sp_mu)
        ep = tuple(ep_mu)

        # Visualize particles and bounding box on current frame
        pf.draw_particles(img, particles_start)
        pf.draw_particles(img, particles_end)
        pf.draw_box(img, [sp_mu, ep_mu], [start_point, end_point])
        cv2.imshow('image', img)

        # Save frame
        output_file = OUTPUT_PATH + "{0:0=5d}".format(i) + '.jpg'
        print(output_file)
        cv2.imwrite(output_file, img)
        cv2.waitKey(10)

        # Resample if necessary
        pf.resample(particles_start, weights_start, RESAMPLE_THRESHOLD * NUM_PARTICLES, RESAMPLE_METHOD)
        pf.resample(particles_end, weights_end, RESAMPLE_THRESHOLD * NUM_PARTICLES, RESAMPLE_METHOD)







