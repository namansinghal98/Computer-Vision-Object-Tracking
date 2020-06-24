import particle_filter as pf
import template_matching as tm
import os
import cv2
import numpy as np

# Some globals
NUM_PARTICLES = 500
MAX_ITERS = 50000
std = 15
# maxNoise = 5.
RESAMPLE_METHOD = 2
RESAMPLE_THRESHOLD = .5


DATA_SETS = ["CarChase1", "CarChase2", 'KinBall3',  # 0,1,2 [Occlusion: Kinball3]
             'DriftCar1', 'Boxing1', 'Elephants']  # 3,4,5 [Occlusion: Boxing1, Elephants]
DATASET_NUMBER = 3
DATASET_NAME = DATA_SETS[DATASET_NUMBER]

PREFIX = "data/" + DATASET_NAME + "/"
DATASET_PATH = PREFIX + "img/%05d.jpg"
GROUNDTRUTH_PATH = PREFIX + "groundtruth_rect.txt"
OUTPUT_PATH = "output/particle/" + DATASET_NAME + "/"
SAVE_IMAGES = True

maxNoise = 5
METHODS_TEMP = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
                cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
TM_METHOD = 1

OUTFILE_PATH = PREFIX + "estimations.log"
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

    fid = open(gt_path, 'r+')
    gt = [int(x) for x in fid.readline().split(',')]
    fid.close()

    gt = gt[1:-1]
    start_point = (gt[0], gt[1])
    end_point = (gt[0] + gt[2], gt[1] + gt[3])

    # Template Matching version
    tm.markObject(start_point, end_point, img, 0)
    template = tm.cropObject(start_point, end_point, img)

    # Or as a normal dist near the initial pos of the points
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

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        sp_v = tuple(p-q for p, q in zip(start_point, sp))
        ep_v = tuple(p-q for p, q in zip(end_point, ep))


        pf.predict(particles_start, None, std=std)
        pf.predict(particles_end, None, std=std)
        print("&&")

        start_point, end_point = tm.find_object(img, template, METHODS_TEMP[TM_METHOD])

        # Measure the particle w/ some noise in the sensor
        noise = np.random.uniform(0.0, maxNoise)

        sp_m = tuple(p + noise for p in start_point)
        ep_m = tuple(p + noise for p in end_point)

        # Update particle weight based on measurement
        pf.update(particles_start, weights_start, 'gaussian', sp_m)
        pf.update(particles_end, weights_end, 'gaussian', ep_m)
        # print(weights)

        # Check if we need to resample

        sp_mu, _ = pf.estimate(particles_start, weights_start)
        ep_mu, _ = pf.estimate(particles_end, weights_end)
        sp = tuple(sp_mu)
        ep = tuple(ep_mu)


        pf.draw_particles(img, particles_start)
        pf.draw_particles(img, particles_end)


        pf.draw_box(img, [sp_mu, ep_mu], [start_point, end_point])
        cv2.imshow('image', img)

        output_file = OUTPUT_PATH + "{0:0=5d}".format(i) + '.jpg'
        print(output_file)
        cv2.imwrite(output_file, img)
        cv2.waitKey(10)

        pf.resample(particles_start, weights_start, RESAMPLE_THRESHOLD * NUM_PARTICLES, RESAMPLE_METHOD)
        pf.resample(particles_end, weights_end, RESAMPLE_THRESHOLD * NUM_PARTICLES, RESAMPLE_METHOD)







