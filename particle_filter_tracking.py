import particle_filter as pf
import template_matching as tm
import cv2
import numpy as np

# Some globals
NUM_PARTICLES = 500
MAX_ITERS = 50000
std = 15
# maxNoise = 5.
RESAMPLE_METHOD = 2
RESAMPLE_THRESHOLD = .5
DATASET = ["CarChase1", "CarChase2", "CarChase3", "Billiards2", "ISS", "Boat"]
NUM_SET = 1
DATASET_NAME = DATASET[NUM_SET]
PREFIX = "data/" + DATASET_NAME + "/"
DATASET_PATH = PREFIX + "img/%05d.jpg"
GROUNDTRUTH_PATH = PREFIX + "groundtruth_rect.txt"
OUTPUT_PATH = "output/particle/" + DATASET_NAME + "/"

maxNoise = 5
METHODS_TEMP = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
                cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
TM_METHOD = 1

OUTFILE_PATH = PREFIX + "estimations.log"
if __name__ == '__main__':

    img_path = DATASET_PATH
    gt_path = GROUNDTRUTH_PATH
    cap = cv2.VideoCapture(img_path)
    ret, img = cap.read()
    if not ret:
        print("Error: Unable to open file.")
        exit(0)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    xlim = (0, np.size(img, 0))
    ylim = (0, np.size(img, 1))

    fid = open(gt_path, 'r+')
    gt = [int(x) for x in fid.readline().split(',')]
    fid.close()

    # fid = open(OUTFILE_PATH, 'w+')
    gt = gt[1:-1]
    start_point = (gt[0], gt[1])
    end_point = (gt[0] + gt[2], gt[1] + gt[3])

    # BackProj version:
    # x, y, w, h = gt
    # roi_hist = tm.hsv_hist_for_win(img, (x, y, w, h), 0.4)
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # params are : images, channels, hist, ranges, scale
    # hist_bp = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # Template Matching version
    tm.markObject(start_point, end_point, img, 0)
    template = tm.cropObject(start_point, end_point, img)

    # init filter
    # Either uniformly across the image
    # particles_start = pf.create_uniform_particles(xlim, ylim, NUM_PARTICLES)
    # particles_end = pf.create_uniform_particles(xlim, ylim, NUM_PARTICLES)

    # Or as a normal dist near the initial pos of the points
    particles_start = pf.create_gaussian_particles(mean=start_point, std=(std, std), N=NUM_PARTICLES)
    particles_end = pf.create_gaussian_particles(mean=end_point, std=(std, std), N=NUM_PARTICLES)

    weights_start = np.ones(NUM_PARTICLES) / NUM_PARTICLES
    weights_end = np.ones(NUM_PARTICLES) / NUM_PARTICLES

    # For object center:
    # init_pos = tuple([x + w/2., y + h/2.])
    # particles = pf.create_gaussian_particles(mean=init_pos, std=(std, std), N=NUM_PARTICLES)
    # weights = np.ones(NUM_PARTICLES) / NUM_PARTICLES

    # Show where particles have been initialized
    # pf.draw_particles(img, particles)
    # img = cv2.circle(img, start_point, 2, (0, 255, 0), -1)
    # cv2.imshow('image', img)
    # cv2.waitKey(2500)

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

        # Move particles based on estimated velocity
        # pf.predict(particles_start, sp_v, std=std)
        # pf.predict(particles_end, ep_v, std=std)
        pf.predict(particles_start, None, std=std)
        pf.predict(particles_end, None, std=std)

        start_point, end_point = tm.find_object(img, template, METHODS_TEMP[TM_METHOD])

        # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # hist_bp = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        # cv2.imshow('hist_bp', hist_bp)
        # cv2.waitKey(0)
        # exit(0)
        # pf.predict(particles, None, 15)
        # particles = particles.clip(np.zeros(2), np.array((img.shape[1], img.shape[0]))-1)
        # f = tm.evalPart(hist_bp, particles.T)

        # good = particles[f >= 1]
        # bad = particles[f < 1]
        # pf.draw_particles(img, good, (255, 0, 0))
        # pf.draw_particles(img, bad)

        # weights = np.float32(f.clip(1))
        # weights /= np.sum(weights)
        # pos = np.sum(particles.T * weights, axis=1).astype(int)
        # img = cv2.circle(img, (pos[0], pos[1]), 4, (0, 255, 0), -1)
        # pf.resample(particles, weights, RESAMPLE_THRESHOLD * NUM_PARTICLES, RESAMPLE_METHOD)

        # Measure the particle w/ some noise in the sensor
        noise = np.random.uniform(0.0, maxNoise)
        # gt = [int(x) for x in fid.readline().split(sep=',')]
        # gt = gt[1:-1]

        # x_a = (gt[0] + (gt[2] / 2))
        # y_a = (gt[1] + (gt[3] / 2))
        #
        sp_m = tuple(p + noise for p in start_point)
        ep_m = tuple(p + noise for p in end_point)

        # Update particle weight based on measurement
        pf.update(particles_start, weights_start, 'gaussian', sp_m)
        pf.update(particles_end, weights_end, 'gaussian', ep_m)
        # print(weights)

        # Check if we need to resample
        # print(pf.neff(weights))

        sp_mu, _ = pf.estimate(particles_start, weights_start)
        ep_mu, _ = pf.estimate(particles_end, weights_end)
        sp = tuple(sp_mu)
        ep = tuple(ep_mu)
        #
        # print(str(sp) + str(ep) + str(start_point) + str(end_point))
        # print(sp_mu, ep_mu)

        pf.draw_particles(img, particles_start)
        pf.draw_particles(img, particles_end)

        # img = cv2.circle(img, (int(mu[0]), int(mu[1])), 2, (0, 255, 0), -1)
        # img = cv2.circle(img, (int(x_a), int(y_a)), 2, (255, 0, 0), -1)
        # cv2.imshow('image', img)
        # cv2.waitKey(2500)
        # tm.markObject(tuple(sp_mu), tuple(ep_mu), img, 0)
        pf.draw_box(img, [sp_mu, ep_mu], [start_point, end_point])
        cv2.imshow('image', img)

        output_file = OUTPUT_PATH + "{0:0=5d}".format(i) + '.jpg'
        cv2.imwrite(output_file, img)
        cv2.waitKey(10)

        pf.resample(particles_start, weights_start, RESAMPLE_THRESHOLD * NUM_PARTICLES, RESAMPLE_METHOD)
        pf.resample(particles_end, weights_end, RESAMPLE_THRESHOLD * NUM_PARTICLES, RESAMPLE_METHOD)
    # fid.close()







