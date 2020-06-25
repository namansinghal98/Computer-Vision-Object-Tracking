import cv2
import numpy as np
from numpy.random import randn
from numpy.random import uniform
import scipy.stats

from filterpy.monte_carlo import systematic_resample
from filterpy.monte_carlo import residual_resample
from filterpy.monte_carlo import stratified_resample
from filterpy.monte_carlo import multinomial_resample


# To initialize particles before tracking starts
def create_uniform_particles(x_range, y_range, N):
    particles = np.empty((N, 2))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    return particles


# To generate new particles during tracking
def create_gaussian_particles(mean, std, N):
    particles = np.empty((N, 2))
    particles[:, 0] = mean[0] + (randn(N) * std[0])
    particles[:, 1] = mean[1] + (randn(N) * std[1])
    return particles


# Move particles based on estimated velocity of the object
def predict(particles, vel=None, std=15):
    N = len(particles)
    if vel is None or vel != vel:  # NaN check
        vel = [0, 0]

    particles[:, 0] += vel[0] + (randn(N) * std)
    particles[:, 1] += vel[1] + (randn(N) * std)


# Find out how likely the particles are based on their proximity to the object
def update(particles, weights, method, posm):
    pos = np.empty((len(particles), 2))
    pos[:, 0].fill(posm[0])
    pos[:, 1].fill(posm[1])

    # Get dist of each particle from measured position
    dist = np.linalg.norm(particles - pos, axis=1)

    if method == 'linear':
        # linear weighting:
        max_dist = np.amax(dist)
        dist = np.add(-dist, max_dist)
        weights.fill(1.0)
        weights *= dist
    elif method == 'gaussian':
        # Assign probabilities as Gaussian dist w/ mean=0 and std=X
        weights *= scipy.stats.norm.pdf(dist, loc=0, scale=15)
    else:
        print("Error: No such method to update weights.")
        exit(0)

    # to avoid zero weights
    weights += 1.e-300

    # normalize
    weights /= sum(weights)


def estimate(particles, weights):
    mean = np.average(particles, weights=weights, axis=0)
    var = np.average((particles - mean) ** 2, weights=weights, axis=0)
    return mean, var


# Resample function
def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights.resize(len(particles))
    weights.fill(1.0 / len(weights))


# Find meaningful particles, if neff < th then resample; (th = N/2)
def neff(weights):
    return 1.0 / np.sum(np.square(weights))


def getCentreFromWindow(win):
    xc = win[0] + win[2] / 2
    yc = win[1] + win[3] / 2
    centre = (xc, yc)
    return centre


def getTrackWindow(centre, win):
    x = int(np.floor(centre[0] - win[2] / 2))
    y = int(np.floor(centre[1] - win[3] / 2))
    track_window = (x, y, win[2], win[3])
    return track_window


def evalParticle(backproj, particlesT):
    p = [int(x) for x in particlesT[0]]
    q = [int(x) for x in particlesT[1]]
    return backproj[q, p]


def resample(particles, weights, th, method):
    if neff(weights) < th:
        indexes = []
        if method == 0:
            indexes = systematic_resample(weights)
        elif method == 1:
            indexes = multinomial_resample(weights)
        elif method == 2:
            indexes = stratified_resample(weights)
        elif method == 3:
            indexes = residual_resample(weights)
        resample_from_index(particles, weights, indexes)
        assert np.allclose(weights, 1 / len(particles))


def estVelocity(x_pos, y_pos, t_count=15, poly=1):
    if len(x_pos) > t_count:
        x_pos = x_pos[-t_count:]
        y_pos = y_pos[-t_count:]
    elif len(x_pos) < t_count:
        return None
    t = list(range(1, len(x_pos) + 1))
    x_model = np.polyfit(t, x_pos, 1)
    y_model = np.polyfit(t, y_pos, 1)
    x_vel = np.poly1d(x_model)
    y_vel = np.poly1d(y_model)
    return [(x_vel(t[-1]) - x_vel(t[0])) / (t[-1] - t[0]), (y_vel(t[-1]) - y_vel(t[0])) / (t[-1] - t[0])]


# To draw particles on the frame
def draw_particles(image, particles, color=(0, 0, 255)):
    for p in particles:
        image = cv2.circle(image, (int(p[0]), int(p[1])), 1, color, -1)


# To draw bounding box on the frame
def draw_box(im, estimated, measured):
    if measured is None:
        measured = [(0, 0), (0, 0)]
    estimated = [int(x) for li in estimated for x in li]
    measured = [int(x) for li in measured for x in li]
    im = cv2.rectangle(im, (estimated[0], estimated[1]), (estimated[2], estimated[3]), (255, 0, 0), 2)
    im = cv2.rectangle(im, (measured[0], measured[1]), (measured[2], measured[3]), (0, 255, 0), 2)
