import cv2
import os
import numpy as np

DATA_SETS = ["CarChase1", "CarChase2", 'KinBall3',
             'DriftCar1', 'Boxing1', 'Elephants']
DATASET_NUMBER = 3
DATASET_NAME = DATA_SETS[DATASET_NUMBER]

OUTPUT_PATH = "output/"
SAVE_IMAGES = True

TRACKERS = ['kalman', 'particle', 'meanshift']


# TRACKERS = ['meanshift', 'meanshift_kalman', 'meanshift_particle']

if __name__ == '__main__':

    output_path = 'output/'

    image_list = {}
    for tracker in TRACKERS:
        path = os.path.join(os.path.join(output_path, tracker), DATASET_NAME)
        images = os.listdir(path)
        images.sort()
        image_list[tracker] = images

    for i in range(540):
        frames = []
        for tracker in TRACKERS:
            path = os.path.join(os.path.join(output_path, tracker), DATASET_NAME)
            image_name = image_list[tracker][i]
            path = os.path.join(path, image_name)
            frame = cv2.imread(path)

            frame = cv2.putText(frame, tracker, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (50, 50, 50), 2, cv2.LINE_AA)
            frame = cv2.resize(frame, (640, 480))
            frames.append(frame)

        combined_image = np.hstack(frames)
        cv2.imshow('output', combined_image)
        cv2.waitKey(30)