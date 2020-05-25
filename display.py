import cv2
import os

if __name__ == '__main__':

    # set_name = "CarChase2"
    set_name = "KinBall3"

    file_source = "./data/TinyTLP/" + set_name + '/img/'
    truth_source = "./data/TinyTLP/" + set_name + '/groundtruth_rect.txt'

    print(file_source)

    bounding_box = []
    with open(truth_source, 'r') as f:
        for line in f:
            line = line.split(',')
            line = [int(x) for x in line]
            box = tuple(line[1:6])
            bounding_box.append(box)

    images = os.listdir(file_source)
    images.sort()
    frames = []

    for img_name in images:
        path = os.path.join(file_source, img_name)
        frame = cv2.imread(path)
        frames.append(frame)

    cnt = 0
    for frame in frames:
        bbox = bounding_box[cnt]
        p1 = (bbox[0], bbox[1])
        p2 = (bbox[0] + bbox[2], bbox[1] + bbox[3])
        cnt += 1
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        cv2.imshow('video', frame)
        cv2.waitKey(30)