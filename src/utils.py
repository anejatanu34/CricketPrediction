__author__ = 'anushabala'
import operator
import cv2
import numpy as np
import os
import glob
import shutil
import re

file_pattern = r'ball([0-9]+)'
def load_centers(dirname, dims=(300,300)):
    centers = {}
    for filename in glob.glob(dirname+'/*.png'):
        filename = filename[filename.rfind('/')+1:]
        print filename, dirname
        shot_name = filename.replace(".png", "")
        img = cv2.imread(os.path.join(dirname, filename))
        if img.shape != dims:
            img = cv2.resize(img, dims, interpolation=cv2.INTER_AREA)
        centers[shot_name] = np.reshape(img, -1)

    return centers


def get_closest_center(centers, image):
    distances = {}
    for (name, center) in centers.iteritems():
        dist = np.linalg.norm(np.abs(center - image), 2)
        distances[name] = dist
    return min(distances, key=distances.get), sorted(distances.items(), key=operator.itemgetter(1))


def write_frames(frames, out_path):
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    for (i, frame) in enumerate(frames):
        fname = 'frame_%d.png' % (i+1)
        cv2.imwrite(os.path.join(out_path, fname), frame)


def add_frames_to_dir(frames, out_path):
    max_num = -1
    for f in os.listdir(out_path):
        if ".png" in f:
            f = f.replace(".png", "")
            match = re.match(file_pattern, f)
            if match:
                ball_num = int(match.group(1))
                if ball_num > max_num:
                    max_num = ball_num
    i = max_num
    for frame in frames:
        fname = 'frame_%d.png' % (i+1)
        cv2.imwrite(os.path.join(out_path, fname), frame)
        i += 1