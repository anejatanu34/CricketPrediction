__author__ = 'anushabala'
import matplotlib.pyplot as plt
import os
import numpy as np
from vgg_utils import Outcome
import re
import json

dir_pattern = r'ball([0-9]+)'


def read_frames(dirname, p=1.0):
    frames = []
    for name in os.listdir(dirname):
        frame = plt.imread(os.path.join(dirname, name))
        if np.random.uniform() <= p:
            frames.append(frame)

    return frames


def read_cricket_labels(innings1_file, innings2_file):
    labels = []
    illegal_balls = []
    ball_num = 1
    for innings in [innings1_file, innings2_file]:
        infile = open(innings, 'r')
        for line in infile.readlines():
            cols = line.strip().split(',')
            detailed_outcome = cols[4]
            if 'wide' or 'no ball' in detailed_outcome:
                illegal_balls.append(ball_num)

            outcome = Outcome.get_label_from_commentary(cols[-1])
            labels.append(outcome)
            ball_num += 1

    return labels, illegal_balls


# todo add support to read in more class types if needed
def read_dataset(json_videos, sample_probability=1.0):
    videos = json.load(json_videos, encoding='utf-8')
    X = []
    y = []
    for video in videos:
        clips_dir = video["clips"]
        innings1 = video["innnings1"]
        innings2 = video["innings2"]

        labels, illegal_balls = read_cricket_labels(innings1, innings2)

        for ball_dir in os.listdir(clips_dir):
            match = re.match(dir_pattern, ball_dir)
            if match:
                ball_num = int(match.group(1))
                if ball_num not in illegal_balls:
                    frames = read_frames(os.path.join(clips_dir, ball_dir), sample_probability)
                    X.append(frames)
                    y.append(labels[ball_num - 1])

    return X, y

