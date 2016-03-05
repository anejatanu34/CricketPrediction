__author__ = 'anushabala'
import matplotlib.pyplot as plt
import os
import numpy as np
from vgg_utils import Outcome
import re
import json
from preprocess import preprocess_frames

dir_pattern = r'ball([0-9]+)'


def read_frames(dirname, p=1.0):
    frames = []
    i = 1
    while True:
        name = 'frame_%d.png' % i
        if os.path.exists(os.path.join(dirname,name)):
            frame = plt.imread(os.path.join(dirname, name))
            if np.random.uniform() <= p:
                frames.append(frame)
        else:
            break

        i+=1

    return frames


def read_cricket_labels(innings1_file, innings2_file):
    print "Reading commentary"
    labels = []
    illegal_balls = []
    ball_num = 1
    for innings in [innings1_file, innings2_file]:
        infile = open(innings, 'r')
        for line in infile.readlines():
            cols = line.strip().split(',')
            detailed_outcome = cols[4]
            if 'wide' in detailed_outcome or 'no ball' in detailed_outcome:
                illegal_balls.append(ball_num)

            outcome = Outcome.get_label_from_commentary(cols[-1])
            labels.append(outcome)
            ball_num += 1

    return labels, illegal_balls


# todo add support to read in more class types if needed
def read_dataset(json_videos, sample_probability=1.0, **kwargs):
    videos = json.load(open(json_videos, 'r'), encoding='utf-8')
    X = []
    raw_X = []
    y = []
    for video in videos:
        clips_dir = video["clips"]
        innings1 = video["innings1"]
        innings2 = video["innings2"]

        labels, illegal_balls = read_cricket_labels(innings1, innings2)

        print "Reading clips from %s" % clips_dir
        for ball_dir in os.listdir(clips_dir):
            match = re.match(dir_pattern, ball_dir)
            ctr = 0
            if match:
                ball_num = int(match.group(1))
                if ctr % 25 == 0 and ctr > 0:
                    print "Finished loading %d balls" % ctr

                if ball_num not in illegal_balls:
                    frames = read_frames(os.path.join(clips_dir, ball_dir), sample_probability)
                    raw_frames, frames = preprocess_frames(frames, **kwargs)
                    raw_X.append(raw_frames)
                    X.append(frames)
                    y.append(labels[ball_num - 1])

                ctr += 1

    return np.array(X), np.array(y), np.array(raw_X)

