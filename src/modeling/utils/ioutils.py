__author__ = 'anushabala'
import os
import numpy as np
from models import Outcome
import re
import json
from preprocess import preprocess_frames
import matplotlib.pyplot as plt

dir_pattern = r'ball([0-9]+)'


def read_frames(dirname, max_frames, p=0.5, mode='sample', **kwargs):
    frames = []
    i = 1
    while True:
        name = 'frame_%d.png' % i
        if os.path.exists(os.path.join(dirname, name)):
            frame = plt.imread(os.path.join(dirname, name), '.png')
            if np.random.uniform() <= p:
                frames.append(frame)
        else:
            break

        i += 1

    if mode == 'sample':
        return sample_frames(frames, max_frames)
    if mode == 'temporal':
        return sample_temporal_frames(frames, max_frames)


def sample_temporal_frames(frames, max_frames):
    window = np.ceil(len(frames)/ float(max_frames))
    selected = []
    indexes = np.arange(len(frames))
    start = 0
    while start < len(frames):
        end = min(start + window, len(frames))
        selected.append(np.random.choice(indexes[start:end]))
        start = end

    frames = np.array(frames)
    frames = frames[selected]
    return frames


def sample_frames(frames, max_frames):

    if len(frames) > max_frames:
        frames = np.array(frames)
        indexes = sorted(np.random.choice(np.arange(len(frames)), max_frames, replace=False))
        frames = frames[indexes]
        return frames

    while len(frames) < max_frames and max_frames > 0:
        frames.append(np.zeros_like(frames[0]))
    return np.array(frames)


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
def read_dataset(json_videos, sample_probability=1.0, max_items=-1, max_frames=60, mode='sample', **kwargs):
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
        ctr = 0
        all_clips = os.listdir(clips_dir)
        indexes = np.arange(len(all_clips))
        np.random.shuffle(indexes)
        for i in indexes:
            ball_dir = all_clips[i]
            match = re.match(dir_pattern, ball_dir)
            if match:
                ball_num = int(match.group(1))

                if ball_num not in illegal_balls:
                    frames = read_frames(os.path.join(clips_dir, ball_dir), p=sample_probability,
                                         mode=mode, max_frames=max_frames)
                    raw_frames, frames = preprocess_frames(frames, **kwargs)
                    raw_X.append(raw_frames)
                    X.append(frames)
                    y.append(labels[ball_num - 1])
                    del frames
                    ctr += 1
            if 0 < max_items == ctr:
                break
            if ctr % 25 == 0 and ctr > 0:
                print "Finished loading %d balls" % ctr

    return split_data(np.array(X), np.array(y).astype(np.int32), np.array(raw_X))


# todo this is just temporary, for testing our code!!!
# before running real experiments we should split our full dataset completely into train,
# test, and val ONCE and always use the same splits for all experiments
def split_data(X, y, raw_X, train_ratio=0.6, test_ratio=0.2, val_ratio=0.2):
    N = X.shape[0]
    shuffled_idx = np.arange(N)
    np.random.shuffle(shuffled_idx)

    num_train = max(train_ratio * N, 1)
    num_val = max(val_ratio * N, 1)

    train_X = X[shuffled_idx[0:num_train]]
    raw_train_X = raw_X[shuffled_idx[0:num_train]]
    train_y = y[shuffled_idx[0:num_train]]

    val_X = X[shuffled_idx[num_train:num_train+num_val]]
    raw_val_X = raw_X[shuffled_idx[num_train:num_train+num_val]]
    val_y = y[shuffled_idx[num_train:num_train+num_val]]

    test_X = X[shuffled_idx[num_train+num_val:]]
    raw_test_X = raw_X[shuffled_idx[num_train+num_val:]]
    test_y = X[shuffled_idx[num_train+num_val:]]

    data = {'train_X': train_X,
            'raw_train_X': raw_train_X,
            'train_y': train_y,
            'test_X': test_X,
            'raw_test_X': raw_test_X,
            'test_y': test_y,
            'val_X': val_X,
            'raw_val_X': raw_val_X,
            'val_y': val_y}
    del X
    return data
