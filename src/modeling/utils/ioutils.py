__author__='anushabala'
import os
import numpy as np
from models import Outcome
import re
import json
from preprocess import preprocess_frames
import matplotlib.pyplot as plt
import scipy

dir_pattern = r'ball([0-9]+)'


def read_frames(dirname, max_frames, p=0.5, mode='sample', **kwargs):
    frames = []
    i = 1
    while True:
        name = 'frame_%d.png' % i
        if os.path.exists(os.path.join(dirname, name)):
            frame = plt.imread(os.path.join(dirname, name), '.png')
            #if np.random.uniform() <= p:
            #    frames.append(frame)
            frames.append(frame)
        else:
            break
        i += 1

    if mode == 'sample':
        return sample_frames(frames, max_frames)
    if mode == 'temporal':
        return sample_temporal_frames(frames, max_frames)


def sample_temporal_frames(frames, max_frames):
    if len(frames) < max_frames:
        while len(frames) < max_frames:
            frames.append(np.zeros_like(frames[0]))
        return np.array(frames)

    window = 1.0 * len(frames)/ max_frames
    selected = []
    indexes = np.arange(len(frames))
    start = 0
    while start < len(frames) - 1 and len(selected) < max_frames:
        end = min(start + window, len(frames))
        selected.append(np.random.choice(indexes[int(start):int(end)]))
        start = end
    frames = np.array(frames)
    frames = frames[selected]
    if len(selected) < max_frames:
        padded = list(frames)
        while len(padded) < max_frames:
            padded.append(np.zeros_like(padded[0]))
        return np.array(padded)

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


def get_frames(video_num, ball_num, videos, sample_probability, mode, max_frames, **kwargs):

    clips_dir = videos[video_num-1]["clips"]
    all_clips = os.listdir(clips_dir)
    ball_dir = 'ball'+str(ball_num) # ll_clips[ball_num-1]
    frames = read_frames(os.path.join(clips_dir, ball_dir), p=sample_probability,
                         mode=mode, max_frames=max_frames)
    raw_frames, frames = preprocess_frames(frames, **kwargs)

    return raw_frames, frames


# todo add support to read in more class types if needed
def read_dataset_tvt(json_videos, sample_probability=1.0, max_frames=60, mode='sample',
                     class_dist=[0.35,0.25,0.2,0.2], tvt_split=[1,1,1], ids_file='clip_ids.txt', **kwargs):
    videos = json.load(open(json_videos, 'r'), encoding='utf-8')

    data = {}
    clip_ids = {} 

    # collect all video-ball labels
    labels_mapping = [[], [], []] # [[video_num], [ball_num], [label]]
    video_num = 1
    for video in videos:
        clips_dir = video["clips"]
        all_clips = os.listdir(clips_dir)
        all_clips_nums = [int(xx[4:]) for xx in all_clips if 'ball' in xx]
        innings1 = video["innings1"]
        innings2 = video["innings2"]
        labels, illegal_balls = read_cricket_labels(innings1, innings2)
        clip_ctr = 0
        for ll in range(len(labels)): # also ball_num
            if ll+1 not in all_clips_nums:    
                continue
            labels_mapping[0].append(video_num)
            labels_mapping[1].append(ll+1)
            labels_mapping[2].append(labels[ll])
            clip_ctr += 1
        video_num += 1

    ctr = 0
    # Get val set
    data['val_X']  = []
    data['val_y']  = []
    clip_ids['val'] = []
    for i in range(tvt_split[1]):
        ind = np.random.randint(0,len(labels_mapping[0]))
        raw_frames, frames = get_frames(labels_mapping[0][ind], labels_mapping[1][ind], videos, sample_probability, mode, max_frames, **kwargs)
        data['val_X'].append(frames)
        data['val_y'].append(labels_mapping[2][ind])
        clip_ids['val'].append(str(labels_mapping[0][ind])+','+str(labels_mapping[1][ind])+','+str(labels_mapping[2][ind]))
        del labels_mapping[0][ind]
        del labels_mapping[1][ind]
        del labels_mapping[2][ind]
        del frames
        ctr += 1
        if ctr % 25 == 0 and ctr > 0:
            print "Finished loading val %d balls" % ctr
    
    # Get test set    
    data['test_X']  = []
    data['test_y']  = []
    clip_ids['test'] = []
    for i in range(tvt_split[2]):
        ind = np.random.randint(0,len(labels_mapping[0]))
        raw_frames, frames = get_frames(labels_mapping[0][ind], labels_mapping[1][ind], videos, sample_probability, mode, max_frames, **kwargs)
        data['test_X'].append(frames)
        data['test_y'].append(labels_mapping[2][ind])
        clip_ids['test'].append(str(labels_mapping[0][ind])+','+str(labels_mapping[1][ind])+','+str(labels_mapping[2][ind]))
        del labels_mapping[0][ind]
        del labels_mapping[1][ind]
        del labels_mapping[2][ind]
        del frames
        ctr += 1
        if ctr % 25 == 0 and ctr > 0:
            print "Finished loading test %d balls" % ctr

    # Get train set
    # determining allocation to each class of videos
    data['train_X']  = []
    data['train_y']  = []
    clip_ids['train'] = []
    counts = (tvt_split[0] * np.array(class_dist)).astype(int)
    if np.sum(counts) < tvt_split[0]:
        counts[:tvt_split[0]-np.sum(counts)] += 1

    # Add video-ball clips
    for cc in range(len(counts)):
        inds = [xx for xx in range(len(labels_mapping[2])) if labels_mapping[2][xx]==cc] # index of all 0/1/2/3s
        np.random.shuffle(inds)

        # number of repeats for each ball
        num_repeats = [0 for ii in range(len(inds))]
        for ii in range(counts[cc]):
            num_repeats[ii%len(inds)] += 1
        # add video-ball data
        for ii in range(len(inds)):
            ind = inds[ii]
            for rr in range(num_repeats[ii]):
                raw_frames, frames = get_frames(labels_mapping[0][ind], labels_mapping[1][ind], videos, sample_probability, mode, max_frames, **kwargs)
                data['train_X'].append(frames)
                data['train_y'].append(labels_mapping[2][ind])
                clip_ids['train'].append(str(labels_mapping[0][ind])+','+str(labels_mapping[1][ind])+','+str(labels_mapping[2][ind]))
                del frames

                ctr += 1
                if ctr % 25 == 0 and ctr > 0:
                    print "Finished loading train %d balls" % ctr

    # print clips ids
    with open(ids_file, 'w') as f:
        f.write(json.dumps(clip_ids)+'\n')

    # turn all dict values to np.arrays
    for k in data.keys():
        data[k] = np.array(data[k])
    return data


def read_dataset(json_videos, sample_probability=1.0, max_items=-1, max_frames=60, mode='sample', class_dist=[0.35,0.25,0.2,0.2], **kwargs):
    videos = json.load(open(json_videos, 'r'))
    X = []
    raw_X = []
    y = []

    # determining allocation to each class of videos
    counts = (max_items * np.array(class_dist)).astype(int)
    if np.sum(counts) < max_items:
        counts[:max_items-np.sum(counts)] += 1
    print 'clips of class [0, 1, 2, 3]:', counts   

    # collect all video-ball labels
    labels_mapping = [[], [], []] # [[video_num], [ball_num], [label]]
    video_num = 1
    for video in videos:
        clips_dir = video["clips"]
        all_clips = os.listdir(clips_dir)
        all_clips_nums = [int(xx[4:]) for xx in all_clips]
        innings1 = video["innings1"]
        innings2 = video["innings2"]
        labels, illegal_balls = read_cricket_labels(innings1, innings2)
        clip_ctr = 0
        for ll in range(len(labels)): # also ball_num
            if ll+1 not in all_clips_nums:    
                continue
            labels_mapping[0].append(video_num)
            labels_mapping[1].append(ll+1)
            labels_mapping[2].append(labels[ll])
            clip_ctr += 1
        video_num += 1

    # Add video-ball clips
    ctr = 0
    for cc in range(len(counts)):
        inds = [xx for xx in range(len(labels_mapping[2])) if labels_mapping[2][xx]==cc] # index of all 0/1/2/3s
        np.random.shuffle(inds)

        # number of repeats for each ball
        num_repeats = [0 for ii in range(len(inds))]
        for ii in range(counts[cc]):
            num_repeats[ii%len(inds)] += 1
        # add video-ball data
        for ii in range(len(inds)):
            ind = inds[ii]
            for rr in range(num_repeats[ii]):
                raw_frames, frames = get_frames(labels_mapping[0][ind], labels_mapping[1][ind], videos, sample_probability, mode, max_frames, **kwargs)
                raw_X.append(raw_frames)
                X.append(frames)
                y.append(labels_mapping[2][ind])
                del frames
                del raw_frames
                ctr += 1

                if ctr % 25 == 0 and ctr > 0:
                    print "Finished loading %d balls" % ctr

    return split_data(np.array(X), np.array(y).astype(np.int32))


# todo this is just temporary, for testing our code!!!
# before running real experiments we should split our full dataset completely into train,
# test, and val ONCE and always use the same splits for all experiments
def split_data(X, y, train_ratio=0.6, test_ratio=0.2, val_ratio=0.2):
    N = X.shape[0]
    shuffled_idx = np.arange(N)
    np.random.shuffle(shuffled_idx)

    num_train = max(train_ratio * N, 1)
    num_val = max(val_ratio * N, 1)

    train_X = X[shuffled_idx[0:num_train]]
    # raw_train_X = raw_X[shuffled_idx[0:num_train]]
    train_y = y[shuffled_idx[0:num_train]]

    val_X = X[shuffled_idx[num_train:num_train+num_val]]
    # raw_val_X = raw_X[shuffled_idx[num_train:num_train+num_val]]
    val_y = y[shuffled_idx[num_train:num_train+num_val]]

    test_X = X[shuffled_idx[num_train+num_val:]]
    # raw_test_X = raw_X[shuffled_idx[num_train+num_val:]]
    test_y = X[shuffled_idx[num_train+num_val:]]

    data = {'train_X': train_X,
            # 'raw_train_X': raw_train_X,
            'train_y': train_y,
            'test_X': test_X,
            # 'raw_test_X': raw_test_X,
            'test_y': test_y,
            'val_X': val_X,
            # 'raw_val_X': raw_val_X,
            'val_y': val_y}
    del X
    return data
