__author__ = 'anushabala'
import operator
import cv2
import numpy as np
import os
import glob
import shutil
import re
from fabric.api import *
from multiprocessing import Process, Queue

file_pattern = r'ball([0-9]+)'


class RemoteHandler(object):
    def __init__(self, num_processes=5):
        self.num_processes = num_processes
        self.queues = [Queue()] * num_processes
        self.processes = []
        for i in xrange(num_processes):
            p = Process(target=self.copy_remote, args=(self.queues[i],))
            self.processes.append(p)
            p.start()

    @staticmethod
    def copy_remote(queue):
        while True:
            if not queue.empty():
                local_path, remote_path = queue.get()
                if local_path == 'quit' and remote_path == 'quit':
                    break
                copy_remote(local_path, remote_path)
        return

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


def write_frames(frames, out_path, remote_upload=False, remote_dir=None, remote_handler=None):
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)
    if remote_upload:
        make_remote_dir(remote_dir)

    num_processes = remote_handler.num_processes
    for (i, frame) in enumerate(frames):
        fname = 'frame_%d.png' % (i+1)
        frame_path = os.path.join(out_path, fname)
        cv2.imwrite(frame_path, frame)
        if remote_upload:
            remote_path = os.path.join(remote_dir, fname)

            remote_handler.queues[i % num_processes].put((frame_path, remote_path))
            # copy_remote(frame_path, remote_path)


def make_remote_dir(dirname):
    # subprocess.Popen(["ssh", "anusha@jamie.stanford.edu", ""])
    with settings(warn_only=True):
        run('rm -rf %s' % dirname)
        run('mkdir -p %s' % dirname)


def copy_remote(local_path, remote_path):
    with settings(warn_only=True):
        put(local_path, remote_path)


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