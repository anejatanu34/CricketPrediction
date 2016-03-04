__author__ = 'anushabala'
from argparse import ArgumentParser
import cv2
import os
from utils import write_frames
import shutil

DEFAULT_SKIP_FACTOR = 2


def playback_help():
    print "Playback options:\n's': start/stop recording"
    print "'p': Pause/resume"
    print "'f': Fast-forward"
    print "'d': Resume playback after fast-forward"
    print "'x': Quit segmentation."
    print "'h': Print this help message again"


def main(args):
    capture = cv2.VideoCapture(args.vid)
    out_dir = args.out
    ball_ctr = max(args.start_ball, 0)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        print "Warning: the output directory specified already exists. Any directories already containing clips " \
              "(for ball %d and above) in that directory will be overwritten if segmentation is performed using this script.." % ball_ctr
    frame_ctr = 0

    recording = False
    frames = []
    last_recorded_frame = -1

    if args.continue_frame > 0:
        capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, args.continue_frame-1)
        frame_ctr = args.continue_frame

    print "Starting segmentation from frame %d." % args.continue_frame
    playback_help()
    skip_factor = args.skip_factor
    fast_fwd = False
    while True:
        ret, frame = capture.read()
        if not ret:
            print "Finished reading video"
            break

        frame_ctr+= skip_factor
        if skip_factor > 1:
            capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame_ctr)

        frame_resized = cv2.resize(frame, (300,350), interpolation = cv2.INTER_AREA)

        dim = args.display_size
        display_frame = cv2.resize(frame_resized, (dim, dim), interpolation=cv2.INTER_AREA)
        frame_cropped = frame_resized[0:300, :, :]
        if recording:
            frames.append(frame_cropped)

        cv2.imshow('video', display_frame)
        key = cv2.waitKey(30)
        key_code = key & 0xFF

        if key_code == ord('s'):
            if recording:
                last_recorded_frame = frame_ctr
                ball_ctr += 1
                if frames:
                    print "Writing ball %d at frame: %d" % (ball_ctr, frame_ctr)
                    ball_dir = os.path.join(out_dir, 'ball%d' % ball_ctr)
                    if os.path.exists(ball_dir):
                        shutil.rmtree(ball_dir)
                        os.makedirs(ball_dir)
                    write_frames(frames, ball_dir)
                frames = []
            recording = not recording
            if recording:
                print "Started recording at frame: %d" % frame_ctr
        elif key_code == ord('x'):
            if recording:
                ball_ctr += 1
                if frames:
                    ball_dir = os.path.join(out_dir, 'ball%d' % ball_ctr)
                    if os.path.exists(ball_dir):
                        shutil.rmtree(ball_dir)
                        os.makedirs(ball_dir)
                    write_frames(frames, ball_dir)
            print "Last ball written: %d\tLast frame recorded: %d\tLast frame read: %d" % (ball_ctr, last_recorded_frame, frame_ctr)
            break
        elif key_code == ord('p'):
            print "Pausing video. Press 'p' again to resume playback."
            cv2.waitKey(0) # pause as long as needed
            print "Resuming normal video playback."
        elif key_code == ord('f'):
            if not fast_fwd:
                skip_factor = 4
                fast_fwd = True
            else:
                skip_factor *= 2
            print "Fast forwarding video (displaying every %d frames). Press 'd' to resume default speed." % skip_factor
        elif key_code == ord('d'):
            print "Playing video at default speed (displaying every %d frames)." % args.skip_factor
            fast_fwd = False
            skip_factor = args.skip_factor
        elif key_code == ord('h'):
            playback_help()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--vid', type=str, required=True, help='path to video to preprocess')
    parser.add_argument('--out', type=str, help='Directory to output clips to')
    parser.add_argument('--continue_frame', type=int, default=-1, help='Continue segmenting video from provided frame number')
    parser.add_argument('--start_ball', type=int, default=0,
                        help='(0-indexed) number of first ball in video (or last ball recorded, if continuing from middle) - '
                             'defaults to 0 (first ball in video is first ball of match)')
    parser.add_argument('--display_size', type=int, default=300, help='Size of video to display')
    parser.add_argument('--skip_factor', type=int, default=DEFAULT_SKIP_FACTOR, help='Number of frames to skip while playing video (by default set to 2, i.e. every other frame is played)')
    clargs = parser.parse_args()
    main(clargs)
