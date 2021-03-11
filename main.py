import os
import re
import argparse
import youtube_dl
import sys
import warnings
from os.path import join
from signal import signal, SIGINT, SIG_DFL

from detection import test_full_image_network

def warn(*args, **kwargs):
    pass

warnings.warn = warn

def banner():
    print("[Fake Video Detector]")


def main():
    signal(SIGINT, SIG_DFL)

    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model_path', '-mi',dest='model', type=str, default='./models/x-model23.p')
    p.add_argument('--output_path', '-o',dest='videoOut', type=str, default='.')
    p.add_argument('--start_frame', type=int, default=0)
    p.add_argument('--end_frame', type=int, default=None)
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--fast', action='store_true')
    requiredNamed = p.add_argument_group('required arguments')
    requiredNamed.add_argument('--video_path', '-i', dest='videoIn', type=str, required=True)
    args = p.parse_args()

    video_path = args.videoIn

    prediction = None

    if video_path.endswith('.mp4'): #Take direct video file
        prediction = test_full_image_network(args.videoIn,args.model,args.videoOut,args.fast)
    else: # Download video from youtube-dl supported websites
        video_url = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', args.videoIn)
        if video_url:
            default_path = 'video.mp4'
            filename = ""
            ydl_opts = {'outtmpl':default_path}
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url[0], download=True)
                filename = ydl.prepare_filename(info)
            prediction = test_full_image_network(filename,args.model,args.videoOut, args.fast)
            os.remove(filename)
        else:
            print("Not valid input format")
            sys.exit(-1)

    print("Prediction of it being fake: " + str(prediction["score"]))
    print("Output video in: " + prediction["file"])

if __name__ == '__main__':
    banner()
    main()
