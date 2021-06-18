# https://github.com/gurkirt/road-dataset/blob/5eee122c42830807ff6bd7cb12d0252b63ece0bc/extract_videos2jpgs.py

import os
import argparse

def main(video_file, outdir):

    video_name = os.path.splitext(os.path.basename(video_file))[0]

    images_dir = os.path.join(outdir, video_name)
    
    if not os.path.isdir(images_dir):
        os.makedirs(images_dir)

    imglist = os.listdir(images_dir)
    imglist = [img for img in imglist if img.endswith('.jpg')]

    if len(imglist)<10: # very few or no frames try extracting again
        command = 'ffmpeg  -i {} -q:v 1 {}/%05d.jpg'.format(video_file, images_dir) # extract at very good quality of 1
        print('run', command)
        os.system(command)
    
    imglist = os.listdir(images_dir)
    imglist = [img for img in imglist if img.endswith('.jpg')]
    
    return len(imglist)
    


if __name__ == '__main__':

    p = argparse.ArgumentParser(description='extract frame from videos')
    p.add_argument('--outdir', type=str, help='Output directory (images will be put here).')
    p.add_argument('--files', nargs='+', type=str, default=[],
        help='Paths to video files.')

    args = p.parse_args()
    print(f'Processing {len(args.files)} files')
    for i, videofile in enumerate(args.files):
            print(f'videofile {i}: {videofile}')
            main(videofile, args.outdir)

