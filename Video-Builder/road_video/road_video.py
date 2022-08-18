import os
import cv2
import json
import csv
import copy
import numpy as np

from tqdm import tqdm
from road_video.road_frame_builders import build_det_frame, build_track_frame, build_action_frame

class ROADDebugVideo(object):
    def __init__ (self, opts):
        ''' ROAD Video Dataloader class:
            Reads ROAD image jpgs and produces a video of tracks on them. 

            Args:
                opts: check config file for explanations of each of the parameters
        '''
        # Load config opts into member variables
        self.load_main_opts(opts)

        # directories of all the possible ROAD videos we can use
        self.video_names = list(self.det_dict['db'].keys())
        if opts.list_videos:
            print("Available Videos to Debug with:")
            for name in self.video_names:
                print(name)

        # array of images which will build the video
        self.video_arr = [] 
        self.video_h = 0
        self.video_w = 0

        self.build_video(self.video_name)
    
    def load_main_opts(self, opts):
        ''' Load Opts: Loads config parameters into member variables. Creates annotation dictionaries if provided an annotation path
        '''
        # Load general params that format the video
        video_opts = opts.Video_Builder
        self.save_path = video_opts.save_path # directory to save the debug video to
        self.video_path = video_opts.video_path # path to all the videos
        self.video_name = opts.video_name

        # cv2 parameters based on the number of video streams
        self.video_formatting_opts = opts.Video_Formatting

        # Load dictionaries and readers for all the available annotations
        if opts.Detector.detections_path is not None: # detections 
            with open(opts.Detector.detections_path, "r") as f:
                fs = f.read()
                self.det_dict = json.loads(fs) # detections dictionary
                self.detection_colours = {} # dictionary for detection colours, coloured by agent class         
    
        if opts.Tracker.tracks_path is not None: # tracks
            self.track_opts = opts.Tracker
            with open(self.track_opts.tracks_path, "r") as f:
                fs = f.read()
                self.track_dict = json.loads(fs) # tracks dictionary
                self.track_colours = {} # dictionary for track colours, coloured by track id
        
        if opts.Action_Classifier.actions_path is not None: # actions
            self.action_opts = opts.Action_Classifier
            with open(opts.Action_Classifier.actions_path, "r") as f:
                self.action_dict = json.load(f) # actions csv reader     
                self.action_colours = {} # dictionary for action colours, coloured by action class

    # TODO the builder needs to change, create a util function that creates titles accordingly, and also add boxes to an image
    # provided we give the proper formatting

    def build_video(self, video_name):
        ''' Build Track video:
            Builds the ROAD video with tracked boxes for the specified video name. Also builds a separate detections
            video stream if an annotation dict is provided

            Args:
                video_name: dtype=char, name of the video with which to build tracks on
        '''
        # if specified video not in the possible video names, raise assertion
        assert video_name in self.video_names

        print(f'Video Builder Enabled: Building video for {video_name}:')
        video_length = len(os.listdir(os.path.join(self.video_path, video_name))) 
        progress = tqdm(total=video_length, ncols=25)

        for img_idx in list(range(video_length - 1)):
            progress.update()
            idx = img_idx + 1

            if idx == 5803:
                print("hi")

            # path to the specific frame in the video
            frame_path = os.path.join(self.video_path, video_name, f'{idx + 1:05}.jpg')

            try:
                img = cv2.imread(frame_path)
                h, w, _ = img.shape
            except BaseException as e:
                raise RuntimeError(
                    'Caught "{}" when loading {}'.format(str(e), frame_path)
                )

            frame_streams = [] # list which temporarily stores the frames created by each of the frame builders
            
            # FRAME BUILDERS: they build an annotated frame according to their formatting
            if hasattr(self, 'det_dict'):
                frame_streams.append(
                    build_det_frame(idx, 
                                    img, 
                                    self.det_dict['db'][video_name], 
                                    self.detection_colours, 
                                    self.video_formatting_opts) # builds detection frame
                ) 
                
            if hasattr(self, 'track_dict'):
                img_track = copy.deepcopy(img)
                frame_streams.append(
                    build_track_frame(idx,
                                    img_track, 
                                    self.track_dict['db'][video_name], 
                                    self.track_colours, 
                                    self.video_formatting_opts, 
                                    self.track_opts) # builds track frame
                )
                
            if hasattr(self, 'action_dict'):
                img_action = copy.deepcopy(img)
                frame_streams.append(
                    build_action_frame(idx, 
                                    img_action, 
                                    self.action_dict['db'][video_name], 
                                    self.action_colours, 
                                    self.video_formatting_opts,
                                    self.action_opts) # builds action frame
                )

            if img_idx == 0:
                num_streams = len(frame_streams)
                self.video_w = w * num_streams
                self.video_h = h

            self.video_arr.append(self.combine_frame_streams(frame_streams))

        return self.save_track_video(video_name, self.video_h, self.video_w)  

    def combine_frame_streams(self, frame_streams): # separate function in case a more sophisticated concat of streams is needed
        combined_frame = frame_streams[0]

        for idx, frame in enumerate(frame_streams):
            if idx == 0: continue
            combined_frame = np.concatenate((combined_frame, frame), axis=1)

        return combined_frame

    def save_track_video(self, video_name, h, w):
        out_path = os.path.join(self.save_path, video_name + '_debug.avi')
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'MJPG'), 15, (w, h))

        for i in range(len(self.video_arr)):
            out.write(self.video_arr[i])

        out.release()

        return True