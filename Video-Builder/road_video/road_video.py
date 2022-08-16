import os
import cv2
import json
import csv
import copy
import numpy as np
from tqdm import tqdm
from road_video.road_video_utils import generate_random_colour, get_box

class ROADDebugVideo(object):
    def __init__ (self, opts):
        ''' ROAD Video Dataloader class:
            Reads ROAD image jpgs and produces a video of tracks on them. 

            Args:
                opts: check config file for explanations of each of the parameters
        '''
        # Load config opts into member variables
        self.load_opts(opts)

        # directories of all the possible ROAD videos we can use
        self.video_names = os.listdir(self.video_path)
        if opts.list_videos:
            print("Available Videos to Debug with:")
            for name in self.video_names:
                print(name)

        # array of images which will build the video
        self.img_arr = [] 
    
    def load_opts(self, opts):
        ''' Load Opts: Loads config parameters into member variables. Creates annotation dictionaries if provided an annotation path
        '''
        # Load general params that format the video
        video_opts = opts.Video_Builder
        self.save_path = video_opts.save_path # directory to save the debug video to
        self.video_path = video_opts.video_path # path to all the videos

        # Load dictionaries and readers for all the available annotations
        self.num_streams = 0
        if opts.Detector.detections_path is not None:
            self.num_streams += 1
            with open(opts.Detector.detections_pat, "r") as f:
                fs = f.read()
                self.det_dict = json.loads(fs) # detections dictionary
                self.detection_colours = {} # dictionary for detection colours, coloured by agent class

        if opts.Tracker.tracks_path is not None:
            self.num_streams += 1
            with open(opts.Tracker.tracks_path, "r") as f:
                fs = f.read()
                self.track_dict = json.loads(fs) # tracks dictionary
                self.track_colours = {} # dictionary for track colours, coloured by track id

        if opts.Action_Classifier.actions_path is not None:
            self.num_streams += 1
            with open(opts.Action_Classifier.actions_path, "r") as f:
                self.action_reader = csv.reader(f) # actions csv reader 
                self.action_colours = {} # dictionary for action colours, coloured by action class
        
        # Calculate and Load cv2 parameters based on the number of video streams
        video_formatting_opts = opts.Video_Formatting
        self.bbox_thickness = video_formatting_opts.bbox_thickness 
        self.title_location = video_formatting_opts.title_location

        self.font_thickness = video_formatting_opts.font_thickness
        self.font_colour = 
        self.font_anno_scale = video_formatting_opts.font_scale
        self.font_header_scale = 
        self.font_text_scale = 

    # TODO the builder needs to change, create a util function that creates titles accordingly, and also add boxes to an image
    # provided we give the proper formatting

    def build_track_video(self, video_name):
        ''' Build Track video:
            Builds the ROAD video with tracked boxes for the specified video name. Also builds a separate detections
            video stream if an annotation dict is provided

            Args:
                video_name: dtype=char, name of the video with which to build tracks on
        '''
        # if specified video not in the possible video names, raise assertion
        assert video_name in self.video_names

        print(f'Video Builder Enabled: Building track video for {video_name}:')
        progress = tqdm(total=len(tracked_clip_annos), ncols=25)

        for idx, tracked_frame_annos in enumerate(tracked_clip_annos):
            # progress
            progress.update()

            # path to the specific frame in the video
            frame_path = os.path.join(self.video_path, video_name, f'{idx + 1:05}.jpg')

            # misc opencv params
            font = cv2.FONT_HERSHEY_TRIPLEX

            try:
                img = cv2.imread(frame_path)
                h, w, _ = img.shape
            except BaseException as e:
                raise RuntimeError(
                    'Caught "{}" when loading {}'.format(str(e), frame_path)
                )
            
            if hasattr(self, 'ann_dict'): # if annodation dict exists, create another image with only detections
                img_det = copy.deepcopy(img)

                cv2.putText(img_det, 'Detection Visualization', (10, 30), 
                            font, 1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(img_det, 'coloured by agent ID', (10, 50), 
                            font, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(img_det, 'transparency is confidence', (10, 70), 
                            font, 0.7, (0, 255, 0), 1, cv2.LINE_AA)

                for anno in self.ann_dict['db'][video_name]['frames'][str(idx + 1)]['annos'].values():
                    img_det_overlay = copy.deepcopy(img_det)

                    # get box
                    x1, y1, x2, y2 = get_box(anno['box'], w, h)

                    # get colour
                    try:
                        colour = self.detection_colours[anno['agent_ids'][0]]
                    except KeyError:
                        self.detection_colours[anno['agent_ids'][0]] = generate_random_colour()
                        colour = self.detection_colours[anno['agent_ids'][0]]

                    a = 0.25 + 0.75 * anno['score']

                    # draw detection boxes with transparency according to score (opaque is high confidence)
                    cv2.rectangle(img_det_overlay, (x1, y1), (x2, y2), colour, self.bbox_thickness)    
                    img_det = cv2.addWeighted(img_det_overlay, a, img_det, 1 - a, 0)

                    id = anno['agent_ids']
                    cv2.putText(img_det, f'{id}', (x1 + self.title_location[0], y1 + self.title_location[1]), 
                            font, self.font_anno_scale, colour, self.font_thickness, cv2.LINE_AA)

            cv2.putText(img, 'Tracking Visualization', (10, 30), 
                            font, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(img, f'det_thresh: {self.opts_OCSORT.det_thresh}', (10, 50), 
                            font, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(img, f'max_age: {self.opts_OCSORT.max_age}', (10, 70), 
                            font, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(img, f'min_hits: {self.opts_OCSORT.min_hits}', (10, 90), 
                            font, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(img, f'iou_threshold: {self.opts_OCSORT.iou_threshold}', (10, 110), 
                            font, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
            
            for anno in tracked_frame_annos:
                # get box
                x1, y1, x2, y2 = get_box(anno[:4], w, h)

                # get colour
                try:
                    colour = self.track_colours[anno[4]]
                except KeyError:
                    self.track_colours[anno[4]] = generate_random_colour()
                    colour = self.track_colours[anno[4]]

                # draw tracking boxes
                cv2.rectangle(img, (x1, y1), (x2, y2), colour, self.bbox_thickness)
                cv2.putText(img, f'tube_id: {int(anno[4])}', (x1 + self.title_location[0], y1 + self.title_location[1]), 
                            font, self.font_anno_scale, colour, self.font_thickness, cv2.LINE_AA)        
            
            if hasattr(self, 'ann_dict'): # combine detection and tracking images beside eachother
                combined_img = np.zeros((h, w * 2, 3), np.uint8)

                combined_img[:h, :w,:3] = img
                combined_img[:h, w:w*2,:3] = img_det    

                self.img_arr.append(combined_img) 
                continue  
                
            self.img_arr.append(img)
        
        if hasattr(self, 'ann_dict'): w = w * 2

        return self.save_track_video(video_name, h, w)  

        

    def save_track_video(self, video_name, h, w):
        out_path = os.path.join(self.save_path, video_name + '_debug.avi')
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'MJPG'), 15, (w, h))

        for i in range(len(self.img_arr)):
            out.write(self.img_arr[i])

        out.release()

        return True