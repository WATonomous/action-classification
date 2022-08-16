import os
import cv2
import json
import copy
import numpy as np
from tqdm import tqdm
from data.road_video_utils import generate_random_colour, get_box

class ROADDebugVideo(object):
    def __init__ (self, opts, opts_OCSORT):
        ''' ROAD Video Dataloader class:
            Reads ROAD image jpgs and produces a video of tracks on them. 

            Args:
                video_path: path to the rgb-images
                save_path: path to where the video should be saved

                compare_detections: create a seperate video stream with detection bboxes beside the tracks 
                detection_path: path to the annotations (should be same as Data.annotation_path)
                
                Misc Cv2 params for bbox layout:
                    bbox_thickness: 2 # pixels thick 

                    # Title Location: based on the following xy plane
                    # o----------------------> X
                    # |     Title(x1 + title_location[0], y1 + title_location[1]) 
                    # |     o----------o
                    # |     | (x1, y1) |
                    # |     |          |
                    # |     o----------o (x2, y2)
                    # |
                    # v    
                    # Y

                    title_location: [0, -10] # pixels from x1, y1
                    font_scale: 0.5
                    font_thickness: 1
        '''
        self.save_path = opts.save_path # directory to save the debug video to
        self.video_path = opts.video_path # path to all the videos
        self.video_names = os.listdir(self.video_path) # directories of all the possible ROAD videos we can use

        self.img_arr = [] # array of images which will build the video
        self.track_colours = {} # dictionary of track colours
        self.bbox_thickness = opts.bbox_thickness 
        self.font_scale = opts.font_scale
        self.font_thickness = opts.font_thickness
        self.title_location = opts.title_location
        self.opts_OCSORT = opts_OCSORT

        if opts.compare_detections:
            with open(opts.detection_path, "r") as f:
                fs = f.read()
                self.ann_dict = json.loads(fs) # ground_truth dictionary
                self.detection_colours = {} # dictionary for detection agent colours

    def build_track_video(self, video_name, tracked_clip_annos):
        ''' Build Track video:
            Builds the ROAD video with tracked boxes for the specified video name. Also builds a separate detections
            video stream if an annotation dict is provided

            Args:
                video_name: dtype=char, name of the video with which to build tracks on
                tracked_clip_annos: 
                    [
                    np.array([[x1, y1, x2, y2, tube_uid, agent_id], ...]), 
                    np.array([[x1, y1, x2, y2, tube_uid, agent_id], ...]), 
                    ...
                    ]
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
                            font, self.font_scale, colour, self.font_thickness, cv2.LINE_AA)

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
                            font, self.font_scale, colour, self.font_thickness, cv2.LINE_AA)        
            
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