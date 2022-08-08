import os
import cv2
from road_video_utils import generate_random_colour

class ROADDebugVideo(object):
    def __init__ (self, opts):
        ''' ROAD Video Dataloader class:
            Reads ROAD image jpgs and produces a video of tracks on them. 

            Args:
                video_path: path to the annotations
        '''
        self.save_path = opts.save_path # directory to save the debug video to
        self.video_path = opts.video_path # path to all the videos
        self.video_names = os.listdir(self.video_path) # directories of all the possible ROAD videos we can use

        self.img_arr = [] # array of images which will build the video
        self.track_colours = {} # dictionary of track colours
        self.bbox_thickness = opts.bbox_thickness 

    def build_track_video(self, video_name, tracked_clip_annos):
        ''' Build Track video:
            Builds the ROAD video with tracked boxes for the specified video name.

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

        for idx, tracked_frame_annos in enumerate(tracked_clip_annos):
            # path to the specific frame in the video
            frame_path = os.path.join(self.video_path, video_name, f'{idx:05}.jpg')

            try:
                img = cv2.imread(frame_path)
                h, w, _ = img.shape
            except BaseException as e:
                raise RuntimeError(
                    'Caught "{}" when loading {}'.format(str(e), frame_path)
                )

            for anno in tracked_frame_annos:
                # get box
                xyxy = anno[:4] # [x1, y1, x2, y2] where x1y1=top-left, x2y2=bottom-right
                x1 = xyxy[0] * w
                x2 = xyxy[2] * w
                y1 = xyxy[1] * h
                y2 = xyxy[3] * h

                # get colour
                try:
                    colour = self.track_colours[anno[4]]
                except KeyError:
                    self.track_colours[anno[4]] = generate_random_colour()
                    colour = self.track_colours[anno[4]]

                # misc opencv params
                font = cv2.FONT_HERSHEY_TRIPLEX
                font_scale = 1

                cv2.rectangle(img, (x1, y1), (x2, y2), colour, self.bbox_thickness)
                cv2.putText(img, f'tube_id: {anno[4]}', (x2 + 10, y2), 
                            font, font_scale, colour, self.bbox_thickness, cv2.LINE_AA)
                
            self.img_arr.append(img)

        return self.save_track_video(video_name, h, w)      

    def save_track_video(self, video_name, h, w):
        out_path = os.path.join(self.save_path, video_name + '_debug.mp4')
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'MP4V'), 15, (h, w))

        for i in range(len(self.img_arr)):
            out.write(self.img_arr[i])

        out.release()

        return True