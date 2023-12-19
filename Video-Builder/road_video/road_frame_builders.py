import cv2
import copy
import csv

from road_video.road_video_utils import generate_random_colour, get_box

detection_classes = ['ped', 'veh', 'cyc', 'mob', 'TL']
action_classes = ['Red', 'Amber', 'Green', 'MovAway', 'MovTow', 'Mov', 'Rev', 'Brake',
                  'Stop', 'IncatLft', 'IncatRht', 'HazLit', 'TurLft', 'TurRht', 'MovRht',
                  'MovLft', 'Ovtak', 'Wait2X', 'XingFmLft', 'XingFmRht', 'Xing', 'PushObj']

def build_det_frame(idx, img, dict, colours, video_formatting_opts):
    ''' Build Detection Video: builds video based on the detection annotations

        Args:
            idx: index of frame
            img: the rgb image of the frame
            dict: detection annotation dictionary
            colours: dictionary used to associate colours with the agent class
            video_formatting_opts: formatting params for cv2
    '''
    h, w, _ = img.shape
    img_det = copy.deepcopy(img)

    # misc opencv params
    font = cv2.FONT_HERSHEY_TRIPLEX

    cv2.putText(img_det, 'Detection Visualization', (10, 30), 
                    font, video_formatting_opts.font_header_scale, 
                    video_formatting_opts.font_colour, 
                    video_formatting_opts.font_thickness, cv2.LINE_AA)
    cv2.putText(img_det, 'coloured by agent ID', (10, 50), 
                    font, video_formatting_opts.font_text_scale, 
                    video_formatting_opts.font_colour, 
                    video_formatting_opts.font_thickness, cv2.LINE_AA)
    cv2.putText(img_det, 'transparency is confidence', (10, 70), 
                    font, video_formatting_opts.font_text_scale, 
                    video_formatting_opts.font_colour, 
                    video_formatting_opts.font_thickness, cv2.LINE_AA)

    for anno in dict['frames'][str(idx)]['annos'].values():
        img_det_overlay = copy.deepcopy(img_det)

        # get box
        x1, y1, x2, y2 = get_box(anno['box'], w, h)

        # get colour
        try:
            colour = colours[anno['agent_ids'][0]]
        except KeyError:
            colours[anno['agent_ids'][0]] = generate_random_colour()
            colour = colours[anno['agent_ids'][0]]

        a = 0.25 + 0.75 * anno['score']

        # draw detection boxes with transparency according to score (opaque is high confidence)
        cv2.rectangle(img_det_overlay, (x1, y1), (x2, y2), colour, video_formatting_opts.bbox_thickness)    
        img_det = cv2.addWeighted(img_det_overlay, a, img_det, 1 - a, 0)

        conf = round(anno['score'], 2)
        agent_class = detection_classes[anno['agent_ids'][0]]

        cv2.putText(img_det, f'{agent_class} {conf:02}', (x1 + video_formatting_opts.title_location[0], y1 + video_formatting_opts.title_location[1]), 
                font, video_formatting_opts.font_anno_scale, colour, video_formatting_opts.font_thickness, cv2.LINE_AA)
    
    return img_det


def build_track_frame(idx, img, dict, colours, video_formatting_opts, track_opts):
    ''' Build Trackl Video: builds video based on the track annotations

        Args:
            idx: index of frame
            img: the rgb image of the frame
            dict: track annotation dictionary
            colours: dictionary used to associate colours with the tube ID
            video_formatting_opts: formatting params for cv2
            track_opts: parameters of tracker for these annotations
    '''
    h, w, _ = img.shape
    img_track = copy.deepcopy(img)

    # misc opencv params
    font = cv2.FONT_HERSHEY_TRIPLEX

    cv2.putText(img_track, 'Tracking Visualization', (10, 30), 
                    font, video_formatting_opts.font_header_scale,
                    video_formatting_opts.font_colour,
                    video_formatting_opts.font_thickness, cv2.LINE_AA)
    cv2.putText(img_track, f'det_thresh: {track_opts.det_thresh}', (10, 50), 
                    font, video_formatting_opts.font_text_scale, 
                    video_formatting_opts.font_colour, 
                    video_formatting_opts.font_thickness, cv2.LINE_AA)
    cv2.putText(img_track, f'max_age: {track_opts.max_age}', (10, 70), 
                    font, video_formatting_opts.font_text_scale, 
                    video_formatting_opts.font_colour, 
                    video_formatting_opts.font_thickness, cv2.LINE_AA)
    cv2.putText(img_track, f'min_hits: {track_opts.min_hits}', (10, 90), 
                    font, video_formatting_opts.font_text_scale, 
                    video_formatting_opts.font_colour, 
                    video_formatting_opts.font_thickness, cv2.LINE_AA)
    cv2.putText(img_track, f'iou_threshold: {track_opts.iou_threshold}', (10, 110), 
                    font, video_formatting_opts.font_text_scale, 
                    video_formatting_opts.font_colour, 
                    video_formatting_opts.font_thickness, cv2.LINE_AA)
    
    for anno in dict['frames'][str(idx)]['annos'].values():
        # get box
        x1, y1, x2, y2 = get_box(anno['box'], w, h)

        # get colour
        try:
            colour = colours[anno['tube_uid']]
        except KeyError:
            colours[anno['tube_uid']] = generate_random_colour()
            colour = colours[anno['tube_uid']]

        # draw tracking boxes
        id = int(anno['tube_uid'])
        cv2.rectangle(img_track, (x1, y1), (x2, y2), colour, video_formatting_opts.bbox_thickness)
        cv2.putText(img_track, f'tube_id: {id}', (x1 + video_formatting_opts.title_location[0], y1 + video_formatting_opts.title_location[1]), 
                    font, video_formatting_opts.font_anno_scale, colour, video_formatting_opts.font_thickness, cv2.LINE_AA)   

    return img_track


def build_action_frame(idx, img, dict, colours, video_formatting_opts, action_opts):
    ''' Build Action Video: builds video based on the action annotations

        Args:
            idx: index of frame
            img: the rgb image of the frame
            reader: action annotations csv reader
            colours: dictionary used to associate colours with the action class
            video_formatting_opts: formatting params for cv2
    '''
    h, w, _ = img.shape
    img_action = copy.deepcopy(img)

    # misc opencv params
    font = cv2.FONT_HERSHEY_TRIPLEX

    cv2.putText(img_action, 'Action Visualization', (10, 30), 
                    font, video_formatting_opts.font_header_scale,
                    video_formatting_opts.font_colour,
                    video_formatting_opts.font_thickness, cv2.LINE_AA)
    cv2.putText(img_action, 'coloured by action ID', (10, 50), 
                    font, video_formatting_opts.font_text_scale, 
                    video_formatting_opts.font_colour, 
                    video_formatting_opts.font_thickness, cv2.LINE_AA)
    cv2.putText(img_action, f'action_thresh: {action_opts.action_thresh}', (10, 70), 
                    font, video_formatting_opts.font_text_scale, 
                    video_formatting_opts.font_colour, 
                    video_formatting_opts.font_thickness, cv2.LINE_AA)

    for anno in dict['frames'][str(idx)]['annos'].values():
        # get box
        x1, y1, x2, y2 = get_box(anno['box'], w, h)
        
        # get action scores, filter out to show scores over a threshold
        try:
            action_scores = anno['action_scores']
        except KeyError:
            return img_action

        classes = 0
        max_score = 0
        for cls, score in action_scores.items():
            if score >= action_opts.action_thresh:
                classes += 1
                # get colour
                try:
                    colour = colours[cls]
                except KeyError:
                    colours[cls] = generate_random_colour()
                    colour = colours[cls]
                
                # draw tracking boxes
                action = action_classes[int(cls) - 1]
                y_offset = video_formatting_opts.title_location[1] * classes
                cv2.putText(img_action, f'{action}', (x1 + video_formatting_opts.title_location[0], y1 + y_offset), 
                            font, video_formatting_opts.font_anno_scale, colour, video_formatting_opts.font_thickness, cv2.LINE_AA)

                if score > max_score:
                    max_score = score
                    cv2.rectangle(img_action, (x1, y1), (x2, y2), colour, video_formatting_opts.bbox_thickness)
                
    return img_action