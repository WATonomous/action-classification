import json
import numpy as np
import pickle
import os
import pandas as pd
from tqdm import tqdm


def format_acar_dets(opts):
    """Uses a path to an acar prediction csv to write .pkl files in the
    format that the evaluation pipeline expects. 

    MAGIC NUMBERS 512 and 682:
    RetinaNet input dimensions are 512x682. Literally of the steps of the evaluations
    have these dimensions hardcoded into them. The most minimal change for us to evaluate properly
    is to multiply by these magic numbers ourselves in this function. 

    Parameters
    ----------
    opts : argparse object
        contains all of the arguments read and initialized in main.py
    """
    # the training data has 22 classes, but the evaluation only considers 19 classes
    # given a class index form acar, we want to map it to the correct index with respect
    # to the evaluation. This array has length 22.
    action_class_selection_map = [0, 1, 2, 3, 4, 5, -1, 6, 7, 8, 9, 10, 11, 12, -1, -1, 13, 14, 15, 16, 17, 18]

    # # read predictions csv into dataframe. predict_epoch_{}.csv has no headers.
    # # columns are {video, frame, xmin, ymin, xmax, ymax, action, confidence} in that order
    # predictions_df = pd.read_csv(opts.prediction_path, header = None)
    # predictions_df.columns = ["video", "frame", "xmin", "ymin", "xmax", "ymax", "action", "confidence", "tube_id"]
    # formatted_predictions = {}
    # print(f"Formatting predictions from {opts.prediction_path}")
    # for _, row in tqdm(predictions_df.iterrows()):
    #     box_hash = f"{row['xmin']}_{row['ymin']}_{row['xmax']}_{row['ymax']}"
    #     if row['video'] not in formatted_predictions:
    #         formatted_predictions[row['video']] = {}
    #     if row['frame'] not in formatted_predictions[row['video']]:
    #         formatted_predictions[row['video']][row['frame']] = {}
    #     if box_hash not in formatted_predictions[row['video']][row['frame']]:
    #         formatted_predictions[row['video']][row['frame']][box_hash] = {
    #             # Magic number time
    #             'bbox': [float(row['xmin'])*682, float(row['ymin'])*512, float(row['xmax'])*682, float(row['ymax'])*512],
    #             'confidences': [0]*19
    #         }
    #     # action class is 1 indexed, so subtract 1
    #     action_idx = action_class_selection_map[int(row['action'])-1]
    #     if action_idx < 0:
    #         continue
    #     formatted_predictions[row['video']][row['frame']][box_hash]['confidences'][action_idx] = float(row['confidence'])

    with open(opts.prediction_path, "r") as f:
        fs = f.read()
        ann_dict = json.loads(fs)

    if not os.path.exists(opts.save_pickles_path):
        os.makedirs(opts.save_pickles_path)

    print(f"Writing .pkl files to {opts.save_pickles_path}")

    for video_name, video in ann_dict['db'].items():
        print(video_name)
        for frame_num, frame_data in tqdm(video['frames'].items()):
            if not frame_data['annotated']: continue

            annos = frame_data['annos']
            
            if int(frame_num) < 50 or int(frame_num) > 5940: continue

            save_data = {'main': np.zeros((len(annos), 23))}
            for i, anno in enumerate(annos.values()):
                # bbox in terms of proper road image sizing
                bbox = anno['box']
                bbox[0] = bbox[0] * 682
                bbox[1] = bbox[1] * 512
                bbox[2] = bbox[2] * 682
                bbox[3] = bbox[3] * 512

                save_data['main'][i][0:4] = np.array(bbox)

                # action scores in terms of what road eval is concerned with
                confidences = np.zeros(19)
                for idx, score in enumerate(anno['action_scores'].values()):
                    if action_class_selection_map[idx] == -1:
                        continue

                    confidences[action_class_selection_map[idx]] = score

                save_data['main'][i][4:23] = confidences

            if not os.path.exists(os.path.join(opts.save_pickles_path, video_name)):
                os.makedirs(os.path.join(opts.save_pickles_path, video_name))
            with open(f"{opts.save_pickles_path}/{video_name}/%05d.pkl" % int(frame_num), 'wb') as f:
                pickle.dump(save_data, f)
    
    print("Done!")

