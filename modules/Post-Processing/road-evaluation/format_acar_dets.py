import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm


def format_acar_dets(args):
    """Uses a path to an acar prediction csv to write .pkl files in the
    format that the evaluation pipeline expects. 

    MAGIC NUMBERS 512 and 682:
    RetinaNet input dimensions are 512x682. Literally of the steps of the evaluations
    have these dimensions hardcoded into them. The most minimal change for us to evaluate properly
    is to multiply by these magic numbers ourselves in this function. 

    Parameters
    ----------
    args : argparse object
        contains all of the arguments read and initialized in main.py
    """
    # the training data has 22 classes, but the evaluation only considers 19 classes
    # given a class index form acar, we want to map it to the correct index with respect
    # to the evaluation. This array has length 22.
    action_class_selection_map = [0, 1, 2, 3, 4, 5, -1, 6, 7, 8, 9, 10, 11, 12, -1, -1, 13, 14, 15, 16, 17, 18]

    # read predictions csv into dataframe. predict_epoch_{}.csv has no headers.
    # columns are {video, frame, xmin, ymin, xmax, ymax, action, confidence} in that order
    predictions_df = pd.read_csv(args.PRED_CSV, header = None)
    predictions_df.columns = ["video", "frame", "xmin", "ymin", "xmax", "ymax", "action", "confidence"]
    formatted_predictions = {}
    print(f"Formatting predictions from {args.PRED_CSV}")
    for _, row in tqdm(predictions_df.iterrows()):
        box_hash = f"{row['xmin']}_{row['ymin']}_{row['xmax']}_{row['ymax']}"
        if row['video'] not in formatted_predictions:
            formatted_predictions[row['video']] = {}
        if row['frame'] not in formatted_predictions[row['video']]:
            formatted_predictions[row['video']][row['frame']] = {}
        if box_hash not in formatted_predictions[row['video']][row['frame']]:
            formatted_predictions[row['video']][row['frame']][box_hash] = {
                # Magic number time
                'bbox': [float(row['xmin'])*682, float(row['ymin'])*512, float(row['xmax'])*682, float(row['ymax'])*512],
                'confidences': [0]*19
            }
        # action class is 1 indexed, so subtract 1
        action_idx = action_class_selection_map[int(row['action'])-1]
        if action_idx < 0:
            continue
        formatted_predictions[row['video']][row['frame']][box_hash]['confidences'][action_idx] = float(row['confidence'])


    if not os.path.exists(args.ACAR_DET_SAVE_DIR):
        os.makedirs(args.ACAR_DET_SAVE_DIR)
    print(f"Writing .pkl files to {args.ACAR_DET_SAVE_DIR}")
    for video_name, video in tqdm(formatted_predictions.items()):
        for frame_num, annos in video.items():
            save_data = {'main': np.zeros((len(annos), 23))}
            for i, anno in enumerate(annos.values()):
                save_data['main'][i][0:4] = np.array(anno['bbox'])
                save_data['main'][i][4:23] = np.array(anno['confidences'])
            if not os.path.exists(os.path.join(args.ACAR_DET_SAVE_DIR, video_name)):
                os.makedirs(os.path.join(args.ACAR_DET_SAVE_DIR, video_name))
            with open(f"{args.ACAR_DET_SAVE_DIR}/{video_name}/%05d.pkl" % frame_num, 'wb') as f:
                pickle.dump(save_data, f)
    
    print("Done!")

