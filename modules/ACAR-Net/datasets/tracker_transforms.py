import random
import numpy as np

class RandomBBoxDrop(object):
    def __init__(self, drop_rate):
        self.drop_rate = drop_rate

    def __call__(self, clip_labels):
        """ Random BBox Drop: Drops random bboxes in the clip to emulate an unreliable tracker.
            This should make the model more generalizable. BBoxes are cropped out 
            by pass by assignment. (changing the contents of clip_labels here will change the 
            contents of the original clip labels so long as its not reassigned)

            Parameters
            ----------
            clip_labels: list(list(dict))
            [
             [{
                'tube_uid': char of unique ID,
                'bounding_box': float([x1, y1, x2, y2]) norm from 0-1,
                'bbox_id': char of unique bbox ID,
                'label': int of detected agent class 
             }, ...],
             ...
            ]
        """

        for i, f_labels in enumerate(clip_labels):
            if i == (len(clip_labels) // 2 - 1):
                continue

            f_labels_len = len(f_labels)
            j = 0
            while j < f_labels_len:
                if random.random() < self.drop_rate:
                    try:
                        del clip_labels[i][j] 
                        f_labels_len -= 1
                    except IndexError:
                        raise RuntimeError(f"i{i} and j{j} for clip length: {len(clip_labels)} and label length:{len(f_labels)}")

                j += 1

