import numpy as np

def xyxy2xywh(x):
        # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
        if x.ndim == 1:
            y = np.copy(x)
            y[0] = (x[0] + x[2]) / 2  # x center
            y[1] = (x[1] + x[3]) / 2  # y center
            y[2] = x[2] - x[0]  # width
            y[3] = x[3] - x[1]  # height

            return y
            
        y = np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height

        return y

class UIDtoNumber(object):
    def __init__(self):
        self.uid_dict = {}
        self.n_uids = 0

    def uid2number(self, uid):
        try:
            return self.uid_dict[uid]
        except KeyError:
            self.n_uids += 1
            self.uid_dict[uid] = self.n_uids

            return self.uid_dict[uid]