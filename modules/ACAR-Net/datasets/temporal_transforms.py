"""
Adapted code from:
    @inproceedings{hara3dcnns,
      author={Kensho Hara and Hirokatsu Kataoka and Yutaka Satoh},
      title={Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?},
      booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      pages={6546--6555},
      year={2018},
    }.
"""

import random


class TemporalSampling(object):
    """Temporally sample the given frame indices with a given stride.

    Args:
        step (int, optional): Stride for sampling.
    """

    def __init__(self, step=1):
        self.step = step

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        return frame_indices[::self.step]

    def __repr__(self):
        return '{self.__class__.__name__}(step={self.step})'.format(self=self)


class TemporalCenterCrop(object):
    """Temporally crop the given frame indices at the center.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
        step (int, optional): Stride when taking the crop.
    """

    def __init__(self, size, step=1):
        self.size = size
        self.step = step

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        center_index = len(frame_indices) // 2
        begin_index = max(0, center_index - (self.size // 2))
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out[::self.step]

    def __repr__(self):
        return '{self.__class__.__name__}(size={self.size}, step={self.step})'.format(self=self)


class TemporalCenterRetentionCrop(object):
    """Temporally crop the given frame indices at the center.
    Retains the value of the center frame index passed in.

    The number of frames coming in must be greater than the desired size

    Args:
        size (int): Desired output size of the crop.
        step (int, optional): Stride when taking the crop.
    """

    def __init__(self, size, step=1):
        self.size = size
        self.step = step

    def __call__(self, frame_indices, center_frame):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
            center_frame (int): frame which will be retained 
                (finds its index in frame_indices and crops around it)
        Returns:
            list: Cropped frame indices.
        """

        if len(frame_indices) < self.size:
            raise Exception("The length of frame_indices is less than the desired size: {self.size}.".format(self=self))

        center_index = frame_indices.index(center_frame)
        begin_index = center_index - self.size // 2
        end_index = center_index + self.size // 2

        if begin_index < 0 or end_index >= len(frame_indices):
            raise Exception("The chosen center frame cannot produce a full temporal crop of size: \
                {self.size}. \n Center_Index: {center_index} \n Length of Frame_indices: {frame_indices} \
                    \n Begin and End Index: ({begin}, {end})".format(
                    self=self, 
                    center_index=center_index, 
                    frame_indices=len(frame_indices), 
                    begin=begin_index,
                    end=end_index
                    ))

        # crops frames in steps right of the center_index, starting at the center_index
        out_right = frame_indices[center_index : end_index + 1 : self.step][1:]
        # crops frames in steps left of the center_index, starting at the center_index
        out_left = frame_indices[center_index : begin_index : -1][::self.step][::-1]

        return out_left + out_right

    def __repr__(self):
        return '{self.__class__.__name__}(size={self.size}, step={self.step})'.format(self=self)


class TemporalRandomCrop(object):
    """Temporally crop the given frame indices at a random location.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
        step (int, optional): Stride when taking the crop.
    """

    def __init__(self, size, step=1):
        self.size = size
        self.step = step

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        rand_end = max(0, len(frame_indices) - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out[::self.step]

    def __repr__(self):
        return '{self.__class__.__name__}(size={self.size}, step={self.step})'.format(self=self)
