# Road Dataset

This folder contains the scripts to pull the [ROAD Dataset](https://github.com/gurkirt/road-dataset). Full instructions can be found in the original repo. The simplified instructions are below.

## Getting Started

```bash
# download videos and annotations
./download_videos.sh <download_dir>
# extract videos into frames for training
python extract_videos2jpgs.py <download_dir>
```
