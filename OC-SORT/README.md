# OCSORT Tracker for ROAD Data
Associates detections through a 'more complex' Kalmann Filter.

Link to OCSORT paper: https://arxiv.org/pdf/2203.14360.pdf

Link to OCSORT repo: https://github.com/noahcao/OC_SORT

##  Tracker Evaluation:
These are what the metrics mean:
Name|Description
:---|:---
IDS|Total number of track switches.
MOTA|Multiple object tracker accuracy.
Prcn| Precision, Number of detected objects over sum of detected and false positives.
Rcll| Recall, Number of detections over number of objects.
IDP|ID measures: global min-cost precision.
IDR|ID measures: global min-cost recall.
IDF1|ID measures: global min-cost F1 score.


### Tracking results based on ground truth detections:
| Videos | MOTA | IDS |  IDP |  IDR | IDF1 |  Prcn | Rcll |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |  ------------- |
| 2014-06-26-09-53-12_stereo_centre_02 | 95.2% |  17 | 97.2% | 92.6% | 94.9% | 100.0% | 95.3% |
| 2014-06-25-16-45-34_stereo_centre_02 | 94.2% |  25 | 96.2% | 90.7% | 93.4% | 100.0% | 94.3% |
| 2014-07-14-14-49-50_stereo_centre_01 | 93.3% |  64 | 90.9% | 85.0% | 87.9% | 100.0% | 93.5% |
| 2014-08-08-13-15-11_stereo_centre_01 | 91.3% | 305 | 92.0% | 84.7% | 88.2% | 100.0% | 92.1% |
| 2014-07-14-15-42-55_stereo_centre_03 | 92.3% |  39 | 94.5% | 87.3% | 90.7% | 100.0% | 92.4% |
| 2014-08-11-10-59-18_stereo_centre_02 | 95.3% |   1 | 98.8% | 94.1% | 96.4% | 100.0% | 95.3% |
| 2014-11-14-16-34-33_stereo_centre_06 | 97.0% |  31 | 92.9% | 90.3% | 91.6% | 100.0% | 97.1% |
| 2014-11-18-13-20-12_stereo_centre_05 | 94.8% | 128 | 92.8% | 88.3% | 90.5% | 100.0% | 95.1% |
| 2014-11-21-16-07-03_stereo_centre_01 | 92.8% | 272 | 85.2% | 79.8% | 82.4% | 100.0% | 93.7% |
| 2014-11-25-09-18-32_stereo_centre_04 | 95.1% |  11 | 96.8% | 92.1% | 94.4% | 100.0% | 95.1% |
| 2014-12-09-13-21-02_stereo_centre_01 | 92.9% | 168 | 85.7% | 80.1% | 82.8% | 100.0% | 93.4% |
| 2015-02-03-08-45-10_stereo_centre_02 | 92.9% | 100 | 97.0% | 90.4% | 93.6% | 100.0% | 93.2% |
| 2015-02-03-19-43-11_stereo_centre_04 | 97.0% |   0 | 96.5% | 93.6% | 95.0% | 100.0% | 97.0% |
| 2015-02-06-13-57-16_stereo_centre_02 | 94.6% |  26 | 98.2% | 93.0% | 95.5% | 100.0% | 94.7% |
| 2015-02-13-09-16-26_stereo_centre_02 | 95.1% |   6 | 97.9% | 93.1% | 95.4% | 100.0% | 95.1% |
| 2015-02-13-09-16-26_stereo_centre_05 | 96.6% |  27 | 96.2% | 93.0% | 94.6% | 100.0% | 96.7% |
| 2015-02-24-12-32-19_stereo_centre_04 | 94.4% |  21 | 96.6% | 91.3% | 93.8% | 100.0% | 94.5% |
| 2015-03-03-11-31-36_stereo_centre_01 | 91.2% | 214 | 83.9% | 77.2% | 80.4% | 100.0% | 92.1% |
OVERALL | 94.0%  |1455 | 93.2% | 87.9% | 90.4% | 100.0% | 94.3% |

## Getting Started
Read up on the comments made in `config.yaml`.

The current `config.yaml` is setup to load the json annotations `/road/val1_detections_inactive_merged_gt_format_score_0.1_new.jsonl` within the ocsort docker container. Please change the config as you please. The annos must be in the structure of the ROAD annotations.

To run the code:
`python3 main.py --config configs/config.yaml`
