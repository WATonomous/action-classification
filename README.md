# ROAD Dataset Research

Repo for training on the ROAD dataset.

## [Baseline (3D-RetinaNet)](https://github.com/gurkirt/3D-RetinaNet)

### Getting Started

1. Set up the [ROAD Dataset](./scripts/road-data) and [kinetics-pt](./scripts/kinetics-pt)
1. Test the training script
   ```bash
   sbatch scripts/retina-job-test.sh
   ```
1. Start training (configuration can be found in the training script)
   ```bash
   sbatch scripts/retina-job.sh
   ```
