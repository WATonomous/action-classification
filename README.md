# ROAD Dataset Research

Dockerized Research Repository for Action Classification.

### [Baseline (3D-RetinaNet)](https://github.com/gurkirt/3D-RetinaNet)

## Getting Started

1. `docker compose build`
2. `docker compose up <service>`, where the services include our action classifier, tracker, video builder, post-processing techniques, and parts of our evaluation scheme.
3. `docker exec -it <service> /bin/bash` to begin developing in the container

## Repo Structure
Most components are dockerized, we transfer data between each of the components via JSON files and CSVs.

For submodules, please refer to their setup instructions.

**This repository is no longer maintained,** but it will be kept available for those who are interested in doing future research in Action Classifcation.