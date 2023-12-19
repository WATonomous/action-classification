# ROAD Dataset Research

Repository for Action Classification.

### [Baseline (3D-RetinaNet)](https://github.com/gurkirt/3D-RetinaNet)

## Getting Started

1. `docker compose build`
2. `docker compose up <service>`, where the services include our action classifier, tracker, video builder, and parts of our evaluation scheme.
3. `docker exec -it <service> /bin/bash` to begin developing in the container

## Repo Structure
Each component is dockerized, we transfer data between each of the components via JSON files and CSVs.