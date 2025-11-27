# Build docker images
docker build --pull --rm -f 'docker/analytics/Dockerfile' -t 'deisa_ray-analytics:latest' 'docker/analytics'
docker build --pull --rm -f 'docker/simulation/Dockerfile' -t 'deisa_ray-simulation:latest' 'docker/simulation'

# Export the docker images to a .tar file
docker save deisa_ray-analytics:latest -o docker/images/deisa_ray-analytics.tar
docker save deisa_ray-simulation:latest -o docker/images/deisa_ray-simulation.tar

# Convert the images to singularity images
singularity build docker/images/deisa_ray-analytics.sif docker-archive://docker/images/deisa_ray-analytics.tar
singularity build docker/images/deisa_ray-simulation.sif docker-archive://docker/images/deisa_ray-simulation.tar
