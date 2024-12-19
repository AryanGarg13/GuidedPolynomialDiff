#!/bin/bash

# Make the script executable with `chmod +x create_video.sh`
# Run the script with `./create_video.sh`

# Convert images to video using ffmpeg
ffmpeg -framerate 24 -i ../../images2/step_%04d.png -c:v libx264 -pix_fmt yuv420p output_video.mp4
