#!/bin/bash

# Variables
IMAGE_DIR="./images_final_trajectory"         # Directory containing the images
OUTPUT_GIF="output.gif"      # Name of the output GIF file
FRAME_RATE=10                # Frames per second

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null
then
    echo "ffmpeg could not be found. Please install ffmpeg to use this script."
    exit
fi

# Check if image directory exists
if [ ! -d "$IMAGE_DIR" ]; then
    echo "Image directory $IMAGE_DIR does not exist."
    exit
fi

# Create the GIF from the images
echo "Converting images from $IMAGE_DIR to $OUTPUT_GIF at $FRAME_RATE fps..."
ffmpeg -framerate $FRAME_RATE -pattern_type glob -i "$IMAGE_DIR/*.png" -vf "scale=640:-1:flags=lanczos,palettegen" palette.png
ffmpeg -framerate $FRAME_RATE -pattern_type glob -i "$IMAGE_DIR/*.png" -i palette.png -filter_complex "scale=640:-1:flags=lanczos[x];[x][1:v]paletteuse" "$OUTPUT_GIF"

# Clean up the temporary palette file
rm -f palette.png

# Check if the GIF was created successfully
if [ -f "$OUTPUT_GIF" ]; then
    echo "GIF created successfully: $OUTPUT_GIF"
else
    echo "An error occurred while creating the GIF."
fi

