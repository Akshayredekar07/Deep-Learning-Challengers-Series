#!/bin/bash

# Initialize a counter
counter=4

# Loop through the files matching "image copy *.png"
for file in image\ copy\ *.png; do
  # Construct the new filename
  new_file="image${counter}.png"

  # Rename the file
  mv "$file" "$new_file"

  # Increment the counter
  counter=$((counter + 1))
done