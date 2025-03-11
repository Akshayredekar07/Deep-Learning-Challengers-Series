#!/bin/bash
n=52  # Starting number for renaming
for file in $(ls image-*.png | sort -V); do
    # Generate the new filename
    new_file=$(printf "image-%d.png" "$n")
    mv "$file" "$new_file"
    ((n++))  # Increment the counter
done

echo "Renaming completed!"
