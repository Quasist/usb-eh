#!/bin/bash
extract_frames() {
    if [ "$1" == '' ] || [ "$2" == '' ] || [ "$3" == '' ] || [ "$4" == '']; then
        echo "Usage: $0 <input folder> <output folder> <file extension> <output extension>";
        exit;
    fi
    for file in "$1"/*/*."$3"; do
        destination="$2${file:${#1}:${#file}-${#1}-${#3}-1}";
        mkdir -p "$destination";
        ffmpeg -i "$file" -vf scale=-1:720 -qscale:v 4 "$destination/image-%03d.$4";
    done
}

extract_frames data frames mp4 jpeg
