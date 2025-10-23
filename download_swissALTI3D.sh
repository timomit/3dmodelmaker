#!/bin/bash

# Usage: ./download_links.sh links.csv downloads_folder

CSV_FILE="$1"
OUT_DIR="$2"

if [[ -z "$CSV_FILE" || -z "$OUT_DIR" ]]; then
  echo "Usage: $0 <csv_file> <output_dir>"
  exit 1
fi

mkdir -p "$OUT_DIR"

# Loop through each URL in the CSV, assuming links are in column 1
while IFS=, read -r link rest || [[ -n "$link" ]]; do
  wget -P "$OUT_DIR" "$link"
done < "$CSV_FILE"
