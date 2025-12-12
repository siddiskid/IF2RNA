#!/bin/bash
# Download real GeoMx data using curl (macOS compatible)

set -e

BASE_DIR="/Users/siddarthchilukuri/Documents/GitHub/IF2RNA/data/geomx_datasets"
mkdir -p "$BASE_DIR"

echo "🚀 Downloading Real GeoMx Data for IF2RNA"
echo "=========================================="
echo ""

# Start with GSE289483 - Pulmonary Cancer (has DCC, PKC, CSV)
DATASET="GSE289483"
echo "📥 Downloading $DATASET - Pulmonary Pleomorphic Carcinoma"
echo "-----------------------------------------------------------"

DATASET_DIR="$BASE_DIR/$DATASET"
mkdir -p "$DATASET_DIR"
cd "$DATASET_DIR"

# GEO FTP base URL
FTP_BASE="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE289nnn/GSE289483/suppl"

echo "Fetching file list from GEO..."
echo ""

# Download using curl
echo "Downloading supplementary files..."
echo ""

# Try to download common file patterns
for pattern in "*_RAW.tar" "*.dcc.gz" "*.pkc.gz" "*.xlsx" "*.csv.gz" "*.txt.gz"; do
    echo "Trying pattern: $pattern"
    curl -s -L -O "$FTP_BASE/$pattern" 2>/dev/null || true
done

# List what we got
echo ""
echo "Files downloaded:"
ls -lh 2>/dev/null || echo "No files found yet"

# Decompress any .gz files
if ls *.gz 1> /dev/null 2>&1; then
    echo ""
    echo "Decompressing files..."
    gunzip -f *.gz
fi

# Extract any tar files
if ls *.tar 1> /dev/null 2>&1; then
    echo ""
    echo "Extracting tar archives..."
    tar -xf *.tar
fi

echo ""
echo "✅ Download attempt complete for $DATASET"
echo "Final files:"
ls -lh

echo ""
echo "========================================="
echo "Download complete!"
echo "Location: $DATASET_DIR"
echo ""
echo "Next: Check if files were downloaded successfully"
