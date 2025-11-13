#!/bin/bash
# Download GeoMx datasets for IF2RNA project

set -e  # Exit on error

# Base directory
BASE_DIR="/Users/siddarthchilukuri/Documents/GitHub/IF2RNA/data/geomx_datasets"
mkdir -p "$BASE_DIR"

echo "🚀 IF2RNA GeoMx Data Downloader"
echo "================================"
echo ""

# Function to download dataset
download_dataset() {
    local GSE_ID=$1
    local DESCRIPTION=$2
    
    echo "📥 Downloading $GSE_ID - $DESCRIPTION"
    echo "-----------------------------------"
    
    # Create dataset directory
    DATASET_DIR="$BASE_DIR/$GSE_ID"
    mkdir -p "$DATASET_DIR"
    cd "$DATASET_DIR"
    
    # Construct FTP path
    GSE_PREFIX="${GSE_ID:0:${#GSE_ID}-3}nnn"
    FTP_BASE="ftp://ftp.ncbi.nlm.nih.gov/geo/series/$GSE_PREFIX/$GSE_ID/suppl"
    
    echo "FTP URL: $FTP_BASE"
    
    # Download supplementary files
    echo "Downloading supplementary files..."
    wget -r -np -nd -A "*.dcc.gz,*.pkc.gz,*.csv.gz,*.xlsx,*.txt.gz" "$FTP_BASE/" 2>&1 | grep -E "^(--.*|Saving to:|Downloaded:)" || true
    
    # Decompress gz files
    if ls *.gz 1> /dev/null 2>&1; then
        echo "Decompressing files..."
        gunzip -f *.gz
    fi
    
    # Summary
    echo "✅ Downloaded to: $DATASET_DIR"
    echo "Files:"
    ls -lh
    echo ""
}

# Check if wget is installed
if ! command -v wget &> /dev/null; then
    echo "❌ Error: wget is not installed"
    echo "Install with: brew install wget"
    exit 1
fi

echo "Starting downloads..."
echo ""

# Download datasets in order of priority

# 1. GSE289483 - Pulmonary Cancer (125 samples)
download_dataset "GSE289483" "Pulmonary Pleomorphic Carcinoma"

# 2. GSE279942 - Rectal Cancer (97 samples)  
download_dataset "GSE279942" "Rectal Cancer Spatial Transcriptomics"

# 3. GSE243408 - Endometrial Cancer (90 samples)
download_dataset "GSE243408" "Endometrial Cancer MMRd"

# 4. GSE306381 - Alzheimer's Brain (270 samples)
download_dataset "GSE306381" "Alzheimer's Disease Microglial Expression"

echo "========================================="
echo "✅ ALL DOWNLOADS COMPLETE!"
echo "========================================="
echo ""
echo "Downloaded datasets:"
ls -d "$BASE_DIR"/*/ 2>/dev/null || echo "No datasets found"
echo ""
echo "Next steps:"
echo "1. Check the files in each dataset directory"
echo "2. Update your GeoMxParser to read real DCC files"
echo "3. Run: python scripts/parse_real_geomx.py"
