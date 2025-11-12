# IF2RNA

Predicting Spatial Gene Expression from Immunofluorescence Imaging using Deep Learning

## Overview

IF2RNA is a deep learning framework for predicting whole-slide gene expression from immunofluorescence (IF) images using paired GeoMx Digital Spatial Profiler data. The model extends the HE2RNA approach to work with spatial transcriptomics data across multiple organ types.

## Project Structure

```
IF2RNA/
├── src/if2rna/          # Core IF2RNA implementation
│   ├── model.py         # IF2RNA architecture (HE2RNA-adapted)
│   ├── data.py          # Data loading and preprocessing
│   ├── experiment.py    # Configuration-driven experiments
│   └── config.py        # Model and training configurations
├── analysis/            # Performance analysis and validation
├── docs/                # Documentation and summaries
├── external/            # Reference implementations (HE2RNA)
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Installation

```bash
# Setup environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Coming soon...

## Development

This project is part of a directed studies program at UBC, supervised by Dr. Amrit Singh and Dr. Jiarui Ding.