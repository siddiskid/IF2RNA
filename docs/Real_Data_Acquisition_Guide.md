# IF2RNA Real Data Acquisition Guide

## 🎯 What Data Format Do You Need?

For IF2RNA to work, you need **PAIRED data** from the same tissue:

### 1. **GeoMx Gene Expression Data** (Required)
- **Format**: `.dcc` files (Digital Count Conversion)
- **Contains**: Raw gene expression counts per ROI
- **Source**: GEO database (downloadable)

### 2. **Probe Configuration** (Required)  
- **Format**: `.pkc` files (Probe Kit Configuration)
- **Contains**: Gene probe definitions and annotations
- **Source**: GEO database or NanoString

### 3. **Metadata** (Required)
- **Format**: `.xlsx` or `.csv` files
- **Contains**: Sample information, ROI coordinates, tissue annotations
- **Source**: GEO supplementary files

### 4. **Immunofluorescence Images** (Ideal but often missing)
- **Format**: `.tif`, `.tiff`, `.czi`, `.qptiff`
- **Contains**: Multi-channel IF images (DAPI + protein markers)
- **Challenge**: Often NOT publicly available on GEO
- **Alternative**: Contact study authors or use simulated IF

---

## 📊 RECOMMENDED DATASETS TO START WITH

I searched GEO and found **actual GeoMx datasets** with downloadable data:

### **🥇 TOP PICK: GSE289483** (Pulmonary Cancer)
- **Study**: Spatial transcriptomics of pulmonary pleomorphic carcinoma
- **Samples**: 125 samples
- **Data Available**: ✅ DCC, ✅ PKC, ✅ CSV
- **Download**: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE289483
- **Why good**: Large sample size, complete GeoMx data files

### **🥈 SECOND CHOICE: GSE279942** (Rectal Cancer)  
- **Study**: Pre-treatment biopsies from LARC patients
- **Samples**: 97 samples (53 biopsies)
- **Data Available**: ✅ DCC, ✅ PKC, ✅ XLSX
- **Download**: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE279942
- **Why good**: Well-annotated clinical data

### **🥉 THIRD CHOICE: GSE243408** (Endometrial Cancer)
- **Study**: Mismatch repair-deficient endometrial cancer
- **Samples**: 90 samples
- **Data Available**: ✅ DCC, ✅ PKC
- **Download**: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE243408
- **Why good**: Immunology focus, good for IF markers

### **🧠 BRAIN: GSE306381** (Alzheimer's)
- **Study**: Microglial expression around Aβ plaques
- **Samples**: 270 samples
- **Data Available**: ✅ DCC, ✅ PKC, ✅ TXT
- **Download**: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE306381
- **Why good**: Large dataset, spatial context important

### **🫁 MULTI-ORGAN: GSE299070** (Arteritis)
- **Study**: Giant cell arteritis temporal artery biopsies
- **Samples**: 108 samples
- **Data Available**: ✅ CSV (may need to check for DCC/PKC)
- **Download**: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE299070
- **Why good**: CD45+ segmentation data

---

## 📥 HOW TO DOWNLOAD THE DATA

### **Step 1: Visit the GEO Page**
```bash
# Example for GSE289483
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE289483
```

### **Step 2: Download Supplementary Files**
Look for the download section at the bottom of the page:
- Click on **"Download data: DCC, PKC, CSV"** links
- Or use FTP links directly

### **Step 3: Automated Download (Recommended)**
```bash
# Create download script
cd /Users/siddarthchilukuri/Documents/GitHub/IF2RNA

# For GSE289483
mkdir -p data/geomx_datasets/GSE289483
cd data/geomx_datasets/GSE289483

# Download using wget or curl
wget "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE289nnn/GSE289483/suppl/GSE289483_*.dcc.gz"
wget "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE289nnn/GSE289483/suppl/GSE289483_*.pkc.gz"
```

### **Step 4: Extract Files**
```bash
# Uncompress all downloaded files
gunzip *.gz

# Check what you got
ls -lh
```

---

## 🔍 WHAT EACH FILE TYPE CONTAINS

### **DCC Files (Digital Count Conversion)**
```
Format: Tab-separated text
Contents:
- Header: Sample metadata
- CodeClass: Gene classification (Endogenous, Housekeeping, Negative)
- GeneName: Gene symbols
- Count: Raw expression counts per gene
```

Example structure:
```
<Header>
FileVersion,1.7
SoftwareVersion,GeoMx_NGS_Pipeline_2.3.3.10
</Header>

<Code_Summary>
RTS0000001,ENSG00000000003,TP53,250
RTS0000002,ENSG00000000005,CD3,1200
...
```

### **PKC Files (Probe Kit Configuration)**
```
Format: Excel or CSV
Contents:
- RTS_ID: Probe identifier
- Gene: Gene symbol
- Probe_Type: Classification
- Module: Panel information
```

### **XLSX/CSV Metadata**
```
Contents:
- ROI_ID: Region identifier
- Sample_ID: Sample name
- X_coord, Y_coord: Spatial coordinates
- Area_um2: ROI area
- Tissue_Type: Annotation (Tumor/Stroma/etc)
- Clinical info: Patient metadata
```

---

## 🎨 WHAT ABOUT IF IMAGES?

### **Reality Check:**
Most GEO datasets **DO NOT** include the original IF images because:
1. Image files are very large (GBs per slide)
2. GEO has upload size limits
3. Images contain protected health information (PHI)

### **Your Options:**

#### **Option 1: Contact Study Authors** (Best)
```
Email template:
---
Subject: Request for IF images from GSE289483

Dear Dr. [Author],

I'm a graduate student working on IF2RNA spatial transcriptomics 
prediction. Your GeoMx study (GSE289483) would be perfect for 
validating our model.

Could you share the immunofluorescence images used for ROI selection? 
We need the multi-channel IF data (DAPI + markers) paired with your 
published DCC files.

Thank you!
Bishneet
```

#### **Option 2: Use NanoString Spatial Organ Atlas**
- Website: https://nanostring.com/products/spatial-organ-atlas
- Contains example datasets with images
- May require registration/access request

#### **Option 3: Realistic Simulation** (Current approach)
- Use real DCC/PKC data for gene expression
- Generate realistic IF images based on tissue annotations
- This is what you're currently doing!

#### **Option 4: Alternative IF Datasets**
Look for datasets that specifically mention having images:
- 10x Genomics Visium (has H&E images)
- Human Protein Atlas (IF images available)
- Then pair with GeoMx expression data

---

## 🚀 IMMEDIATE ACTION PLAN

### **Week 1: Download Real Expression Data**

1. **Start with GSE289483** (easiest, most complete)
```bash
cd /Users/siddarthchilukuri/Documents/GitHub/IF2RNA
mkdir -p data/geomx_datasets/GSE289483
cd data/geomx_datasets/GSE289483

# Download supplementary files
wget -r -np -nd "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE289nnn/GSE289483/suppl/"
```

2. **Parse real DCC files**
- Update your `RealGeoMxParser` to read actual DCC format
- Extract gene names, counts, ROI information

3. **Train on real expression data**
- Keep using simulated IF images
- Use actual gene expression targets from DCC files

### **Week 2-3: Expand to Multi-Organ**

Download 2-3 more datasets:
- GSE279942 (Rectal cancer)
- GSE243408 (Endometrial cancer)  
- GSE306381 (Brain/Alzheimer's)

This gives you 3-4 different tissue types!

### **Month 4: Contact Authors for Images**

Once you have good results with real expression + simulated IF:
- Email authors of best-performing datasets
- Request original IF images
- Explain your research and potential collaboration

---

## 📋 DATASET TRACKING TABLE

Create this in your project:

| Dataset ID | Organ | Samples | DCC | PKC | Images | Status |
|-----------|-------|---------|-----|-----|--------|--------|
| GSE289483 | Lung | 125 | ✅ | ✅ | ❌ | To download |
| GSE279942 | Colon | 97 | ✅ | ✅ | ❌ | To download |
| GSE243408 | Uterus | 90 | ✅ | ✅ | ❌ | To download |
| GSE306381 | Brain | 270 | ✅ | ✅ | ❌ | To download |
| GSE299070 | Artery | 108 | ✅ | ❓ | ❌ | To evaluate |

---

## 🎯 SUCCESS CRITERIA

### **Minimum Viable Dataset** (for your proposal):
- ✅ 3-5 organ types
- ✅ Real gene expression (DCC files)
- ✅ 200+ total ROIs across datasets
- ⚠️ Simulated IF images (realistic, tissue-specific)

### **Ideal Complete Dataset** (if authors share):
- ✅ Everything above PLUS
- ✅ Original multi-channel IF images
- ✅ Paired image-ROI mappings

---

## 🔗 QUICK REFERENCE LINKS

### **GEO GeoMx Search**
https://www.ncbi.nlm.nih.gov/gds/?term=geomx

### **NanoString Resources**
- Spatial Organ Atlas: https://nanostring.com/products/spatial-organ-atlas
- GeoMx Data Center: https://nanostring.com/products/geomx-data-center

### **Data Download Tools**
```bash
# Install GEO download tools
pip install GEOparse

# Or use command-line tools
brew install wget  # macOS
```

---

## ✅ NEXT STEPS

1. **TODAY**: Download GSE289483 dataset
2. **THIS WEEK**: Parse real DCC files and update your parser
3. **NEXT WEEK**: Train IF2RNA on real expression data
4. **THIS MONTH**: Expand to 3+ organs, create dataset table for proposal

You can make HUGE progress by using real expression data with simulated IF images!
