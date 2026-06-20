# Data Card: HAM10000

## Dataset Overview
- **Name**: HAM10000 ("Human Against Machine with 10000 training images")
- **Size**: 10,015 dermoscopic images, 7 diagnostic categories
- **Source**: ISIC Archive / Harvard Dataverse
- **License**: CC BY-NC 4.0

## Class Distribution
| Code | Diagnosis | Count | % |
|---|---|---|---|
| mel | Melanoma | 1113 | 11.1% |
| nv | Melanocytic nevi | 6705 | 66.9% |
| bcc | Basal cell carcinoma | 514 | 5.1% |
| akiec | Actinic keratoses | 327 | 3.3% |
| bkl | Benign keratosis | 1099 | 11.0% |
| df | Dermatofibroma | 115 | 1.1% |
| vasc | Vascular lesions | 142 | 1.4% |

## Data Splitting
- Patient-aware splits using `lesion_id` grouping
- Train/Val/Test: 70%/15%/15%
- Zero patient leakage between splits (verified by assertion)

## Known Limitations
- Heavily skewed toward lighter skin tones (Fitzpatrick I-III)
- Multiple images per lesion may introduce subtle correlations
- Images captured with different dermoscope models
- Some images contain hair artifacts, rulers, and ink markings

## Preprocessing
- Hair removal (DullRazor algorithm) applied to training images
- Macenko color normalization for consistent color representation
- Resized to 224x224 pixels
- ImageNet normalization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
