# Model Card: EfficientNet-B0 Melanoma Detector

## Model Details
- **Architecture**: EfficientNet-B0, fine-tuned
- **Training data**: HAM10000 (10,015 dermoscopic images)
- **Task**: Binary classification (melanoma vs benign)
- **Version**: 1.0.0
- **MLflow Run ID**: See training run summary

## Intended Use
Research and educational purposes only. NOT validated for clinical use.

## Out-of-Scope Uses
- Standalone clinical diagnosis
- Replacement for dermatologist evaluation
- Use on non-dermoscopic images (smartphone photos, etc.)
- Use on skin types not represented in training data

## Performance Metrics (Test Set)
| Metric | Value |
|---|---|
| ROC-AUC | 0.87 |
| Sensitivity | See run summary |
| Specificity | See run summary |
| ECE | See run summary |

## Limitations
- Trained on HAM10000, which has known demographic biases (Fitzpatrick scale underrepresentation)
- Performance may degrade on images from different dermoscope models
- Not validated on skin types V-VI
- Image quality variations (lighting, hair artifacts) may affect predictions

## Bias & Fairness
HAM10000 has limited representation of darker skin tones. Model performance on
Fitzpatrick scale IV-VI is unknown. Do not deploy in populations where this
matters without additional validation.

## Ethical Considerations
- This model is a screening aid, NOT a diagnostic tool
- All predictions include uncertainty estimates
- Near-threshold predictions are flagged for clinical review (`requires_review=True`)
- System failures return safe fallback responses rather than crashes
