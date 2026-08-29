# SONARIS — MVP Phase 1

This phase detects the single verified dataset class, **Pipeline**, in
SubPipeMiniSSS side-scan sonar imagery. It does not classify marine debris.
Potentially unusual detections receive a transparent, heuristic review score;
they are not confirmed anomalies or debris.

## Commands

Create a project-local virtual environment with Python 3.11 or 3.12, then:

```powershell
python -m pip install -r requirements.txt
python -m src.data.prepare_dataset
python -m src.models.train_detector
python -m src.models.inference --image data/raw/SubPipeMiniSSS/DATA/SSS_LF_images/Image/<image>.pbm
python -m unittest discover -s tests -v
```

Dataset preparation leaves `data/raw` unchanged. It decodes sonar rasters by
their content, applies percentile-based grayscale contrast normalization, and creates derived PNG files and YOLO labels under
`data/processed/sonaris_detection`, validates labels, excludes orphan labels,
and creates a single chronological 80/20 train/validation holdout shared by HF
and LF data. To replace an existing derived dataset, pass `--overwrite`.
