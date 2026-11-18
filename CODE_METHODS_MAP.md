# Code ↔ methods.md alignment

| methods.md item | Implementation |
|---|---|
| 1,404 clips / 88 infants, ChildID StratifiedGroupKFold×10 | `config/shared_fold_indices.json`, seed 3443 |
| 20 fps, 224², T=20, intelligent padding | `src/video_preprocess.py` |
| Uniform + high optical-flow frames | `select_frame_indices` + Farneback |
| Rotation / brightness / flip / temporal jitter & drop | `augment_clip` (enabled by default) |
| ResNet-50 ImageNet embeddings | `src/data_processor.py` |
| MediaPipe face 15-D, hands 45-D, motion 8-D | `scripts/run_multimodal.py` |
| CNN-LSTM kernels {3,5,7}, BiLSTM 256→128, 8 heads | `models/cnn_lstm.py`, `config` |
| 3D-CNN filters [64…512], dual spatial–temporal attention | `models/cnn_3d.py` |
| TAN 4 blocks, 8 heads | `models/tan.py` |
| Multi-cue cross-attn + gating; 7 ablations | `models/multimodal.py`, `scripts/run_multimodal_ablation.py` |
| Adam lr=5e-4, β=(0.9,0.999), ε=1e-7, clipnorm=1, label smoothing 0.05 | model `compile` + `StandardConfig` |
| ReduceLROnPlateau patience 8; ES patience 25 | callbacks |
| Batch 16 (8 for 3D-CNN) | runners / `batch_size_3d` |
| Macro P/R/F1 + Cohen’s κ | `src/eval_metrics.py` |
| Nested selection then outer-train refit | `src/cross_validation.py` |
| Seeds TF/NP/Py/sklearn/CV | `config.config.RANDOM_SEEDS`, `cv_random_state=3443` |
