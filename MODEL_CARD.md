# Model card — Zero Mouse Clicks floor-plan readers

Three small semantic-segmentation models used together by `zmc/pipeline.py`.

| File | Classes | Parameters | Size | Input |
|---|---|---|---|---|
| `zmc/models/unet_v2.onnx` | 5 | ~190 k | 0.8 MB | 1×640×640×3 |
| `zmc/models/unet_v3.onnx` | 5 | ~330 k | 1.3 MB | 1×768×768×3 |
| `zmc/models/unet_v4.onnx` | 8 | ~330 k | 1.3 MB | 1×768×768×3 |

Classes: `0 background · 1 interior space · 2 wall · 3 door · 4 window` (v2, v3) plus `5 counter/closet/bar · 6 fixed fixture (toilet, sink, cooktop, appliance) · 7 loose furniture` (v4).

**Input**: an RGB image as float32 in 0–255, **BGR channel order** (as read by OpenCV), padded with white to the fixed size. **Output**: per-pixel logits, shape 1×H×W×classes.

## Architecture

U-Net with depthwise-separable convolutions, five levels, filters (24, 48, 96, 160, 256) for v3/v4 and (16, 32, 64, 128, 192) for v2, bilinear upsampling, batch normalization. Exported with tf2onnx, opset 17.

## Training data

**Synthetic only.** `zmc/synth.py` builds random apartments with rules (recursive room splits, door tree, room types) and draws them in three styles: colored marketing brochure, 3D top view, and black-and-white line drawing. It adds textured floors, furniture, colored and patterned backgrounds, labels, logos and noise. The label of every pixel is exact. No real floor plan and no third-party dataset (such as CubiCasa5k, whose license is non-commercial) was used for training.

Training recipe (v4): 512 px crops, 20 epochs × 600 steps, batch 20 on 2× NVIDIA T4 (Kaggle), fp32, AdamW with cosine decay, class-weighted cross-entropy, random flips. See `training/train_gpu.ipynb`. v2 and v3 were trained with earlier versions of the same generator.

## Evaluation

IoU on 80 fresh plans from the current generator (512 px). This is in-distribution data, so real plans score lower:

| Model | Background | Space | Wall | Door | Window | Counter/closet | Fixture | Furniture |
|---|---|---|---|---|---|---|---|---|
| v2 | 0.90 | 0.83 | 0.73 | 0.48 | 0.65 | | | |
| v3 | 0.91 | 0.85 | 0.92 | 0.81 | 0.90 | | | |
| v4 | 0.89 | 0.79 | 0.84 | 0.77 | 0.83 | 0.87 | 0.87 | 0.95 |

On 9 real marketing plans (not distributed) the evaluation was qualitative. The geometry is recognizable in all of them. Typical errors: dark furniture (kitchen islands) read as walls, white rugs read as background, chairs and cushions read as fixtures. The pipeline averages the three models and corrects much of this with rules.

## Intended use

Drafting a first architectural plan set from a marketing floor plan, for exploration, education, renovation quotes and research. **Not** for construction, permits or legal surveys without verification on site.

## Limitations

The models only know the styles of the generator. Plans with unusual conventions, very low resolution, perspective photos, curved walls or multiple levels fail more often. See the README for pipeline-level limitations.

## License

Apache License 2.0, like the code.
