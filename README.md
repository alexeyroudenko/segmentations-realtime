# segmentations-realtime

Real-time webcam segmentation. One script, multiple backends — switch on the fly with `[` / `]`.

## Setup

Requires Python 3.11 and (preferably) an NVIDIA GPU with CUDA.

```bash
conda env create -f environment.yml
conda activate segmentations-realtime
```

On first use of each backend, weights are downloaded into `models/`:
- `.pt` / `.tflite` — directly in `models/`
- HuggingFace (sam2, humanparsing, faceparsing, clothing) — in `models/huggingface/`
- RF-DETR — in `models/rfdetr/`

## Run

```bash
python segmentations-realtime.py
python segmentations-realtime.py --backend fastsam
python segmentations-realtime.py 1 --backend sam2 --width 512 --points 6
```

| Argument | Description |
|----------|-------------|
| `cam` | Camera index (default: `0`) |
| `--backend` | Backend (default: `yolo`) |
| `--weights` | Custom weights path or HF repo id for the selected backend |
| `--imgsz` | Inference size for yolo/fastsam/clothing/rfdetr (640) |
| `--conf` | Confidence threshold for yolo/fastsam/clothing/rfdetr (0.25) |
| `--points` | sam2 `points_per_side`; lower = faster (8) |
| `--width` | Processing width for sam2 / humanparsing / faceparsing (512) |
| `--classes` | Classes for humanparsing/faceparsing, comma-separated (`skin,hair`) |
| `--every` | Run segmentation every Nth frame (1) |

### Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `Space` | Pause |
| `s` | Screenshot (`YYYY-MM-DD-HH-MM-segment-<backend>.jpg`, quality 80%) |
| `[` / `]` | Previous / next backend |

## Backends

### `yolo` — [YOLO-seg](https://docs.ultralytics.com/tasks/segment/) (Ultralytics)

Fastest option. Detects COCO objects (person, car, …) with instance masks.  
Default weights: [`yolo11n-seg.pt`](https://docs.ultralytics.com/models/yolo11/) ([Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)).

### `clothing` — [YOLOv8n Clothing Detection](https://huggingface.co/kesimeg/yolov8n-clothing-detection)

Detects fashion items: **Clothing**, **Shoes**, **Bags**, **Accessories**.  
Finetuned on [Fashionpedia](https://huggingface.co/datasets/detection-datasets/fashionpedia_4_categories). Bounding boxes only — masks are drawn as rectangles inside boxes.  
Default weights: `models/clothing-yolov8n.pt` (downloaded from HuggingFace on first run).

### `fastsam` — [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) (Ultralytics)

Fast class-agnostic “segment everything” mode. Use when you need all regions, not specific categories.  
Default weights: [`FastSAM-s.pt`](https://docs.ultralytics.com/models/fast-sam/).

### `rfdetr` — [RF-DETR](https://github.com/roboflow/rf-detr) (Roboflow)

COCO instance segmentation with masks. Higher accuracy than YOLO-seg at similar latency; Apache 2.0 license.  
Default model: `RFDETRSegSmall` (384×384). Select size via `--weights`: `nano`, `small`, `medium`, `large`.  
Fine-tuned checkpoints: `--weights path/to/checkpoint.pth`.

### `sam2` — [SAM 2](https://github.com/facebookresearch/sam2) (Meta)

Detailed class-agnostic segmentation via [`AutomaticMaskGenerator`](https://github.com/facebookresearch/sam2). Slower than yolo/fastsam, but sharper masks.  
Default model: [`facebook/sam2-hiera-tiny`](https://huggingface.co/facebook/sam2-hiera-tiny). Tune speed with `--points` and `--width`.

### `mediapipe` — [MediaPipe Image Segmenter](https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter)

Lightweight human segmentation: background, hair, face/body skin, clothes.  
Default model: [`selfie_multiclass_256x256`](https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite) (auto-downloaded).

### `humanparsing` — [SegFormer](https://huggingface.co/docs/transformers/model_doc/segformer) (clothes / body)

Semantic human parsing. By default highlights `skin` and `hair` groups; configurable via `--classes`.  
Default model: [`mattmdjaga/segformer_b2_clothes`](https://huggingface.co/mattmdjaga/segformer_b2_clothes).

### `faceparsing` — [SegFormer](https://huggingface.co/docs/transformers/model_doc/segformer) (face)

Face parsing: skin, hair, nose, ears, neck, etc. `skin` / `hair` groups by default.  
Default model: [`jonathandinu/face-parsing`](https://huggingface.co/jonathandinu/face-parsing).

## Examples

```bash
# Quick start
python segmentations-realtime.py --backend yolo

# Face parsing, skin only
python segmentations-realtime.py --backend faceparsing --classes skin

# Faster SAM2
python segmentations-realtime.py --backend sam2 --points 6 --width 384 --every 2

# Clothing detection
python segmentations-realtime.py --backend clothing --conf 0.4

# RF-DETR instance segmentation
python segmentations-realtime.py --backend rfdetr
python segmentations-realtime.py --backend rfdetr --weights nano --conf 0.4
python segmentations-realtime.py --backend rfdetr --weights medium --imgsz 432

# Custom YOLO weights
python segmentations-realtime.py --backend yolo --weights models/yolo11n-seg.pt
```
