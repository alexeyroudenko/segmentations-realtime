# Unified realtime segmentation runner with selectable backends.
#
#   python realtime_segment.py --backend yolo
#   python realtime_segment.py --backend fastsam --width 512
#   python realtime_segment.py --backend sam2 --points 8
#   python realtime_segment.py --backend clothing
#
# Backends:
#   yolo     - YOLO*-seg (ultralytics): class labels + boxes, fastest
#   fastsam  - FastSAM (ultralytics): class-agnostic "segment everything", fast
#   sam2     - SAM2 AutomaticMaskGenerator: class-agnostic "everything", slow but detailed
#   clothing - YOLOv8n clothing detection (Clothing, Shoes, Bags, Accessories)
#
# Common keys:  q quit | space pause | s screenshot | [ ] prev/next backend

import os
import sys
import time
import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"


def setup_models_dir():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(MODELS_DIR / "huggingface"))
    try:
        from ultralytics import settings
        settings.update({"weights_dir": str(MODELS_DIR)})
    except ImportError:
        pass


def resolve_weights(weights: str | None, default: str) -> str:
    """Local .pt/.tflite files live under models/; HF repo ids are passed through."""
    name = weights or default
    if name.count("/") == 1 and not name.endswith((".pt", ".tflite", ".onnx")):
        return name
    path = Path(name)
    if path.is_file():
        return str(path)
    return str(MODELS_DIR / path.name)


setup_models_dir()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AUTOCAST = DEVICE == "cuda"
AUTOCAST_DTYPE = torch.bfloat16

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

COLORS = [
    (46, 204, 113),
    (52, 152, 219),
    (231, 76, 60),
    (241, 196, 15),
    (155, 89, 182),
    (230, 126, 34),
    (26, 188, 156),
    (236, 100, 165),
    (149, 165, 166),
]


@dataclass
class Instance:
    mask: np.ndarray            # uint8 HxW (0/1), full frame resolution
    box: tuple                  # (x1, y1, x2, y2) in frame coords
    score: float = 0.0          # conf (yolo) or predicted_iou (sam)
    label: str = ""             # class name; empty for class-agnostic backends


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class Segmenter:
    """Common interface. process() returns a list of Instance for one BGR frame."""

    name = "base"

    def process(self, frame_bgr) -> list:
        raise NotImplementedError


class YoloSeg(Segmenter):
    name = "yolo"

    def __init__(self, weights="yolo11n-seg.pt", imgsz=640, conf=0.25):
        from ultralytics import YOLO
        self.model = YOLO(resolve_weights(weights, "yolo11n-seg.pt"))
        self.imgsz = imgsz
        self.conf = conf
        self.names = self.model.names

    def process(self, frame_bgr):
        r = self.model.predict(
            frame_bgr, imgsz=self.imgsz, conf=self.conf,
            retina_masks=True, device=DEVICE, verbose=False,
        )[0]
        out = []
        if r.masks is None:
            return out
        masks = r.masks.data.cpu().numpy()
        for m, b in zip(masks, r.boxes):
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            out.append(Instance(
                mask=m.astype(np.uint8),
                box=(x1, y1, x2, y2),
                score=float(b.conf[0]),
                label=self.names[int(b.cls[0])],
            ))
        return out


class ClothingYolo(Segmenter):
    """YOLOv8n finetuned for fashion categories (kesimeg/yolov8n-clothing-detection).

    Detection-only model — masks are rectangular regions inside bounding boxes.
    """

    name = "clothing"
    HF_REPO = "kesimeg/yolov8n-clothing-detection"
    DEFAULT_WEIGHTS = "clothing-yolov8n.pt"

    def __init__(self, weights=None, imgsz=640, conf=0.25):
        from ultralytics import YOLO
        self.model = YOLO(self._ensure_weights(weights))
        self.imgsz = imgsz
        self.conf = conf
        self.names = self.model.names

    def _ensure_weights(self, weights):
        if weights:
            return resolve_weights(weights, self.DEFAULT_WEIGHTS)
        path = MODELS_DIR / self.DEFAULT_WEIGHTS
        if path.is_file():
            return str(path)
        print(f"Downloading clothing model -> {path}")
        from huggingface_hub import hf_hub_download
        import shutil
        cached = hf_hub_download(repo_id=self.HF_REPO, filename="best.pt")
        shutil.copy2(cached, path)
        return str(path)

    def process(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        r = self.model.predict(
            frame_bgr, imgsz=self.imgsz, conf=self.conf,
            device=DEVICE, verbose=False,
        )[0]
        out = []
        if r.boxes is None:
            return out
        for b in r.boxes:
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            mask = np.zeros((h, w), dtype=np.uint8)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 1
            out.append(Instance(
                mask=mask,
                box=(x1, y1, x2, y2),
                score=float(b.conf[0]),
                label=self.names[int(b.cls[0])],
            ))
        return out


class FastSamEverything(Segmenter):
    name = "fastsam"

    def __init__(self, weights="FastSAM-s.pt", imgsz=640, conf=0.4):
        from ultralytics import FastSAM
        self.model = FastSAM(resolve_weights(weights, "FastSAM-s.pt"))
        self.imgsz = imgsz
        self.conf = conf

    def process(self, frame_bgr):
        r = self.model.predict(
            frame_bgr, imgsz=self.imgsz, conf=self.conf,
            retina_masks=True, device=DEVICE, verbose=False,
        )[0]
        out = []
        if r.masks is None:
            return out
        masks = r.masks.data.cpu().numpy()
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else None
        for i, m in enumerate(masks):
            if boxes is not None:
                x1, y1, x2, y2 = (int(v) for v in boxes[i])
            else:
                ys, xs = np.where(m > 0)
                x1, y1, x2, y2 = (int(xs.min()), int(ys.min()),
                                  int(xs.max()), int(ys.max())) if len(xs) else (0, 0, 0, 0)
            out.append(Instance(mask=m.astype(np.uint8), box=(x1, y1, x2, y2)))
        return out


class Sam2Everything(Segmenter):
    name = "sam2"

    def __init__(self, model_name="facebook/sam2-hiera-tiny", points_per_side=8,
                 proc_width=512):
        from sam2.build_sam import build_sam2_hf
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        model = build_sam2_hf(model_name, device=DEVICE)
        self.gen = SAM2AutomaticMaskGenerator(
            model,
            points_per_side=points_per_side,
            points_per_batch=128,
            pred_iou_thresh=0.75,
            stability_score_thresh=0.92,
            min_mask_region_area=200,
        )
        self.proc_width = proc_width

    def process(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        scale = self.proc_width / w
        small = cv2.resize(frame_bgr, (self.proc_width, int(h * scale)))
        small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        with torch.inference_mode():
            if USE_AUTOCAST:
                with torch.autocast(DEVICE, dtype=AUTOCAST_DTYPE):
                    masks = self.gen.generate(small_rgb)
            else:
                masks = self.gen.generate(small_rgb)

        masks = sorted(masks, key=lambda x: x["area"], reverse=True)
        inv = 1.0 / scale
        out = []
        for m in masks:
            x, y, bw, bh = m["bbox"]
            full = cv2.resize(m["segmentation"].astype(np.uint8), (w, h),
                              interpolation=cv2.INTER_NEAREST)
            out.append(Instance(
                mask=full,
                box=(int(x * inv), int(y * inv), int((x + bw) * inv), int((y + bh) * inv)),
                score=float(round(m["predicted_iou"], 3)),
            ))
        return out


class MediaPipeSeg(Segmenter):
    name = "mediapipe"

    # Categories of the selfie multiclass model (the default).
    SELFIE_LABELS = ["background", "hair", "body-skin", "face-skin", "clothes", "others"]
    DEFAULT_MODEL = "selfie_multiclass_256x256.tflite"
    DEFAULT_URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/"
                   "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite")

    def __init__(self, model_path=None, min_area=400):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self.mp = mp
        path = model_path or self._ensure_model()
        self.labels = self.SELFIE_LABELS if model_path is None else None

        options = vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(model_asset_path=path),
            running_mode=vision.RunningMode.IMAGE,
            output_category_mask=True,
        )
        self.segmenter = vision.ImageSegmenter.create_from_options(options)
        self.min_area = min_area

    def _ensure_model(self):
        import urllib.request
        path = MODELS_DIR / self.DEFAULT_MODEL
        if not path.is_file():
            print(f"Downloading MediaPipe model -> {path}")
            urllib.request.urlretrieve(self.DEFAULT_URL, path)
        return str(path)

    def process(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.segmenter.segment(mp_image)
        cat = np.squeeze(result.category_mask.numpy_view())  # HxW uint8 of category indices

        out = []
        for idx in np.unique(cat):
            if idx == 0:  # background
                continue
            mask = (cat == idx).astype(np.uint8)
            ys, xs = np.where(mask > 0)
            if len(xs) < self.min_area:
                continue
            if self.labels is not None and idx < len(self.labels):
                label = self.labels[idx]
            else:
                label = f"class{int(idx)}"
            out.append(Instance(
                mask=mask,
                box=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                score=1.0,
                label=label,
            ))
        return out


class _SegformerParsing(Segmenter):
    """Base for HuggingFace SegFormer semantic-segmentation parsers.

    Subclasses define MODEL and GROUPS (group name -> set of model label names).
    `classes` selects which groups/labels to emit (comma-separated); each becomes
    a single merged Instance.
    """

    MODEL = None
    GROUPS = {}

    def __init__(self, model_name=None, classes="skin,hair", proc_width=640, min_area=300):
        from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
        name = model_name or self.MODEL
        self.processor = SegformerImageProcessor.from_pretrained(name)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(name).to(DEVICE).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.label2id = {v.lower(): k for k, v in self.id2label.items()}
        self.proc_width = proc_width
        self.min_area = min_area
        self.targets = self._resolve_classes(classes)
        names = ", ".join(n for n, _ in self.targets) or "(none)"
        print(f"[{self.name}] classes: {names}")

    def _resolve_classes(self, classes):
        out = []
        for tok in [c.strip() for c in classes.split(",") if c.strip()]:
            key = tok.lower()
            if key in self.GROUPS:
                ids = {self.label2id[n.lower()] for n in self.GROUPS[key]
                       if n.lower() in self.label2id}
            elif key in self.label2id:
                ids = {self.label2id[key]}
            else:
                print(f"[{self.name}] unknown class '{tok}', skipping")
                continue
            if ids:
                out.append((tok, ids))
        return out

    def process(self, frame_bgr):
        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.proc_width and W > self.proc_width:
            scale = self.proc_width / W
            rgb_in = cv2.resize(rgb, (self.proc_width, int(H * scale)))
        else:
            rgb_in = rgb

        inputs = self.processor(images=rgb_in, return_tensors="pt").to(DEVICE)
        with torch.inference_mode():
            if USE_AUTOCAST:
                with torch.autocast(DEVICE, dtype=AUTOCAST_DTYPE):
                    logits = self.model(**inputs).logits
            else:
                logits = self.model(**inputs).logits

        up = torch.nn.functional.interpolate(
            logits.float(), size=(H, W), mode="bilinear", align_corners=False)
        seg = up.argmax(1)[0].to(torch.int32).cpu().numpy()

        out = []
        for name, ids in self.targets:
            mask = np.isin(seg, list(ids)).astype(np.uint8)
            ys, xs = np.where(mask > 0)
            if len(xs) < self.min_area:
                continue
            out.append(Instance(
                mask=mask,
                box=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                score=1.0,
                label=name,
            ))
        return out


class HumanParsing(_SegformerParsing):
    name = "humanparsing"
    MODEL = "mattmdjaga/segformer_b2_clothes"
    GROUPS = {
        "skin": {"Face", "Left-leg", "Right-leg", "Left-arm", "Right-arm"},
        "hair": {"Hair"},
    }


class FaceParsing(_SegformerParsing):
    name = "faceparsing"
    MODEL = "jonathandinu/face-parsing"
    GROUPS = {
        "skin": {"skin", "nose", "l_ear", "r_ear", "neck", "neck_l"},
        "hair": {"hair"},
    }


BACKENDS = {
    "yolo": lambda a: YoloSeg(a.weights or "yolo11n-seg.pt", a.imgsz, a.conf),
    "clothing": lambda a: ClothingYolo(a.weights, a.imgsz, a.conf),
    "fastsam": lambda a: FastSamEverything(a.weights or "FastSAM-s.pt", a.imgsz, a.conf),
    "sam2": lambda a: Sam2Everything(a.weights or "facebook/sam2-hiera-tiny",
                                     a.points, a.width),
    "mediapipe": lambda a: MediaPipeSeg(a.weights),
    "humanparsing": lambda a: HumanParsing(a.weights, a.classes, a.width),
    "faceparsing": lambda a: FaceParsing(a.weights, a.classes, a.width),
}


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def draw(img, instances):
    overlay = img.copy()
    for i, inst in enumerate(instances):
        overlay[inst.mask > 0] = COLORS[i % len(COLORS)]
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, inst in enumerate(instances):
        color = COLORS[i % len(COLORS)]
        x1, y1, x2, y2 = inst.box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        if inst.label:
            label = f"{inst.label} {inst.score:.2f}"
        else:
            label = f"#{i} iou:{inst.score:.2f}"

        (tw, th), _ = cv2.getTextSize(label, font, 0.45, 1)
        tx, ty = x1, y1 - 6
        if ty - th < 0:
            ty = y1 + th + 6
        cv2.rectangle(img, (tx, ty - th - 4), (tx + tw + 4, ty + 4), color, -1)
        cv2.putText(img, label, (tx + 2, ty), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def draw_hud(img, fps, count, paused, backend):
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    lines = [
        f"Backend: {backend}",
        f"FPS: {fps:.1f}",
        f"Objects: {count}",
        f"Res: {w}x{h}",
    ]
    if paused:
        lines.append("PAUSED (space to resume)")

    for i, line in enumerate(lines):
        y = 24 + i * 22
        cv2.putText(img, line, (10, y), font, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (10, y), font, 0.55, (0, 255, 200), 1, cv2.LINE_AA)

    help_lines = ["q: quit", "space: pause", "s: screenshot", "[ ]: prev/next backend"]
    for i, line in enumerate(help_lines):
        y = h - 10 - (len(help_lines) - 1 - i) * 20
        cv2.putText(img, line, (10, y), font, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, line, (10, y), font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def print_debug_info(args):
    print("=" * 50)
    print(f"Python:      {sys.version.split()[0]}")
    print(f"PyTorch:     {torch.__version__}")
    print(f"OpenCV:      {cv2.__version__}")
    print(f"CUDA avail:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:         {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM:        {vram:.1f} GB")
        print(f"CUDA ver:    {torch.version.cuda}")
    else:
        print("GPU:         none (running on CPU)")
    print(f"Device:      {DEVICE}")
    print(f"Models dir:  {MODELS_DIR}")
    print(f"Backend:     {args.backend}")
    print("=" * 50)


def parse_args():
    p = argparse.ArgumentParser(description="Realtime segmentation with selectable backends")
    p.add_argument("cam", nargs="?", type=int, default=0, help="camera index (default: 0)")
    p.add_argument("--backend", choices=sorted(BACKENDS), default="yolo",
                   help="segmentation engine (default: yolo)")
    p.add_argument("--weights", default=None,
                   help="override model weights / HF id for the chosen backend")
    p.add_argument("--imgsz", type=int, default=640,
                   help="inference size for yolo/fastsam/clothing (default: 640)")
    p.add_argument("--conf", type=float, default=0.25,
                   help="confidence threshold for yolo/fastsam/clothing (default: 0.25)")
    p.add_argument("--points", type=int, default=8,
                   help="sam2 points_per_side; lower = faster (default: 8)")
    p.add_argument("--width", type=int, default=512,
                   help="processing width for sam2 / humanparsing / faceparsing (default: 512)")
    p.add_argument("--classes", default="skin,hair",
                   help="comma-separated classes/groups for humanparsing/faceparsing "
                        "(default: skin,hair)")
    p.add_argument("--every", type=int, default=1,
                   help="run segmentation every Nth frame, reuse between (default: 1)")
    return p.parse_args()


def main():
    args = parse_args()
    print_debug_info(args)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"Cannot open camera {args.cam}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera {args.cam}: {actual_w}x{actual_h}")

    backend_names = sorted(BACKENDS)
    backend_idx = backend_names.index(args.backend)

    def build_backend(name, use_weights):
        """(Re)create a segmenter, freeing the previous one's GPU memory."""
        import copy
        a = args if use_weights else copy.copy(args)
        if not use_weights:
            a.weights = None
        t0 = time.time()
        seg = BACKENDS[name](a)
        print(f"Backend '{name}' ready in {time.time() - t0:.1f}s")
        return seg

    segmenter = build_backend(args.backend, use_weights=True)
    cur_backend = args.backend

    window = "Realtime Segmentation"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, actual_w, actual_h)

    paused = False
    frame = None
    instances = []
    fps = 0.0
    frame_count = 0
    iter_count = 0
    screenshot_idx = 0
    every = max(1, args.every)

    print("Running... press 'q' to quit")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed")
                break

            if iter_count % every == 0:
                t = time.time()
                instances = segmenter.process(frame)
                dt = time.time() - t
                fps = 1.0 / dt if dt > 0 else 0.0
                frame_count += 1

            iter_count += 1
            display = draw(frame.copy(), instances)
        else:
            display = frame.copy() if frame is not None else np.zeros((480, 640, 3), np.uint8)
            display = draw(display, instances)

        display = draw_hud(display, fps, len(instances), paused, cur_backend)
        cv2.imshow(window, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord("s"):
            fname = f"screenshot_{screenshot_idx:04d}.png"
            cv2.imwrite(fname, display)
            print(f"Saved {fname}")
            screenshot_idx += 1
        elif key in (ord("["), ord("]")):
            backend_idx = (backend_idx + (1 if key == ord("]") else -1)) % len(backend_names)
            new_backend = backend_names[backend_idx]

            # Show a quick "loading" overlay before the (blocking) model load.
            loading = display.copy()
            msg = f"Loading {new_backend}..."
            (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cx, cy = (loading.shape[1] - tw) // 2, (loading.shape[0] + th) // 2
            cv2.rectangle(loading, (cx - 20, cy - th - 20), (cx + tw + 20, cy + 20), (0, 0, 0), -1)
            cv2.putText(loading, msg, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 200), 2, cv2.LINE_AA)
            cv2.imshow(window, loading)
            cv2.waitKey(1)

            try:
                new_segmenter = build_backend(new_backend, use_weights=False)
            except Exception as e:
                # Keep the current backend running instead of crashing the app.
                print(f"Failed to load backend '{new_backend}': {type(e).__name__}: {e}")
                backend_idx = backend_names.index(cur_backend)
                err = display.copy()
                emsg = f"'{new_backend}' unavailable - staying on {cur_backend}"
                cv2.putText(err, emsg, (20, err.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow(window, err)
                cv2.waitKey(600)
                continue

            del segmenter
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            segmenter = new_segmenter
            cur_backend = new_backend
            instances = []
            fps = 0.0

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done — processed {frame_count} segmented frames")


if __name__ == "__main__":
    main()
