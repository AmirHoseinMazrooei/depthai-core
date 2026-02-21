#!/usr/bin/env python3
"""Real-time person bbox + depth-based volume/weight demo (no segmentation APIs)."""

from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass

import cv2
import depthai as dai
import numpy as np

DEFAULT_MODEL = "luxonis/yolov8n:coco-640x352"
PERSON_LABEL = "person"


@dataclass
class EstimationConfig:
    min_depth_mm: int
    max_depth_mm: int
    voxel_size_m: float
    density_kg_per_m3: float
    single_view_correction: float


def parse_size(size: str) -> tuple[int, int]:
    try:
        w_str, h_str = size.lower().split("x")
        return int(w_str), int(h_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected WxH size, got: {size}") from exc


def safe_set_stereo_buffers(stereo: dai.node.StereoDepth) -> None:
    for fn_name in ("setSippBufferSize", "setSippDmaBufferSize"):
        fn = getattr(stereo, fn_name, None)
        if callable(fn):
            fn(8)


def build_pipeline(device: dai.Device, model: str, rgb_size: tuple[int, int], mono_size: tuple[int, int]) -> tuple[dai.Pipeline, dai.node.DetectionNetwork, dai.node.ImageAlign]:
    pipeline = dai.Pipeline(device)

    cam_rgb = pipeline.create(dai.node.Camera)
    cam_left = pipeline.create(dai.node.Camera)
    cam_right = pipeline.create(dai.node.Camera)
    stereo = pipeline.create(dai.node.StereoDepth)
    align = pipeline.create(dai.node.ImageAlign)

    cam_left.build(dai.CameraBoardSocket.CAM_B)
    cam_right.build(dai.CameraBoardSocket.CAM_C)

    rgb = cam_rgb.build()
    rgb_out = rgb.requestOutput(rgb_size, dai.ImgFrame.Type.BGR888i)
    left_out = cam_left.requestOutput(mono_size)
    right_out = cam_right.requestOutput(mono_size)

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    stereo.setLeftRightCheck(False)
    stereo.setSubpixel(False)
    safe_set_stereo_buffers(stereo)

    median_filter = getattr(dai.MedianFilter, "MEDIAN_OFF", None)
    if median_filter is None:
        median_filter = dai.MedianFilter.KERNEL_3x3
    stereo.initialConfig.setMedianFilter(median_filter)

    left_out.link(stereo.left)
    right_out.link(stereo.right)

    stereo.depth.link(align.input)
    rgb_out.link(align.inputAlignTo)

    nn = pipeline.create(dai.node.DetectionNetwork).build(rgb, dai.NNModelDescription(model))
    rgb_out.link(nn.input)

    return pipeline, nn, align


def depth_to_points(depth_mm: np.ndarray, mask: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    z = depth_mm[ys, xs].astype(np.float32) / 1000.0
    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy
    return np.stack((x, y, z), axis=1)


def voxel_volume(points_m: np.ndarray, voxel_size_m: float) -> float:
    if points_m.shape[0] == 0:
        return 0.0
    vox = np.floor(points_m / voxel_size_m).astype(np.int32)
    uniq = np.unique(vox, axis=0)
    return float(len(uniq)) * (voxel_size_m**3)


def det_to_bbox(det: object, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = int(np.clip(getattr(det, "xmin"), 0.0, 1.0) * width)
    y1 = int(np.clip(getattr(det, "ymin"), 0.0, 1.0) * height)
    x2 = int(np.clip(getattr(det, "xmax"), 0.0, 1.0) * width)
    y2 = int(np.clip(getattr(det, "ymax"), 0.0, 1.0) * height)
    return max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2)


def det_person_candidates(det_msg: dai.ImgDetections, labels: list[str], width: int, height: int) -> list[tuple[object, tuple[int, int, int, int]]]:
    candidates = []
    for det in det_msg.detections:
        label_idx = int(getattr(det, "label", -1))
        if 0 <= label_idx < len(labels):
            if labels[label_idx] != PERSON_LABEL:
                continue
        bbox = det_to_bbox(det, width, height)
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        candidates.append((det, bbox))
    return candidates


def select_target(candidates: list[tuple[object, tuple[int, int, int, int]]], depth: np.ndarray, target: str, min_depth_mm: int, max_depth_mm: int) -> tuple[object, tuple[int, int, int, int]] | None:
    if not candidates:
        return None
    if target == "largest":
        return max(candidates, key=lambda item: (item[1][2] - item[1][0]) * (item[1][3] - item[1][1]))

    best = None
    best_depth = float("inf")
    for item in candidates:
        x1, y1, x2, y2 = item[1]
        roi = depth[y1:y2, x1:x2]
        valid = roi[(roi > min_depth_mm) & (roi < max_depth_mm)]
        if valid.size == 0:
            continue
        med = float(np.median(valid))
        if med < best_depth:
            best_depth = med
            best = item
    return best if best is not None else max(candidates, key=lambda item: (item[1][2] - item[1][0]) * (item[1][3] - item[1][1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--rgb-size", type=parse_size, default=(640, 400))
    parser.add_argument("--mono-size", type=parse_size, default=(480, 270))
    parser.add_argument("--min-depth-mm", type=int, default=400)
    parser.add_argument("--max-depth-mm", type=int, default=3500)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--density", type=float, default=985.0)
    parser.add_argument("--single-view-correction", type=float, default=1.85)
    parser.add_argument("--accum-seconds", type=float, default=1.5)
    parser.add_argument("--accum-max-frames", type=int, default=45)
    parser.add_argument("--accum-min-frames", type=int, default=10)
    parser.add_argument("--accum-reset-on-miss", type=int, default=5)
    parser.add_argument("--target", choices=["largest", "closest"], default="largest")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--record-path", default="realtime_demo.mp4")
    parser.add_argument("--record-fps", type=int, default=20)
    args = parser.parse_args()

    cfg = EstimationConfig(
        min_depth_mm=args.min_depth_mm,
        max_depth_mm=args.max_depth_mm,
        voxel_size_m=args.voxel_size,
        density_kg_per_m3=args.density,
        single_view_correction=args.single_view_correction,
    )

    with dai.Device() as device:
        pipeline, nn, align = build_pipeline(device, args.model, args.rgb_size, args.mono_size)
        q_rgb = nn.passthrough.createOutputQueue(maxSize=4, blocking=False)
        q_det = nn.out.createOutputQueue(maxSize=4, blocking=False)

        depth_stream = getattr(align, "outputAligned", None)
        if depth_stream is None:
            depth_stream = align.out
        q_depth = depth_stream.createOutputQueue(maxSize=4, blocking=False)

        device.startPipeline(pipeline)

        calib = device.readCalibration()
        intrinsics = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, args.rgb_size[0], args.rgb_size[1]), dtype=np.float32)
        labels = nn.getClasses() or []

        point_buffer: deque[np.ndarray] = deque()
        ts_buffer: deque[float] = deque()
        smoothed_weight: float | None = None
        miss_count = 0
        writer: cv2.VideoWriter | None = None

        frame_idx = 0
        t0 = time.monotonic()

        while True:
            rgb_msg = q_rgb.get()
            det_msg = q_det.get()
            depth_msg = q_depth.get()

            frame = rgb_msg.getCvFrame()
            depth = depth_msg.getCvFrame()
            h, w = depth.shape[:2]

            candidates = det_person_candidates(det_msg, labels, w, h)
            selected = select_target(candidates, depth, args.target, cfg.min_depth_mm, cfg.max_depth_mm)

            status = "ACCUMULATING..."
            raw_vol = 0.0
            corr_vol = 0.0
            bbox = None

            if selected is not None:
                _, bbox = selected
                x1, y1, x2, y2 = bbox
                roi_mask = np.zeros((h, w), dtype=bool)
                roi_mask[y1:y2, x1:x2] = True
                valid_depth = (depth > cfg.min_depth_mm) & (depth < cfg.max_depth_mm)
                mask = roi_mask & valid_depth

                if np.any(mask):
                    pts = depth_to_points(depth, mask, intrinsics)
                    now = time.monotonic()

                    point_buffer.append(pts)
                    ts_buffer.append(now)

                    while ts_buffer and now - ts_buffer[0] > args.accum_seconds:
                        ts_buffer.popleft()
                        point_buffer.popleft()

                    while len(point_buffer) > args.accum_max_frames:
                        ts_buffer.popleft()
                        point_buffer.popleft()

                    miss_count = 0

                    if len(point_buffer) >= args.accum_min_frames:
                        merged = np.vstack(point_buffer)
                        if merged.shape[0] > 500_000:
                            keep = np.random.choice(merged.shape[0], 500_000, replace=False)
                            merged = merged[keep]

                        raw_vol = voxel_volume(merged, cfg.voxel_size_m)
                        corr_vol = raw_vol * cfg.single_view_correction
                        weight = corr_vol * cfg.density_kg_per_m3
                        smoothed_weight = weight if smoothed_weight is None else (0.85 * smoothed_weight + 0.15 * weight)
                        status = f"Weight: {smoothed_weight:.1f} kg"
                else:
                    miss_count += 1
            else:
                miss_count += 1

            if miss_count >= args.accum_reset_on_miss:
                point_buffer.clear()
                ts_buffer.clear()
                smoothed_weight = None

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 230, 70), 2)
                cv2.putText(frame, status, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (70, 230, 70), 2)
                if smoothed_weight is not None:
                    cv2.putText(frame, f"Vraw={raw_vol:.3f}m3 Vcorr={corr_vol:.3f}m3", (x1, min(frame.shape[0] - 10, y2 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 255), 1)
            else:
                cv2.putText(frame, "No person detected", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            frame_idx += 1
            elapsed = max(1e-6, time.monotonic() - t0)
            fps = frame_idx / elapsed
            cv2.putText(frame, f"FPS: {fps:.1f}  Buffer: {len(point_buffer)}", (20, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if args.record:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.record_path, fourcc, args.record_fps, (frame.shape[1], frame.shape[0]))
                writer.write(frame)

            cv2.imshow("Realtime BBox Weight Demo", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

        if writer is not None:
            writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
