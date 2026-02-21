#!/usr/bin/env python3
"""Offline volume/weight estimation from HDF5 RGBD capture + external masks."""

from __future__ import annotations

import argparse
import csv
import os
import time
from collections import deque

import cv2
import h5py
import numpy as np


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


def segment_bbox(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    x1, x2 = int(0.2 * w), int(0.8 * w)
    y1, y2 = int(0.1 * h), int(0.95 * h)
    mask[y1:y2, x1:x2] = True
    return mask


def segment_from_file(mask_path: str, shape_hw: tuple[int, int], threshold: int) -> np.ndarray:
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        return np.zeros(shape_hw, dtype=bool)
    if mask_img.shape[:2] != shape_hw:
        mask_img = cv2.resize(mask_img, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return mask_img >= threshold


def load_mask_for_frame(masks_dir: str, frame_id: int, shape_hw: tuple[int, int], threshold: int) -> np.ndarray:
    names = [
        f"frame_{frame_id:06d}.png",
        f"frame_{frame_id:05d}.png",
        f"{frame_id:06d}.png",
        f"{frame_id}.png",
    ]
    for name in names:
        path = os.path.join(masks_dir, name)
        if os.path.exists(path):
            return segment_from_file(path, shape_hw, threshold)
    return np.zeros(shape_hw, dtype=bool)


def draw_mask_bbox(frame: np.ndarray, mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 240, 40), 2)
    return x1, y1, x2, y2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-h5", default="capture.h5")
    parser.add_argument("--masks-dir", default=None)
    parser.add_argument("--out-csv", default="results.csv")
    parser.add_argument("--out-mp4", default="offline_demo.mp4")
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--density", type=float, default=985.0)
    parser.add_argument("--single-view-correction", type=float, default=1.85)
    parser.add_argument("--min-depth-mm", type=int, default=400)
    parser.add_argument("--max-depth-mm", type=int, default=3500)
    parser.add_argument("--accum-seconds", type=float, default=1.5)
    parser.add_argument("--accum-max-frames", type=int, default=45)
    parser.add_argument("--accum-min-frames", type=int, default=10)
    parser.add_argument("--mask-threshold", type=int, default=128)
    args = parser.parse_args()

    with h5py.File(args.in_h5, "r") as h5:
        rgb_ds = h5["rgb"]
        depth_ds = h5["depth_mm"]
        t_ds = h5["t"]
        frame_id_ds = h5["frame_id"]
        K = np.array(h5["K"], dtype=np.float32)

        n_frames = rgb_ds.shape[0]
        if n_frames == 0:
            raise RuntimeError("Input HDF5 has no frames")

        h, w = rgb_ds.shape[1], rgb_ds.shape[2]
        fps = float(h5.attrs.get("fps", 10.0))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out_mp4, fourcc, fps, (w, h))

        point_buffer: deque[np.ndarray] = deque()
        ts_buffer: deque[float] = deque()
        smoothed_weight: float | None = None

        with open(args.out_csv, "w", newline="", encoding="utf-8") as f_csv:
            csv_writer = csv.writer(f_csv)
            csv_writer.writerow(["frame_id", "timestamp", "raw_volume_m3", "corrected_volume_m3", "weight_kg"])

            t0 = time.monotonic()
            for i in range(n_frames):
                rgb = rgb_ds[i]
                depth = depth_ds[i]
                ts = float(t_ds[i])
                frame_id = int(frame_id_ds[i])

                if args.masks_dir:
                    mask = load_mask_for_frame(args.masks_dir, frame_id, (h, w), args.mask_threshold)
                    mode = "external"
                else:
                    mask = segment_bbox(rgb)
                    mode = "bbox"

                valid_depth = (depth > args.min_depth_mm) & (depth < args.max_depth_mm)
                mask = mask & valid_depth

                raw_vol = 0.0
                corr_vol = 0.0
                weight = 0.0

                if np.any(mask):
                    pts = depth_to_points(depth, mask, K)
                    point_buffer.append(pts)
                    ts_buffer.append(ts)

                    while ts_buffer and ts - ts_buffer[0] > args.accum_seconds:
                        ts_buffer.popleft()
                        point_buffer.popleft()

                    while len(point_buffer) > args.accum_max_frames:
                        ts_buffer.popleft()
                        point_buffer.popleft()

                    if len(point_buffer) >= args.accum_min_frames:
                        merged = np.vstack(point_buffer)
                        if merged.shape[0] > 700_000:
                            idx = np.random.choice(merged.shape[0], 700_000, replace=False)
                            merged = merged[idx]
                        raw_vol = voxel_volume(merged, args.voxel_size)
                        corr_vol = raw_vol * args.single_view_correction
                        weight_now = corr_vol * args.density
                        smoothed_weight = weight_now if smoothed_weight is None else 0.85 * smoothed_weight + 0.15 * weight_now
                        weight = smoothed_weight

                csv_writer.writerow([frame_id, ts, raw_vol, corr_vol, weight])

                vis = rgb.copy()
                bbox = draw_mask_bbox(vis, mask)
                if bbox is None:
                    cv2.putText(vis, f"mode={mode} no-person-mask", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                else:
                    x1, y1, _, _ = bbox
                    text = "ACCUMULATING..." if weight <= 0 else f"Weight: {weight:.1f} kg"
                    cv2.putText(vis, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 240, 40), 2)

                elapsed = max(1e-6, time.monotonic() - t0)
                proc_fps = (i + 1) / elapsed
                cv2.putText(vis, f"mode={mode} proc_fps={proc_fps:.1f}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                writer.write(vis)

        writer.release()


if __name__ == "__main__":
    main()
