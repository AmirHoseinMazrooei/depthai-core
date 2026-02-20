#!/usr/bin/env python3
"""Estimate human volume and weight from OAK-D Lite using segmentation + depth."""

from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass

import cv2
import depthai as dai
import numpy as np

DEFAULT_MODEL = "luxonis/yolov8-instance-segmentation-nano:coco-512x288"
PERSON_LABEL = "person"


@dataclass
class EstimateConfig:
    voxel_size_m: float = 0.02
    min_depth_mm: int = 400
    max_depth_mm: int = 3500
    density_kg_per_m3: float = 985.0
    single_view_correction: float = 1.85


def build_pipeline(device: dai.Device, model_name: str) -> tuple[dai.Pipeline, dai.node.DetectionNetwork]:
    pipeline = dai.Pipeline(device)

    cam = pipeline.create(dai.node.Camera)
    left = pipeline.create(dai.node.Camera)
    right = pipeline.create(dai.node.Camera)
    stereo = pipeline.create(dai.node.StereoDepth)
    align = pipeline.create(dai.node.ImageAlign)

    left.build(dai.CameraBoardSocket.CAM_B)
    right.build(dai.CameraBoardSocket.CAM_C)

    rgb = cam.build()
    rgb_out = rgb.requestOutput((640, 400), dai.ImgFrame.Type.BGR888i)

    nn = pipeline.create(dai.node.DetectionNetwork).build(rgb, dai.NNModelDescription(model_name))

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    stereo.enableLeftRightCheck(True)
    stereo.setSubpixel(True)

    left.requestOutput((640, 400)).link(stereo.left)
    right.requestOutput((640, 400)).link(stereo.right)

    stereo.depth.link(align.input)
    rgb_out.link(align.inputAlignTo)

    rgb_out.link(nn.input)

    return pipeline, nn


def depth_to_points(depth_mm: np.ndarray, mask: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    ys, xs = np.where(mask)
    if ys.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    z = depth_mm[ys, xs].astype(np.float32) / 1000.0
    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy

    return np.stack((x, y, z), axis=1)


def voxel_volume(points_m: np.ndarray, voxel_size_m: float) -> float:
    if len(points_m) == 0:
        return 0.0
    voxel_idx = np.floor(points_m / voxel_size_m).astype(np.int32)
    uniq = np.unique(voxel_idx, axis=0)
    return float(len(uniq)) * (voxel_size_m**3)


def person_mask_from_segmentation(det_msg: dai.ImgDetections, labels: list[str]) -> np.ndarray | None:
    try:
        person_idx = labels.index(PERSON_LABEL)
    except ValueError:
        return None

    # Prefer instance masks if available in current SDK/model output. Fallback to class mask.
    if hasattr(det_msg, "getCvInstanceMask"):
        instance_mask = det_msg.getCvInstanceMask()  # type: ignore[attr-defined]
        if instance_mask is not None:
            return instance_mask > 0

    seg = det_msg.getCvSegmentationMaskByClass(person_idx)
    if seg is None:
        return None

    return seg == person_idx


def safe_init_writer(writer: cv2.VideoWriter | None, path: str, fps: int, frame: np.ndarray) -> cv2.VideoWriter:
    if writer is not None:
        return writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w = frame.shape[:2]
    return cv2.VideoWriter(path, fourcc, fps, (w, h))


def select_component(mask: np.ndarray, depth: np.ndarray, target: str, min_depth_mm: int, max_depth_mm: int) -> np.ndarray:
    if not np.any(mask):
        return mask

    try:
        num_labels, labels_img = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    except cv2.error:
        return mask

    if num_labels <= 2:
        return mask

    valid_depth = (depth > min_depth_mm) & (depth < max_depth_mm)
    components: list[tuple[int, int, float]] = []
    for comp_id in range(1, num_labels):
        comp = labels_img == comp_id
        area = int(np.count_nonzero(comp))
        if area == 0:
            continue

        comp_valid = comp & valid_depth
        if np.any(comp_valid):
            median_depth = float(np.median(depth[comp_valid]))
        else:
            median_depth = float("inf")
        components.append((comp_id, area, median_depth))

    if not components:
        return mask

    if target == "closest":
        best_id = min(components, key=lambda x: (x[2], -x[1]))[0]
    else:
        best_id = max(components, key=lambda x: x[1])[0]

    return labels_img == best_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--density", type=float, default=985.0)
    parser.add_argument("--single-view-correction", type=float, default=1.85)
    parser.add_argument("--accum-seconds", type=float, default=1.5)
    parser.add_argument("--accum-max-frames", type=int, default=45)
    parser.add_argument("--accum-min-frames", type=int, default=10)
    parser.add_argument("--accum-reset-on-miss", type=int, default=5)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--record-path", default="demo_output.mp4")
    parser.add_argument("--record-fps", type=int, default=20)
    parser.add_argument("--box-mode", choices=["mask", "detections"], default="mask")
    parser.add_argument("--target", choices=["largest", "closest"], default="largest")
    args = parser.parse_args()

    cfg = EstimateConfig(
        voxel_size_m=args.voxel_size,
        density_kg_per_m3=args.density,
        single_view_correction=args.single_view_correction,
    )

    with dai.Device() as device:
        pipeline, nn = build_pipeline(device, args.model)
        q_rgb = nn.passthrough.createOutputQueue(maxSize=4, blocking=False)
        q_det = nn.out.createOutputQueue(maxSize=4, blocking=False)

        q_depth = None
        for node in pipeline.getAllNodes():
            if isinstance(node, dai.node.ImageAlign):
                q_depth = node.outputAligned.createOutputQueue(maxSize=4, blocking=False)
                break
        if q_depth is None:
            raise RuntimeError("ImageAlign output queue was not created")

        device.startPipeline(pipeline)

        calib = device.readCalibration()
        intr = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 640, 400), dtype=np.float32)

        labels = nn.getClasses() or []
        if PERSON_LABEL not in labels:
            raise RuntimeError(f"Model has no '{PERSON_LABEL}' class: {labels}")

        smoothed_weight = None
        point_buffer = deque()
        time_buffer = deque()
        mask_area_buffer = deque()
        miss_counter = 0
        writer = None

        t0 = time.monotonic()
        frames = 0

        while True:
            rgb_msg = q_rgb.get()
            det_msg = q_det.get()
            depth_msg = q_depth.get()

            frame = rgb_msg.getCvFrame()
            depth = depth_msg.getCvFrame()

            person_mask = person_mask_from_segmentation(det_msg, labels)
            selected_mask = None
            bbox = None
            raw_volume = 0.0
            corrected_volume = 0.0
            status_text = "ACCUMULATING..."

            if person_mask is not None:
                person_mask = cv2.resize(person_mask.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                if np.any(person_mask):
                    selected_mask = select_component(person_mask, depth, args.target, cfg.min_depth_mm, cfg.max_depth_mm)

            if selected_mask is not None and np.any(selected_mask):
                valid_depth = (depth > cfg.min_depth_mm) & (depth < cfg.max_depth_mm)
                selected_mask = selected_mask & valid_depth

                if np.any(selected_mask):
                    points = depth_to_points(depth, selected_mask, intr)
                    now = time.monotonic()

                    point_buffer.append(points)
                    time_buffer.append(now)
                    mask_area_buffer.append(int(np.count_nonzero(selected_mask)))

                    while len(time_buffer) > 0 and (now - time_buffer[0]) > args.accum_seconds:
                        point_buffer.popleft()
                        time_buffer.popleft()
                        mask_area_buffer.popleft()

                    while len(point_buffer) > args.accum_max_frames:
                        point_buffer.popleft()
                        time_buffer.popleft()
                        mask_area_buffer.popleft()

                    miss_counter = 0

                    if len(point_buffer) >= args.accum_min_frames:
                        combined = np.vstack(point_buffer)
                        if combined.shape[0] > 600000:
                            idx = np.random.choice(combined.shape[0], 600000, replace=False)
                            combined = combined[idx]

                        raw_volume = voxel_volume(combined, cfg.voxel_size_m)
                        corrected_volume = raw_volume * cfg.single_view_correction
                        weight_kg = corrected_volume * cfg.density_kg_per_m3
                        smoothed_weight = weight_kg if smoothed_weight is None else 0.85 * smoothed_weight + 0.15 * weight_kg
                        status_text = f"Weight: {smoothed_weight:.1f} kg"

                    overlay = frame.copy()
                    overlay[selected_mask] = (0, 255, 0)
                    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

                    if args.box_mode == "detections":
                        person_idx = labels.index(PERSON_LABEL)
                        person_dets = [d for d in det_msg.detections if d.label == person_idx]
                        if person_dets:
                            if args.target == "closest":
                                det_scores = []
                                for det in person_dets:
                                    x0 = max(0, int(det.xmin * depth.shape[1]))
                                    y0 = max(0, int(det.ymin * depth.shape[0]))
                                    x1 = min(depth.shape[1] - 1, int(det.xmax * depth.shape[1]))
                                    y1 = min(depth.shape[0] - 1, int(det.ymax * depth.shape[0]))
                                    roi = depth[y0:y1 + 1, x0:x1 + 1]
                                    roi_valid = (roi > cfg.min_depth_mm) & (roi < cfg.max_depth_mm)
                                    median_d = float(np.median(roi[roi_valid])) if np.any(roi_valid) else float("inf")
                                    det_scores.append(((x0, y0, x1, y1), median_d))
                                bbox = min(det_scores, key=lambda x: x[1])[0]
                            else:
                                def det_area(det: dai.ImgDetection) -> float:
                                    return max(0.0, (det.xmax - det.xmin) * (det.ymax - det.ymin))

                                best = max(person_dets, key=det_area)
                                bbox = (
                                    max(0, int(best.xmin * depth.shape[1])),
                                    max(0, int(best.ymin * depth.shape[0])),
                                    min(depth.shape[1] - 1, int(best.xmax * depth.shape[1])),
                                    min(depth.shape[0] - 1, int(best.ymax * depth.shape[0])),
                                )

                    if bbox is None:
                        ys, xs = np.where(selected_mask)
                        if ys.size > 0:
                            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

            else:
                miss_counter += 1
                if miss_counter >= args.accum_reset_on_miss:
                    point_buffer.clear()
                    time_buffer.clear()
                    mask_area_buffer.clear()
                    smoothed_weight = None

            if bbox is not None:
                x0, y0, x1, y1 = bbox
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                txt_y = max(0, y0 - 10)
                cv2.putText(frame, status_text, (x0, txt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if smoothed_weight is None else (0, 0, 255), 2)

            cv2.putText(frame, f"Frames: {len(point_buffer)} / min{args.accum_min_frames}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Window: {args.accum_seconds:.1f}s", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if len(point_buffer) < args.accum_min_frames:
                cv2.putText(frame, "ACCUMULATING...", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            elif smoothed_weight is not None:
                cv2.putText(frame, f"Weight: {smoothed_weight:.1f} kg", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Raw volume: {raw_volume:.3f} m^3", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Corrected volume: {corrected_volume:.3f} m^3", (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            frames += 1
            fps = frames / max(1e-6, (time.monotonic() - t0))
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if args.record:
                writer = safe_init_writer(writer, args.record_path, args.record_fps, frame)
                writer.write(frame)

            cv2.imshow("OAK-D Lite human volume/weight estimator", frame)
            key = cv2.waitKey(1)
            if key in (ord("q"), 27):
                break

    if writer is not None:
        writer.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
