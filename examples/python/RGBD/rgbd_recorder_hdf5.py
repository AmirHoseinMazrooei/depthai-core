#!/usr/bin/env python3
"""Record aligned RGB + depth + intrinsics into a single HDF5 file."""

from __future__ import annotations

import argparse
import datetime as dt
import time
from collections import deque

import cv2
import depthai as dai
import h5py
import numpy as np


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


def build_pipeline(device: dai.Device, rgb_size: tuple[int, int], mono_size: tuple[int, int], fps: float) -> tuple[dai.Pipeline, dai.node.ImageAlign, dai.Node.Output]:
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

    rgb.setFps(fps)
    cam_left.setFps(fps)
    cam_right.setFps(fps)

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

    return pipeline, align, rgb_out


def to_seconds(msg: dai.ImgFrame) -> float:
    ts = msg.getTimestamp()
    return float(ts.total_seconds()) if ts is not None else 0.0


def create_or_resize(ds: h5py.Dataset, size: int, axis0: int = 0) -> None:
    shape = list(ds.shape)
    shape[axis0] = size
    ds.resize(tuple(shape))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="capture.h5")
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--rgb-size", type=parse_size, default=(640, 400))
    parser.add_argument("--mono-size", type=parse_size, default=(480, 270))
    parser.add_argument("--duration-seconds", type=float, default=0.0)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--record-preview-mp4", action="store_true")
    parser.add_argument("--preview-mp4-path", default="preview.mp4")
    parser.add_argument("--preview-mp4-fps", type=float, default=None)
    parser.add_argument("--compression", choices=["lzf", "gzip", "none"], default="lzf")
    parser.add_argument("--chunk-frames", type=int, default=64)
    parser.add_argument("--flush-every", type=int, default=50)
    args = parser.parse_args()

    compression = None if args.compression == "none" else args.compression
    preview_writer: cv2.VideoWriter | None = None

    with dai.Device() as device:
        pipeline, align, rgb_out = build_pipeline(device, args.rgb_size, args.mono_size, args.fps)

        q_rgb = rgb_out.createOutputQueue(maxSize=8, blocking=False)
        depth_stream = getattr(align, "outputAligned", None)
        if depth_stream is None:
            depth_stream = align.out
        q_depth = depth_stream.createOutputQueue(maxSize=8, blocking=False)

        device.startPipeline(pipeline)

        calib = device.readCalibration()
        K = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, args.rgb_size[0], args.rgb_size[1]), dtype=np.float32)

        h, w = args.rgb_size[1], args.rgb_size[0]
        chunk_rgb = (max(1, args.chunk_frames), h, w, 3)
        chunk_depth = (max(1, args.chunk_frames), h, w)

        with h5py.File(args.out, "w") as h5:
            ds_rgb = h5.create_dataset("rgb", shape=(0, h, w, 3), maxshape=(None, h, w, 3), dtype=np.uint8, chunks=chunk_rgb, compression=compression)
            ds_depth = h5.create_dataset("depth_mm", shape=(0, h, w), maxshape=(None, h, w), dtype=np.uint16, chunks=chunk_depth, compression=compression)
            ds_t = h5.create_dataset("t", shape=(0,), maxshape=(None,), dtype=np.float64, chunks=(max(1, args.chunk_frames),), compression=compression)
            ds_id = h5.create_dataset("frame_id", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=(max(1, args.chunk_frames),), compression=compression)
            h5.create_dataset("K", data=K.astype(np.float32))

            h5.attrs["device_id"] = device.getDeviceInfo().getMxId()
            h5.attrs["rgb_size"] = f"{w}x{h}"
            h5.attrs["mono_size"] = f"{args.mono_size[0]}x{args.mono_size[1]}"
            h5.attrs["fps"] = float(args.fps)
            h5.attrs["units"] = "mm"
            h5.attrs["aligned"] = "depth->rgb"
            h5.attrs["depthai_version"] = getattr(dai, "__version__", "unknown")
            h5.attrs["created_utc_iso"] = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

            rgb_buf: deque[tuple[float, dai.ImgFrame]] = deque(maxlen=30)
            depth_buf: deque[tuple[float, dai.ImgFrame]] = deque(maxlen=30)
            tolerance_s = 0.03

            next_idx = 0
            frame_id = 0
            t_start = time.monotonic()

            while True:
                rgb_msg = q_rgb.tryGet()
                if rgb_msg is not None:
                    rgb_buf.append((to_seconds(rgb_msg), rgb_msg))
                depth_msg = q_depth.tryGet()
                if depth_msg is not None:
                    depth_buf.append((to_seconds(depth_msg), depth_msg))

                if not rgb_buf or not depth_buf:
                    if args.duration_seconds > 0 and (time.monotonic() - t_start) > args.duration_seconds:
                        break
                    key = cv2.waitKey(1) & 0xFF if args.preview else -1
                    if key in (ord("q"), 27):
                        break
                    continue

                rgb_t, rgb_m = rgb_buf[0]
                best_i = None
                best_dt = float("inf")
                for i, (d_t, _) in enumerate(depth_buf):
                    dt_abs = abs(d_t - rgb_t)
                    if dt_abs < best_dt:
                        best_dt = dt_abs
                        best_i = i

                if best_i is None or best_dt > tolerance_s:
                    # keep buffers moving forward by dropping older frame
                    if rgb_buf[0][0] < depth_buf[0][0]:
                        rgb_buf.popleft()
                    else:
                        depth_buf.popleft()
                    continue

                _, depth_m = depth_buf[best_i]
                rgb_buf.popleft()
                for _ in range(best_i + 1):
                    depth_buf.popleft()

                rgb = rgb_m.getCvFrame()
                depth = depth_m.getFrame().astype(np.uint16)

                create_or_resize(ds_rgb, next_idx + 1)
                create_or_resize(ds_depth, next_idx + 1)
                create_or_resize(ds_t, next_idx + 1)
                create_or_resize(ds_id, next_idx + 1)

                ds_rgb[next_idx] = rgb
                ds_depth[next_idx] = depth
                ds_t[next_idx] = rgb_t
                ds_id[next_idx] = frame_id

                if args.preview:
                    vis = rgb.copy()
                    depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
                    overlay = cv2.addWeighted(vis, 0.65, depth_vis, 0.35, 0.0)
                    cv2.putText(overlay, f"frames={next_idx + 1}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imshow("RGBD Recorder Preview", overlay)

                    if args.record_preview_mp4:
                        if preview_writer is None:
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            preview_fps = args.preview_mp4_fps if args.preview_mp4_fps is not None else args.fps
                            preview_writer = cv2.VideoWriter(args.preview_mp4_path, fourcc, preview_fps, (overlay.shape[1], overlay.shape[0]))
                        preview_writer.write(overlay)

                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break

                next_idx += 1
                frame_id += 1

                if next_idx % args.flush_every == 0:
                    h5.flush()

                if args.duration_seconds > 0 and (time.monotonic() - t_start) > args.duration_seconds:
                    break

            h5.flush()

    if preview_writer is not None:
        preview_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
