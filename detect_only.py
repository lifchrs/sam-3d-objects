"""
Detection-only script for SAM3D object detection.
Runs the detection + mask propagation pipeline but skips 3D reconstruction.
Saves mask visualizations and binary masks for each frame.
"""
import cv2
import json
import torch
import numpy as np
from PIL import Image
from transformers import Sam3VideoModel, Sam3VideoProcessor
from accelerate import Accelerator
import sys
import os
import argparse

THIS_DIR = os.path.join(os.path.dirname(__file__), "sam3d_objects")
SAM3D_ROOT = os.path.abspath(os.path.dirname(__file__))
NOTEBOOK_DIR = os.path.join(SAM3D_ROOT, "notebook")
if NOTEBOOK_DIR not in sys.path:
    sys.path.insert(0, NOTEBOOK_DIR)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from detection_strategies import (
    detect_object, DetectionResult, _masks_iou,
    project_joints_to_2d, get_hand_bbox_from_joints,
)


def load_video_cv2(video_path):
    """Load video frames using OpenCV, returning list of PIL Images and fps."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR -> RGB -> PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames, fps


def main():
    parser = argparse.ArgumentParser(description="Detection-only: detect objects in video using SAM3")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--prompt", type=str, default="object held in hand", help="Text prompt for SAM3")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for masks")
    parser.add_argument("--stride", type=int, default=2, help="Stride for frame processing")
    parser.add_argument(
        "--detection_strategy",
        type=str,
        choices=["simple", "two_stage", "backward", "consensus", "combined", "hand_point", "grounding_dino"],
        default="consensus",
        help="Object detection strategy",
    )
    parser.add_argument("--fps", type=float, default=None, help="Output video FPS (auto-detected from input if not set)")
    parser.add_argument("--max_objects", type=int, default=1, help="Max objects to detect (>1 enables multi-object)")
    parser.add_argument("--min_mask_area", type=float, default=0.005, help="Min mask area as fraction of frame")
    parser.add_argument("--additional_prompts", type=str, default=None,
                        help="Comma-separated additional prompts for multi-object (e.g. 'phone,cup,tool')")
    parser.add_argument("--vitra_annotation", type=str, default=None,
                        help="Path to VITRA .npy annotation for hand-anchored detection")
    parser.add_argument("--use_llm", action="store_true",
                        help="Use a local LLM (Llama-3.1-8B) for object noun extraction")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable verbose print output (default: True). Use --no-verbose to suppress.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    verbose = args.verbose

    accelerator = Accelerator()
    device = accelerator.device
    if verbose:
        print(f"Using device: {device}")

    # Load SAM3 model
    if verbose:
        print("Loading SAM3 video model...")
    model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
    processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")

    # Load video using OpenCV (works with webm)
    if verbose:
        print(f"Loading video: {args.video}")
    video_frames, video_fps = load_video_cv2(args.video)
    # Use auto-detected fps unless user explicitly set --fps
    output_fps = video_fps if args.fps is None else args.fps
    if verbose:
        print(f"  Total frames: {len(video_frames)}, fps: {video_fps:.1f} (output: {output_fps:.1f})")

    if len(video_frames) == 0:
        print("ERROR: No frames loaded from video")
        return

    # Run detection
    detection_results = None
    hand_bboxes_per_frame = {}  # frame_idx -> list of (bbox, hand_side)

    # Try hand-anchored detection if VITRA annotation is provided
    if args.vitra_annotation:
        if verbose:
            print(f"\n=== Loading VITRA annotation: {args.vitra_annotation} ===")
        annotation = np.load(args.vitra_annotation, allow_pickle=True).item()
        intrinsics = annotation.get('intrinsics')
        text_data = annotation.get('text', {})

        hand_results = []

        for hand_side in ['left', 'right']:
            hand_raw = annotation.get(hand_side)
            if hand_raw is None:
                if verbose:
                    print(f"  {hand_side} hand: not present")
                continue
            kept_frames = hand_raw.get('kept_frames', np.array([]))
            if len(kept_frames) == 0 or kept_frames.sum() == 0:
                if verbose:
                    print(f"  {hand_side} hand: no valid frames")
                continue

            joints = hand_raw['joints_camspace']  # (T, 21, 3)

            # Get per-hand action text, fall back to user prompt
            hand_texts = text_data.get(hand_side, [])
            if hand_texts:
                hand_prompt = hand_texts[0][0]
            else:
                hand_prompt = args.prompt

            if verbose:
                print(f"\n=== Detecting for {hand_side} hand (prompt: '{hand_prompt}') ===")

            result = detect_object(
                model=model,
                processor=processor,
                video_frames=video_frames,
                prompt=hand_prompt,
                strategy="hand_point",
                device=device,
                dtype=torch.bfloat16,
                verbose=verbose,
                hand_joints=joints,
                intrinsics=intrinsics,
                kept_frames=kept_frames,
                use_llm=args.use_llm,
            )

            if result.masks:
                result.metadata['hand'] = hand_side
                result.metadata['hand_prompt'] = hand_prompt
                hand_results.append(result)
                if verbose:
                    print(f"  {hand_side} hand: {len(result.masks)} frames with masks")
            else:
                if verbose:
                    print(f"  {hand_side} hand: no detection")

        # De-duplicate: if both hands found the same object, keep higher scorer
        if len(hand_results) == 2:
            common_frames = sorted(
                set(hand_results[0].masks.keys()) & set(hand_results[1].masks.keys())
            )
            if common_frames:
                mid_frame = common_frames[len(common_frames) // 2]
                mask0 = hand_results[0].masks[mid_frame].astype(bool)
                mask1 = hand_results[1].masks[mid_frame].astype(bool)
                iou = _masks_iou(mask0, mask1)

                if iou > 0.5:
                    # Same object — keep the one with more masks / higher score
                    scores = []
                    for det in hand_results:
                        if det.scores:
                            scores.append(sum(det.scores.values()) / len(det.scores))
                        else:
                            scores.append(0)
                    better = 0 if scores[0] >= scores[1] else 1
                    kept_hand = hand_results[better].metadata['hand']
                    if verbose:
                        print(f"\n  Both hands found same object (IoU={iou:.2f}), keeping {kept_hand} hand")
                    hand_results = [hand_results[better]]
                else:
                    if verbose:
                        print(f"\n  Two distinct objects found (IoU={iou:.2f})")

        # Precompute per-frame hand bboxes for overlay visualization
        hand_bboxes_per_frame = {}  # frame_idx -> list of (bbox, hand_side)
        img_h, img_w = np.array(video_frames[0]).shape[:2]
        for hand_side in ['left', 'right']:
            hand_raw = annotation.get(hand_side)
            if hand_raw is None:
                continue
            kf = hand_raw.get('kept_frames', np.array([]))
            joints = hand_raw.get('joints_camspace')
            if joints is None:
                continue
            for fidx in range(min(len(video_frames), joints.shape[0])):
                if len(kf) > fidx and kf[fidx] == 0:
                    continue
                joints_2d = project_joints_to_2d(joints[fidx], intrinsics)
                bbox = get_hand_bbox_from_joints(joints_2d, (img_h, img_w), padding=50)
                if fidx not in hand_bboxes_per_frame:
                    hand_bboxes_per_frame[fidx] = []
                hand_bboxes_per_frame[fidx].append((bbox, hand_side))

        if hand_results:
            detection_results = hand_results
        else:
            if verbose:
                print("  No hand-anchored detections, falling back to text-only")

    # Fall back to text-only detection
    if detection_results is None:
        if verbose:
            print(f"\n=== Running {args.detection_strategy} detection ===")
            print(f"Text prompt: {args.prompt}")

        additional_prompts = None
        if args.additional_prompts:
            additional_prompts = [p.strip() for p in args.additional_prompts.split(",")]

        result = detect_object(
            model=model,
            processor=processor,
            video_frames=video_frames,
            prompt=args.prompt,
            strategy=args.detection_strategy,
            device=device,
            dtype=torch.bfloat16,
            verbose=verbose,
            max_objects=args.max_objects,
            min_mask_area_pct=args.min_mask_area,
            additional_prompts=additional_prompts,
        )

        if isinstance(result, list):
            detection_results = result
        else:
            detection_results = [result]

    # Object overlay colors (RGB)
    OBJECT_COLORS = [
        (255, 0, 0),    # red
        (0, 100, 255),  # blue
        (0, 200, 0),    # green
        (255, 165, 0),  # orange
        (200, 0, 200),  # purple
        (0, 200, 200),  # cyan
    ]
    BOX_COLORS = [
        (0, 255, 0),    # green
        (255, 255, 0),  # yellow
        (255, 0, 255),  # magenta
        (0, 255, 255),  # cyan
        (255, 128, 0),  # dark orange
        (128, 255, 0),  # lime
    ]

    if verbose:
        print(f"\nDetection complete: {len(detection_results)} object(s)")
        for i, det in enumerate(detection_results):
            print(f"  Object {i}: strategy={det.strategy_used}, best_frame={det.best_frame_idx}, "
                  f"frames_with_masks={len(det.masks)}")
            if det.scores:
                avg_score = sum(det.scores.values()) / len(det.scores)
                print(f"    avg_score={avg_score:.4f}")

    # Save mask visualizations for each object
    for obj_i, detection_result in enumerate(detection_results):
        obj_dir = os.path.join(args.output_dir, f"object_{obj_i}")
        os.makedirs(obj_dir, exist_ok=True)

        sorted_frames = sorted(detection_result.masks.keys())
        frames_to_save = sorted_frames[::args.stride]
        color = OBJECT_COLORS[obj_i % len(OBJECT_COLORS)]

        for frame_idx in frames_to_save:
            mask = detection_result.masks[frame_idx]
            if mask.sum() == 0:
                continue

            image_pil = video_frames[frame_idx]
            image_np = np.array(image_pil)
            mask_np = mask.astype(bool)

            mask_vis = image_np.copy()
            mask_overlay = np.zeros_like(mask_vis)
            mask_overlay[mask_np] = color
            mask_vis = cv2.addWeighted(mask_vis, 0.6, mask_overlay, 0.4, 0)

            y_indices, x_indices = np.where(mask_np)
            if len(y_indices) > 0:
                box_color = BOX_COLORS[obj_i % len(BOX_COLORS)]
                cv2.rectangle(mask_vis, (x_indices.min(), y_indices.min()),
                              (x_indices.max(), y_indices.max()), box_color, 2)

            mask_path = os.path.join(obj_dir, f"frame_{frame_idx:04d}_mask.jpg")
            cv2.imwrite(mask_path, cv2.cvtColor(mask_vis, cv2.COLOR_RGB2BGR))

            mask_binary_path = os.path.join(obj_dir, f"frame_{frame_idx:04d}_mask.npy")
            np.save(mask_binary_path, mask_np)

        num_masks = len([f for f in os.listdir(obj_dir) if f.endswith('_mask.jpg')])
        if verbose:
            print(f"  Object {obj_i} mask images saved: {num_masks}")

    # Save detection log (always, regardless of verbose)
    all_log_entries = []
    for obj_i, det in enumerate(detection_results):
        obj_log = det.metadata.get("detection_log", [])
        all_log_entries.append({
            "object_index": obj_i,
            "strategy": det.strategy_used,
            "best_frame_idx": det.best_frame_idx,
            "num_frames_with_masks": len(det.masks),
            "metadata": {k: v for k, v in det.metadata.items() if k != "detection_log"},
            "log_entries": obj_log,
        })
    log_path = os.path.join(args.output_dir, "detection_log.json")
    with open(log_path, "w") as f:
        json.dump(all_log_entries, f, indent=2, default=str)
    if verbose:
        print(f"\nDetection log saved to {log_path}")

    # Generate overlay video (all objects combined)
    if verbose:
        print(f"Generating tracked overlay video...")
    h, w = np.array(video_frames[0]).shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    overlay_path = os.path.join(args.output_dir, "tracked_overlay.mp4")
    overlay_writer = cv2.VideoWriter(overlay_path, fourcc, output_fps, (w, h))

    comparison_path = os.path.join(args.output_dir, "tracked_comparison.mp4")
    comparison_writer = cv2.VideoWriter(comparison_path, fourcc, output_fps, (2 * w, h))

    for frame_idx, frame_pil in enumerate(video_frames):
        frame_np = np.array(frame_pil)
        overlay_np = frame_np.copy()

        for obj_i, detection_result in enumerate(detection_results):
            if frame_idx in detection_result.masks:
                mask = detection_result.masks[frame_idx]
                if mask.sum() > 0:
                    mask_bool = mask.astype(bool)
                    color = OBJECT_COLORS[obj_i % len(OBJECT_COLORS)]
                    box_color = BOX_COLORS[obj_i % len(BOX_COLORS)]

                    color_overlay = np.zeros_like(overlay_np)
                    color_overlay[mask_bool] = color
                    overlay_np = cv2.addWeighted(overlay_np, 0.7, color_overlay, 0.3, 0)

                    y_idx, x_idx = np.where(mask_bool)
                    cv2.rectangle(overlay_np, (x_idx.min(), y_idx.min()),
                                  (x_idx.max(), y_idx.max()), box_color, 2)

                    # Label each object
                    if len(detection_results) > 1:
                        cv2.putText(overlay_np, f"obj{obj_i}", (x_idx.min(), y_idx.min() - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

            # Mark best frame
            if frame_idx == detection_result.best_frame_idx:
                cv2.putText(overlay_np, "BEST FRAME", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Draw hand bounding boxes (if VITRA annotation was used)
        HAND_COLORS = {'left': (255, 255, 0), 'right': (0, 255, 255)}  # yellow / cyan
        if frame_idx in hand_bboxes_per_frame:
            for (hbbox, hand_side) in hand_bboxes_per_frame[frame_idx]:
                hx0, hy0, hx1, hy1 = hbbox
                hcolor = HAND_COLORS.get(hand_side, (255, 255, 255))
                cv2.rectangle(overlay_np, (hx0, hy0), (hx1, hy1), hcolor, 2)
                cv2.putText(overlay_np, f"{hand_side[0].upper()} hand",
                            (hx0, hy0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, hcolor, 1)

        overlay_writer.write(cv2.cvtColor(overlay_np, cv2.COLOR_RGB2BGR))

        side_by_side = np.concatenate([frame_np, overlay_np], axis=1)
        comparison_writer.write(cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))

    overlay_writer.release()
    comparison_writer.release()
    if verbose:
        print(f"  Overlay video: {overlay_path}")
        print(f"  Comparison video: {comparison_path}")

    if verbose:
        print(f"\nDone! All outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
