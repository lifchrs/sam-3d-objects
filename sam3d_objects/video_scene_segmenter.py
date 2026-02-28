"""
Full-scene video segmentation module.

Segments **all visible objects** in egocentric video clips using SAM3's
automatic mask generator (for per-frame discovery) and SAM3 Tracker
(for consistent multi-object tracking across frames).

This is text-independent and separate from the manipulated-object detector.

Usage:
    segmenter = VideoSceneSegmenter(device="cuda")

    # Per-frame segmentation
    frame_seg = segmenter.segment_frame(pil_image)

    # Video-level tracking with consistent IDs
    video_seg = segmenter.segment_video(video_frames, keyframe_idx=0)

    # Visualization
    overlay = segmenter.visualize_frame(image_np, frame_seg)
    segmenter.visualize_video(video_frames, video_seg, "output.mp4")
"""

import cv2
import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from detection_strategies import get_bbox_from_mask


def _patch_sam3_nms():
    """Fix dtype mismatch in SAM3 mask-generation NMS postprocess.

    The upstream ``_post_process_for_mask_generation`` casts ``mask_boxes``
    to ``.float()`` but leaves ``iou_scores`` in the model dtype, causing
    ``torchvision.ops.nms`` to raise *"dets should have the same type as
    scores"*.  We replace the function with one that casts both to float32.
    """
    try:
        import transformers.models.sam3.image_processing_sam3_fast as _mod
        from torchvision.ops import batched_nms as _batched_nms

        _original = _mod._post_process_for_mask_generation

        def _patched(rle_masks, iou_scores, mask_boxes, amg_crops_nms_thresh=0.7):
            keep_by_nms = _batched_nms(
                boxes=mask_boxes.float(),
                scores=iou_scores.float(),
                idxs=torch.zeros(mask_boxes.shape[0]),
                iou_threshold=amg_crops_nms_thresh,
            )
            iou_scores = iou_scores[keep_by_nms]
            rle_masks = [rle_masks[i] for i in keep_by_nms]
            mask_boxes = mask_boxes[keep_by_nms]
            masks = [_mod._rle_to_mask(rle) for rle in rle_masks]
            return masks, iou_scores, rle_masks, mask_boxes

        _mod._post_process_for_mask_generation = _patched
    except Exception:
        pass  # If the module layout changes, fall through silently


# Consistent color palette for up to 50 objects (beyond that, colors wrap)
_PALETTE = np.array([
    [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255],
    [0, 255, 255], [128, 0, 0], [0, 128, 0], [0, 0, 128], [128, 128, 0],
    [128, 0, 128], [0, 128, 128], [255, 128, 0], [255, 0, 128], [128, 255, 0],
    [0, 255, 128], [0, 128, 255], [128, 0, 255], [255, 128, 128], [128, 255, 128],
    [128, 128, 255], [255, 255, 128], [255, 128, 255], [128, 255, 255],
    [64, 0, 0], [0, 64, 0], [0, 0, 64], [64, 64, 0], [64, 0, 64],
    [0, 64, 64], [192, 0, 0], [0, 192, 0], [0, 0, 192], [192, 192, 0],
    [192, 0, 192], [0, 192, 192], [64, 128, 0], [128, 64, 0], [0, 64, 128],
    [0, 128, 64], [128, 0, 64], [64, 0, 128], [192, 128, 64], [64, 192, 128],
    [128, 64, 192], [192, 64, 128], [128, 192, 64], [64, 128, 192],
    [160, 82, 45], [70, 130, 180],
], dtype=np.uint8)


def _get_color(obj_id: int) -> np.ndarray:
    return _PALETTE[obj_id % len(_PALETTE)]


@dataclass
class SegmentedObject:
    """A single segmented object in one frame."""
    obj_id: int
    mask: np.ndarray                       # (H, W) boolean
    score: float                           # model confidence / stability
    bbox: Tuple[int, int, int, int]        # (x_min, y_min, x_max, y_max)
    area: int                              # pixel count


@dataclass
class FrameSegmentation:
    """All objects detected in a single frame."""
    frame_idx: int
    objects: List[SegmentedObject]

    def label_map(self) -> np.ndarray:
        """Full label map (H, W) int; pixel value = obj_id, 0 = background."""
        if not self.objects:
            return np.zeros((1, 1), dtype=np.int32)
        h, w = self.objects[0].mask.shape
        labels = np.zeros((h, w), dtype=np.int32)
        for obj in self.objects:
            labels[obj.mask] = obj.obj_id
        return labels


@dataclass
class VideoSegmentation:
    """Tracked segmentation across a full video."""
    frames: Dict[int, FrameSegmentation]   # frame_idx -> segmentation
    object_ids: List[int]                  # all unique obj IDs across video
    num_frames: int


class VideoSceneSegmenter:
    """
    Segments all visible objects in images/video using SAM3.

    Loads two model variants from ``facebook/sam3``:
    - ``pipeline("mask-generation")`` for automatic per-frame mask discovery
    - ``Sam3TrackerVideoModel`` + ``Sam3TrackerVideoProcessor`` for
      multi-object tracking with point prompts derived from auto-masks
    """

    def __init__(self, device="cuda", dtype=torch.bfloat16):
        from transformers import (
            pipeline,
            Sam3TrackerVideoModel,
            Sam3TrackerVideoProcessor,
        )

        self.device = device
        self.dtype = dtype

        # Monkey-patch the SAM3 NMS postprocess to cast scores to float32.
        # The upstream code casts boxes to .float() but leaves iou_scores in
        # the model's dtype, which makes torchvision NMS explode.
        _patch_sam3_nms()

        # Auto-mask generator (image-level)
        print("Loading SAM3 mask-generation pipeline...")
        self.mask_generator = pipeline(
            "mask-generation",
            model="facebook/sam3",
            device=device,
            torch_dtype=dtype,
        )

        # Tracker model for multi-object video propagation
        print("Loading SAM3 Tracker video model...")
        self.tracker_model = Sam3TrackerVideoModel.from_pretrained(
            "facebook/sam3"
        ).to(device, dtype=dtype)
        self.tracker_processor = Sam3TrackerVideoProcessor.from_pretrained(
            "facebook/sam3"
        )

    # ------------------------------------------------------------------
    # Per-frame mode
    # ------------------------------------------------------------------

    def segment_frame(
        self,
        image,
        frame_idx: int = 0,
        **amg_kwargs,
    ) -> FrameSegmentation:
        """
        Run automatic mask generation on a single PIL image.

        Args:
            image: PIL Image (RGB).
            frame_idx: Index to assign to this frame in the result.
            **amg_kwargs: Forwarded to the mask-generation pipeline
                (``points_per_batch``, ``pred_iou_thresh``,
                ``stability_score_thresh``, etc.).

        Returns:
            FrameSegmentation with all discovered objects.
        """
        amg_kwargs.setdefault("points_per_batch", 64)

        outputs = self.mask_generator(image, **amg_kwargs)

        masks_list = outputs["masks"]    # list of (H, W) PIL/tensor masks
        scores_list = outputs["scores"]  # list of float scores

        objects: List[SegmentedObject] = []
        for idx, (mask_raw, score) in enumerate(zip(masks_list, scores_list)):
            # Convert to numpy bool
            mask_np = np.array(mask_raw).astype(bool)
            area = int(mask_np.sum())
            if area == 0:
                continue
            bbox = get_bbox_from_mask(mask_np)
            if bbox is None:
                continue
            score_val = float(score) if not isinstance(score, float) else score
            objects.append(SegmentedObject(
                obj_id=idx + 1,  # 1-indexed (0 = background)
                mask=mask_np,
                score=score_val,
                bbox=bbox,
                area=area,
            ))

        # Sort by area descending
        objects.sort(key=lambda o: o.area, reverse=True)
        # Re-assign contiguous IDs after sorting
        for new_id, obj in enumerate(objects, start=1):
            obj.obj_id = new_id

        return FrameSegmentation(frame_idx=frame_idx, objects=objects)

    # ------------------------------------------------------------------
    # Tracked video mode
    # ------------------------------------------------------------------

    def segment_video(
        self,
        video_frames,
        keyframe_idx: int = 0,
        min_mask_area: int = 500,
        max_objects: int = 50,
        **amg_kwargs,
    ) -> VideoSegmentation:
        """
        Segment all objects on a keyframe and track them across the video.

        Steps:
        1. Run ``segment_frame`` on the keyframe.
        2. Filter tiny masks and cap at ``max_objects``.
        3. Initialise tracker session and feed each mask as a separate object.
        4. Propagate through all frames.

        Args:
            video_frames: List of PIL Images (the full video).
            keyframe_idx: Which frame to run auto-mask on.
            min_mask_area: Drop masks smaller than this (pixels).
            max_objects: Keep at most this many objects (by score).
            **amg_kwargs: Forwarded to ``segment_frame``.

        Returns:
            VideoSegmentation with consistent object IDs.
        """
        # Step 1: auto-mask on keyframe
        print(f"Running auto-mask on keyframe {keyframe_idx}...")
        keyframe_seg = self.segment_frame(
            video_frames[keyframe_idx],
            frame_idx=keyframe_idx,
            **amg_kwargs,
        )
        print(f"  Auto-mask found {len(keyframe_seg.objects)} objects")

        # Step 2: filter — drop tiny masks AND masks that cover >80% of frame
        # (those are typically background / full-scene detections)
        if keyframe_seg.objects:
            h, w = keyframe_seg.objects[0].mask.shape
            max_area = int(h * w * 0.8)
        else:
            max_area = float("inf")
        filtered = [
            o for o in keyframe_seg.objects
            if min_mask_area <= o.area <= max_area
        ]
        # Sort by score descending, then cap
        filtered.sort(key=lambda o: o.score, reverse=True)
        filtered = filtered[:max_objects]
        # Re-assign contiguous 1-indexed IDs
        for new_id, obj in enumerate(filtered, start=1):
            obj.obj_id = new_id

        print(f"  After filtering: {len(filtered)} objects "
              f"(min_area={min_mask_area}, max_objects={max_objects})")

        if not filtered:
            print("  Warning: no objects survived filtering")
            return VideoSegmentation(
                frames={}, object_ids=[], num_frames=len(video_frames)
            )

        # Step 3: init tracker session
        print("Initialising tracker session...")
        session = self.tracker_processor.init_video_session(
            video=video_frames,
            inference_device=self.device,
            dtype=self.dtype,
        )

        # Step 4: add each auto-mask as a tracked object via its centroid
        # point prompt.  We add all objects in one call using the batched
        # multi-object format expected by add_inputs_to_inference_session.
        all_obj_ids = [o.obj_id for o in filtered]

        # Build batched point/label tensors: [1, num_objects, 1, 2] / [1, num_objects, 1]
        points_list = []
        for obj in filtered:
            ys, xs = np.where(obj.mask)
            cx, cy = float(xs.mean()), float(ys.mean())
            points_list.append([[cx, cy]])

        input_points = [[pts for pts in points_list]]   # [1, N_obj, 1, 2]
        input_labels = [[[1] for _ in filtered]]         # [1, N_obj, 1]

        self.tracker_processor.add_inputs_to_inference_session(
            inference_session=session,
            frame_idx=keyframe_idx,
            obj_ids=list(all_obj_ids),  # copy — the processor mutates the list
            input_points=input_points,
            input_labels=input_labels,
        )

        # Run inference on the keyframe to establish the starting point
        _ = self.tracker_model(
            inference_session=session, frame_idx=keyframe_idx,
        )

        # Step 5: propagate through entire video
        print("Propagating through video...")
        frames_dict: Dict[int, FrameSegmentation] = {}

        for output in self.tracker_model.propagate_in_video_iterator(session):
            fidx = output.frame_idx
            pred_masks = self.tracker_processor.post_process_masks(
                [output.pred_masks],
                original_sizes=[
                    [session.video_height, session.video_width]
                ],
                binarize=False,
            )[0]  # (num_objects, 1, H, W)  — logits, not binary

            frame_objects: List[SegmentedObject] = []
            for i, oid in enumerate(all_obj_ids):
                if i >= pred_masks.shape[0]:
                    break
                mask_t = pred_masks[i]
                while mask_t.ndim > 2:
                    mask_t = mask_t.squeeze(0)
                mask_np = (mask_t > 0).cpu().numpy().astype(bool)
                area = int(mask_np.sum())
                if area == 0:
                    continue
                bbox = get_bbox_from_mask(mask_np)
                if bbox is None:
                    continue
                frame_objects.append(SegmentedObject(
                    obj_id=oid, mask=mask_np, score=1.0,
                    bbox=bbox, area=area,
                ))

            frames_dict[fidx] = FrameSegmentation(
                frame_idx=fidx, objects=frame_objects,
            )

        print(f"  Tracked {len(all_obj_ids)} objects across "
              f"{len(frames_dict)} frames")

        return VideoSegmentation(
            frames=frames_dict,
            object_ids=all_obj_ids,
            num_frames=len(video_frames),
        )

    # ------------------------------------------------------------------
    # Visualisation utilities
    # ------------------------------------------------------------------

    def visualize_frame(
        self,
        image: np.ndarray,
        frame_seg: FrameSegmentation,
        alpha: float = 0.5,
        draw_bbox: bool = True,
        draw_label: bool = True,
    ) -> np.ndarray:
        """
        Colorised overlay of all object masks on the frame.

        Args:
            image: (H, W, 3) uint8 RGB array.
            frame_seg: Segmentation result for this frame.
            alpha: Blending factor for the mask overlay.
            draw_bbox: Draw bounding box rectangles.
            draw_label: Draw obj_id labels.

        Returns:
            (H, W, 3) uint8 RGB array with overlaid masks.
        """
        vis = image.copy()
        overlay = np.zeros_like(vis)

        for obj in frame_seg.objects:
            color = _get_color(obj.obj_id).tolist()
            overlay[obj.mask] = color

        vis = cv2.addWeighted(vis, 1 - alpha, overlay, alpha, 0)

        if draw_bbox or draw_label:
            # Convert to BGR for cv2 drawing, then back
            vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            for obj in frame_seg.objects:
                color_bgr = _get_color(obj.obj_id)[::-1].tolist()
                x0, y0, x1, y1 = obj.bbox
                if draw_bbox:
                    cv2.rectangle(vis_bgr, (x0, y0), (x1, y1), color_bgr, 2)
                if draw_label:
                    cv2.putText(
                        vis_bgr, str(obj.obj_id),
                        (x0, max(y0 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2,
                    )
            vis = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)

        return vis

    def visualize_video(
        self,
        video_frames,
        video_seg: VideoSegmentation,
        output_path: str,
        fps: int = 30,
        alpha: float = 0.5,
    ) -> str:
        """
        Write an MP4 with colorised mask overlay on every frame.

        Args:
            video_frames: List of PIL Images.
            video_seg: Video segmentation result.
            output_path: Path for output MP4.
            fps: Frames per second.
            alpha: Blending factor.

        Returns:
            Path to the written video file.
        """
        first = np.array(video_frames[0])
        h, w = first.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for fidx, frame_pil in enumerate(video_frames):
            frame_np = np.array(frame_pil)
            if fidx in video_seg.frames:
                frame_np = self.visualize_frame(
                    frame_np, video_seg.frames[fidx], alpha=alpha,
                )
            writer.write(cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))

        writer.release()
        print(f"Saved visualisation to {output_path}")
        return output_path
