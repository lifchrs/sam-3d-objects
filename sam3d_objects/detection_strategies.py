"""
Object detection strategies for hand-object interaction in egocentric video.

This module implements multiple detection strategies to robustly extract masks
for objects held by hands, especially when objects may be partially occluded
or unclear in the first frame.

Strategies:
1. simple: Basic text prompt detection from frame 0
2. two_stage: Use hand detection as spatial anchor, then detect object
3. backward: Find best frame with clearest detection, track backward to frame 0
4. consensus: Run detection on multiple frames, use best detection
5. combined: Two-stage + backward tracking for maximum robustness
6. hand_point: Use VITRA hand joints as point prompts for spatial grounding
7. grounding_dino: Use GroundingDINO open-vocab detector for bounding box prompts
"""

import json
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


class DetectionLog:
    """Simple structured log accumulator for detection diagnostics."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.entries = []  # list of dicts

    def log(self, phase: str, message: str, **data):
        """Append a structured entry. Print if verbose."""
        entry = {"phase": phase, "message": message, **data}
        self.entries.append(entry)
        if self.verbose:
            print(f"  {message}")

    def to_dict(self) -> list:
        return self.entries


@dataclass
class DetectionResult:
    """Result from a detection strategy."""
    masks: Dict[int, np.ndarray]  # frame_idx -> mask (H, W) boolean
    object_ids: List[int]
    scores: Dict[int, float]  # frame_idx -> detection score
    best_frame_idx: int
    strategy_used: str
    metadata: Dict[str, Any]


def _seed_object_in_session(session, prompt_id: int, frame_idx: int,
                            mask_tensor: torch.Tensor, obj_id: int = 0,
                            keep_alive: int = 30):
    """Register a mask-seeded object in a SAM3 session with all required state.

    SAM3's _det_track_one_frame expects several state dicts to be populated
    for every tracked object (obj_id_to_prompt_id, trk_keep_alive, etc.).
    When objects are discovered via detection these are set automatically,
    but for manually seeded objects we must initialize them ourselves.
    """
    session.obj_id_to_idx(obj_id=obj_id)
    session.obj_id_to_prompt_id[obj_id] = prompt_id
    session.obj_first_frame_idx[obj_id] = frame_idx
    session.trk_keep_alive[obj_id] = keep_alive
    session.obj_id_to_score[obj_id] = 1.0  # high confidence seed
    if obj_id > session.max_obj_id:
        session.max_obj_id = obj_id
    session.add_mask_inputs(
        obj_idx=session._obj_id_to_idx[obj_id],
        frame_idx=frame_idx,
        inputs=mask_tensor,
    )
    session.obj_with_new_inputs = [obj_id]


def get_bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Extract bounding box from binary mask.

    Args:
        mask: Boolean mask (H, W)

    Returns:
        (x_min, y_min, x_max, y_max) or None if mask is empty
    """
    if mask.sum() == 0:
        return None

    y_indices, x_indices = np.where(mask)
    return (
        int(x_indices.min()),
        int(y_indices.min()),
        int(x_indices.max()),
        int(y_indices.max())
    )


def expand_bbox(bbox: Tuple[int, int, int, int], scale: float,
                img_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """
    Expand bounding box by a scale factor, clamping to image bounds.

    Args:
        bbox: (x_min, y_min, x_max, y_max)
        scale: Scale factor (e.g., 2.0 doubles the size)
        img_size: (height, width) of the image

    Returns:
        Expanded bbox (x_min, y_min, x_max, y_max)
    """
    x_min, y_min, x_max, y_max = bbox
    h, w = img_size

    # Calculate center and size
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    # Expand
    new_width = width * scale
    new_height = height * scale

    # New bbox
    new_x_min = max(0, int(cx - new_width / 2))
    new_y_min = max(0, int(cy - new_height / 2))
    new_x_max = min(w, int(cx + new_width / 2))
    new_y_max = min(h, int(cy + new_height / 2))

    return (new_x_min, new_y_min, new_x_max, new_y_max)


def crop_frame(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop frame to bounding box region."""
    x_min, y_min, x_max, y_max = bbox
    return frame[y_min:y_max, x_min:x_max]


def uncrop_mask(mask: np.ndarray, bbox: Tuple[int, int, int, int],
                orig_size: Tuple[int, int]) -> np.ndarray:
    """
    Map cropped mask back to full frame coordinates.

    Args:
        mask: Cropped mask
        bbox: (x_min, y_min, x_max, y_max) used for cropping
        orig_size: (height, width) of original frame

    Returns:
        Full-frame mask
    """
    x_min, y_min, x_max, y_max = bbox
    h, w = orig_size

    full_mask = np.zeros((h, w), dtype=mask.dtype)
    full_mask[y_min:y_max, x_min:x_max] = mask

    return full_mask


def score_detection(mask: np.ndarray, method: str = "mask_area") -> float:
    """
    Score a detection for quality/confidence.

    Args:
        mask: Boolean mask (H, W)
        method: Scoring method - "mask_area", "confidence", "hand_overlap"

    Returns:
        Score (higher is better)
    """
    if method == "mask_area":
        # Larger masks are generally better for held objects
        # Normalize by reasonable object size range
        area = mask.sum()
        # Score between 0-1, assuming reasonable object is 1-10% of frame
        total_pixels = mask.shape[0] * mask.shape[1]
        normalized_area = area / total_pixels
        # Penalize very small (<0.5%) or very large (>20%) masks
        if normalized_area < 0.005:
            return normalized_area * 10  # Low score for tiny masks
        elif normalized_area > 0.2:
            return 0.2 / normalized_area  # Penalize huge masks
        else:
            return normalized_area * 5  # Normal range

    elif method == "confidence":
        # Would need actual confidence scores from model
        # Fallback to mask area
        return mask.sum()

    elif method == "hand_overlap":
        # Would need hand mask to compute
        # Fallback to mask area
        return mask.sum()

    return float(mask.sum())


def run_sam3_detection(model, processor, inference_session, video_frames, prompt: str,
                       start_frame: int = 0, max_frames: Optional[int] = None,
                       reverse: bool = False):
    """
    Run SAM3 video detection with text prompt.

    Args:
        model: Sam3VideoModel
        processor: Sam3VideoProcessor
        inference_session: Initialized video session
        video_frames: List of PIL Images
        prompt: Text prompt for detection
        start_frame: Starting frame index
        max_frames: Maximum frames to track
        reverse: Whether to track in reverse

    Returns:
        Dict of frame_idx -> processed outputs
    """
    # Add text prompt
    inference_session = processor.add_text_prompt(
        inference_session=inference_session,
        text=prompt,
    )

    outputs_per_frame = {}

    for model_outputs in model.propagate_in_video_iterator(
        inference_session=inference_session,
        start_frame_idx=start_frame,
        max_frame_num_to_track=max_frames,
        reverse=reverse,
    ):
        processed_outputs = processor.postprocess_outputs(inference_session, model_outputs)
        outputs_per_frame[model_outputs.frame_idx] = processed_outputs

    return outputs_per_frame


def simple_detection(model, processor, video_frames, prompt: str, device,
                     dtype=torch.bfloat16) -> DetectionResult:
    """
    Strategy 1: Simple text prompt detection from frame 0.

    This is the baseline strategy - just use the text prompt to detect
    and track the object starting from frame 0.

    Args:
        model: Sam3VideoModel
        processor: Sam3VideoProcessor
        video_frames: List of PIL Images
        prompt: Text prompt for object detection
        device: Torch device
        dtype: Torch dtype

    Returns:
        DetectionResult with masks for all frames
    """
    # Initialize session
    inference_session = processor.init_video_session(
        video=video_frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=dtype,
    )

    outputs_per_frame = run_sam3_detection(
        model, processor, inference_session, video_frames, prompt
    )

    # Extract masks
    masks = {}
    scores = {}

    for frame_idx, results in outputs_per_frame.items():
        frame_masks = results['masks']
        if frame_masks.ndim == 4:
            frame_masks = frame_masks.squeeze(1)

        if len(frame_masks) > 0:
            # Take first object (highest confidence)
            mask = frame_masks[0].cpu().numpy()
            masks[frame_idx] = mask
            scores[frame_idx] = score_detection(mask)

    # Get object IDs
    obj_ids = outputs_per_frame[0]['object_ids'].tolist() if 0 in outputs_per_frame else []

    return DetectionResult(
        masks=masks,
        object_ids=obj_ids,
        scores=scores,
        best_frame_idx=0,
        strategy_used="simple",
        metadata={"prompt": prompt}
    )


def two_stage_detection(model, processor, video_frames, object_prompt: str,
                        device, dtype=torch.bfloat16,
                        bbox_scale: float = 2.0) -> DetectionResult:
    """
    Strategy 2: Two-stage detection (hand -> object).

    Use hand detection as spatial anchor, then detect object within
    expanded hand bounding box region.

    Args:
        model: Sam3VideoModel
        processor: Sam3VideoProcessor
        video_frames: List of PIL Images
        object_prompt: Text prompt for object detection
        device: Torch device
        dtype: Torch dtype
        bbox_scale: Scale factor for hand bounding box expansion

    Returns:
        DetectionResult with masks for all frames
    """
    # Stage 1: Detect hands
    print("  Stage 1: Detecting hands...")
    hand_session = processor.init_video_session(
        video=video_frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=dtype,
    )

    hand_outputs = run_sam3_detection(
        model, processor, hand_session, video_frames, "hand"
    )

    # Get hand bbox from frame 0
    if 0 not in hand_outputs or len(hand_outputs[0]['masks']) == 0:
        print("  Warning: No hand detected in frame 0, falling back to simple detection")
        return simple_detection(model, processor, video_frames, object_prompt, device, dtype)

    hand_mask = hand_outputs[0]['masks']
    if hand_mask.ndim == 4:
        hand_mask = hand_mask.squeeze(1)
    hand_mask_np = hand_mask[0].cpu().numpy()

    hand_bbox = get_bbox_from_mask(hand_mask_np)
    if hand_bbox is None:
        print("  Warning: Empty hand mask, falling back to simple detection")
        return simple_detection(model, processor, video_frames, object_prompt, device, dtype)

    # Expand hand bbox
    img_h, img_w = np.array(video_frames[0]).shape[:2]
    expanded_bbox = expand_bbox(hand_bbox, bbox_scale, (img_h, img_w))
    print(f"  Hand bbox: {hand_bbox} -> Expanded: {expanded_bbox}")

    # Stage 2: Detect object in cropped region
    print("  Stage 2: Detecting object in hand region...")

    # Crop all frames to expanded bbox
    from PIL import Image
    cropped_frames = []
    for frame in video_frames:
        frame_np = np.array(frame)
        cropped = crop_frame(frame_np, expanded_bbox)
        cropped_frames.append(Image.fromarray(cropped))

    # Run object detection on cropped frames
    obj_session = processor.init_video_session(
        video=cropped_frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=dtype,
    )

    obj_outputs = run_sam3_detection(
        model, processor, obj_session, cropped_frames, object_prompt
    )

    # Map masks back to full frame coordinates
    masks = {}
    scores = {}

    for frame_idx, results in obj_outputs.items():
        frame_masks = results['masks']
        if frame_masks.ndim == 4:
            frame_masks = frame_masks.squeeze(1)

        if len(frame_masks) > 0:
            cropped_mask = frame_masks[0].cpu().numpy()
            full_mask = uncrop_mask(cropped_mask, expanded_bbox, (img_h, img_w))
            masks[frame_idx] = full_mask
            scores[frame_idx] = score_detection(full_mask)

    obj_ids = obj_outputs[0]['object_ids'].tolist() if 0 in obj_outputs else []

    return DetectionResult(
        masks=masks,
        object_ids=obj_ids,
        scores=scores,
        best_frame_idx=0,
        strategy_used="two_stage",
        metadata={
            "prompt": object_prompt,
            "hand_bbox": hand_bbox,
            "expanded_bbox": expanded_bbox,
            "bbox_scale": bbox_scale
        }
    )


def backward_tracking(model, processor, video_frames, prompt: str,
                      device, dtype=torch.bfloat16,
                      score_metric: str = "mask_area") -> DetectionResult:
    """
    Strategy 3: Backward tracking from best frame.

    Find the frame with the clearest object detection, then track
    backward to frame 0 to get a more accurate initial mask.

    Args:
        model: Sam3VideoModel
        processor: Sam3VideoProcessor
        video_frames: List of PIL Images
        prompt: Text prompt for object detection
        device: Torch device
        dtype: Torch dtype
        score_metric: Metric for selecting best frame

    Returns:
        DetectionResult with masks for all frames
    """
    # Forward pass to find best frame
    print("  Pass 1: Forward detection to find best frame...")
    forward_session = processor.init_video_session(
        video=video_frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=dtype,
    )

    forward_outputs = run_sam3_detection(
        model, processor, forward_session, video_frames, prompt
    )

    # Score each frame and find best
    frame_scores = {}
    forward_masks = {}

    for frame_idx, results in forward_outputs.items():
        frame_masks = results['masks']
        if frame_masks.ndim == 4:
            frame_masks = frame_masks.squeeze(1)

        if len(frame_masks) > 0:
            mask = frame_masks[0].cpu().numpy()
            forward_masks[frame_idx] = mask
            frame_scores[frame_idx] = score_detection(mask, method=score_metric)

    if not frame_scores:
        print("  Warning: No detections found, returning empty result")
        return DetectionResult(
            masks={},
            object_ids=[],
            scores={},
            best_frame_idx=0,
            strategy_used="backward",
            metadata={"prompt": prompt, "error": "no_detections"}
        )

    # Find best frame
    best_frame_idx = max(frame_scores.keys(), key=lambda k: frame_scores[k])
    print(f"  Best frame: {best_frame_idx} (score: {frame_scores[best_frame_idx]:.4f})")

    if best_frame_idx == 0:
        # Best frame is already frame 0, just use forward results
        print("  Best frame is frame 0, using forward results")
        obj_ids = forward_outputs[0]['object_ids'].tolist() if 0 in forward_outputs else []
        return DetectionResult(
            masks=forward_masks,
            object_ids=obj_ids,
            scores=frame_scores,
            best_frame_idx=0,
            strategy_used="backward",
            metadata={"prompt": prompt, "forward_only": True}
        )

    # For backward tracking, we run detection on reversed video starting from best frame
    # This uses SAM3's built-in tracking to propagate backward
    print(f"  Pass 2: Backward tracking from frame {best_frame_idx} to frame 0...")

    # Create a reversed video (frames 0 to best_frame, reversed)
    # Use list() to avoid negative stride issues
    reversed_frames = list(video_frames[:best_frame_idx + 1])[::-1]

    # Initialize session on reversed frames
    backward_session = processor.init_video_session(
        video=reversed_frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=dtype,
    )

    # Run detection with the same prompt on reversed video
    # Frame 0 in reversed = best_frame_idx in original
    backward_session = processor.add_text_prompt(
        inference_session=backward_session,
        text=prompt,
    )

    backward_outputs = {}
    for model_outputs in model.propagate_in_video_iterator(
        inference_session=backward_session,
        start_frame_idx=0,
        max_frame_num_to_track=None,
    ):
        processed_outputs = processor.postprocess_outputs(backward_session, model_outputs)
        # Map back to original frame indices
        # reversed frame 0 = original best_frame_idx
        # reversed frame N = original (best_frame_idx - N)
        orig_frame_idx = best_frame_idx - model_outputs.frame_idx
        if orig_frame_idx >= 0:
            backward_outputs[orig_frame_idx] = processed_outputs

    # Combine results: use backward tracking for frames 0 to best_frame,
    # use forward tracking for frames after best_frame
    final_masks = {}
    final_scores = {}

    # Backward tracked frames (0 to best_frame)
    for frame_idx, results in backward_outputs.items():
        frame_masks = results['masks']
        if frame_masks.ndim == 4:
            frame_masks = frame_masks.squeeze(1)

        if len(frame_masks) > 0:
            mask = frame_masks[0].cpu().numpy()
            final_masks[frame_idx] = mask
            final_scores[frame_idx] = score_detection(mask, method=score_metric)

    # Forward tracked frames (after best_frame)
    for frame_idx, mask in forward_masks.items():
        if frame_idx > best_frame_idx:
            final_masks[frame_idx] = mask
            final_scores[frame_idx] = frame_scores[frame_idx]

    obj_ids = forward_outputs[0]['object_ids'].tolist() if 0 in forward_outputs else []

    return DetectionResult(
        masks=final_masks,
        object_ids=obj_ids,
        scores=final_scores,
        best_frame_idx=best_frame_idx,
        strategy_used="backward",
        metadata={
            "prompt": prompt,
            "score_metric": score_metric,
            "forward_scores": frame_scores
        }
    )


def consensus_detection(model, processor, video_frames, prompt: str,
                        device, dtype=torch.bfloat16,
                        sample_frames: Optional[List[int]] = None,
                        score_metric: str = "mask_area") -> DetectionResult:
    """
    Strategy 4: Multi-frame consensus.

    Run detection on multiple sample frames, find the frame with
    highest confidence, and propagate that detection to all frames.

    Args:
        model: Sam3VideoModel
        processor: Sam3VideoProcessor
        video_frames: List of PIL Images
        prompt: Text prompt for object detection
        device: Torch device
        dtype: Torch dtype
        sample_frames: Specific frames to sample (default: [0, N//4, N//2, 3N//4, N-1])
        score_metric: Metric for scoring detections

    Returns:
        DetectionResult with masks for all frames
    """
    n_frames = len(video_frames)

    # Default sample frames
    if sample_frames is None:
        sample_frames = [0, n_frames // 4, n_frames // 2, 3 * n_frames // 4, n_frames - 1]
        sample_frames = sorted(set(sample_frames))  # Remove duplicates, sort

    print(f"  Sampling frames: {sample_frames}")

    # Run detection on each sample frame
    sample_scores = {}
    sample_masks = {}

    for frame_idx in sample_frames:
        print(f"  Detecting in frame {frame_idx}...")

        # Initialize session starting from this frame
        session = processor.init_video_session(
            video=video_frames,
            inference_device=device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=dtype,
        )

        session = processor.add_text_prompt(
            inference_session=session,
            text=prompt,
        )

        # Just get detection for this specific frame
        for model_outputs in model.propagate_in_video_iterator(
            inference_session=session,
            start_frame_idx=frame_idx,
            max_frame_num_to_track=1,  # Just one frame
        ):
            processed_outputs = processor.postprocess_outputs(session, model_outputs)

            frame_masks = processed_outputs['masks']
            if frame_masks.ndim == 4:
                frame_masks = frame_masks.squeeze(1)

            if len(frame_masks) > 0:
                mask = frame_masks[0].cpu().numpy()
                sample_masks[frame_idx] = mask
                sample_scores[frame_idx] = score_detection(mask, method=score_metric)
                print(f"    Frame {frame_idx}: score = {sample_scores[frame_idx]:.4f}")
            break

    if not sample_scores:
        print("  Warning: No detections in any sample frame")
        return DetectionResult(
            masks={},
            object_ids=[],
            scores={},
            best_frame_idx=0,
            strategy_used="consensus",
            metadata={"prompt": prompt, "error": "no_detections"}
        )

    # Find best sample frame
    best_frame_idx = max(sample_scores.keys(), key=lambda k: sample_scores[k])
    print(f"  Best sample frame: {best_frame_idx} (score: {sample_scores[best_frame_idx]:.4f})")

    # Propagate bidirectionally from best frame, with forward-from-0 fallback.
    # Pass 1: Forward from best_frame to end of video
    # Pass 2: Backward on same session (tracker has object in memory)
    # Pass 3: Forward from frame 0 as fallback for any uncovered frames
    print(f"  Pass 1: Forward from best frame {best_frame_idx}...")

    def _collect_masks(model, processor, session, start, reverse=False):
        masks = {}
        for out in model.propagate_in_video_iterator(
            inference_session=session, start_frame_idx=start, reverse=reverse,
        ):
            processed = processor.postprocess_outputs(session, out)
            fm = processed['masks']
            if fm.ndim == 4:
                fm = fm.squeeze(1)
            if len(fm) > 0:
                masks[out.frame_idx] = fm[0].cpu().numpy()
        return masks

    bidir_session = processor.init_video_session(
        video=video_frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=dtype,
    )
    bidir_session = processor.add_text_prompt(
        inference_session=bidir_session,
        text=prompt,
    )

    forward_masks = _collect_masks(model, processor, bidir_session, best_frame_idx, reverse=False)
    print(f"    {len(forward_masks)} frames")

    # Pass 2: Backward on same session (no reset — tracker retains object memory)
    backward_masks = {}
    if best_frame_idx > 0:
        print(f"  Pass 2: Backward from frame {best_frame_idx}...")
        backward_masks = _collect_masks(model, processor, bidir_session, best_frame_idx, reverse=True)
        print(f"    {len(backward_masks)} frames")

    # Pass 3: Forward from 0 as fallback for frames not covered above
    print(f"  Pass 3: Forward from frame 0 (fallback)...")
    fallback_session = processor.init_video_session(
        video=video_frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=dtype,
    )
    fallback_session = processor.add_text_prompt(
        inference_session=fallback_session,
        text=prompt,
    )
    fallback_masks = _collect_masks(model, processor, fallback_session, 0, reverse=False)
    print(f"    {len(fallback_masks)} frames")

    # Merge: forward-from-best > backward > forward-from-0
    final_masks = {**fallback_masks, **backward_masks, **forward_masks}
    print(f"  Total: {len(final_masks)} frames with masks")
    final_scores = {idx: score_detection(m, method=score_metric) for idx, m in final_masks.items()}

    return DetectionResult(
        masks=final_masks,
        object_ids=[0],
        scores=final_scores,
        best_frame_idx=best_frame_idx,
        strategy_used="consensus",
        metadata={
            "prompt": prompt,
            "sample_frames": sample_frames,
            "sample_scores": sample_scores,
            "score_metric": score_metric
        }
    )


def _masks_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    intersection = (mask1.astype(bool) & mask2.astype(bool)).sum()
    union = (mask1.astype(bool) | mask2.astype(bool)).sum()
    return float(intersection / union) if union > 0 else 0.0


def _generate_prompt_variations(prompt: str) -> List[str]:
    """Generate diverse prompt variations to find different objects."""
    variations = []

    # Size-based variations
    variations.append("large object in hand")
    variations.append("small object in hand")

    # Action-based variations
    variations.append("thing being manipulated")
    variations.append("item being held")

    # Remove duplicates and the original prompt
    prompt_lower = prompt.lower().strip()
    return [v for v in variations if v.lower().strip() != prompt_lower]


def multi_object_consensus_detection(
    model, processor, video_frames, prompt: str,
    device, dtype=torch.bfloat16,
    max_objects: int = 3,
    sample_frames: Optional[List[int]] = None,
    score_metric: str = "mask_area",
    min_mask_area_pct: float = 0.005,
    additional_prompts: Optional[List[str]] = None,
    iou_dedup_threshold: float = 0.5,
) -> List[DetectionResult]:
    """
    Multi-object detection via prompt cascade + de-duplication.

    Tries the primary prompt and several variations. Each prompt may find a
    different object. Results are de-duplicated by mask IoU so that only
    distinct objects are returned.

    Args:
        model: Sam3VideoModel
        processor: Sam3VideoProcessor
        video_frames: List of PIL Images
        prompt: Text prompt for object detection
        device: Torch device
        dtype: Torch dtype
        max_objects: Maximum number of objects to return
        sample_frames: Specific frames to sample (default: auto)
        score_metric: Metric for scoring detections
        min_mask_area_pct: Minimum mask area as fraction of frame
        additional_prompts: Extra prompts to try (default: auto-generated variations)
        iou_dedup_threshold: IoU threshold above which two detections are considered the same object

    Returns:
        List of DetectionResult, one per detected object, sorted by avg score (best first)
    """
    n_frames = len(video_frames)
    img_h, img_w = np.array(video_frames[0]).shape[:2]
    total_pixels = img_h * img_w

    if sample_frames is None:
        sample_frames = [0, n_frames // 4, n_frames // 2, 3 * n_frames // 4, n_frames - 1]
        sample_frames = sorted(set(sample_frames))

    # Build prompt list
    if additional_prompts is None:
        additional_prompts = _generate_prompt_variations(prompt)
    all_prompts = [prompt] + additional_prompts

    print(f"  Multi-object detection: up to {max_objects} objects")
    print(f"  Prompts to try: {all_prompts}")

    # Phase 1: Quick sampling — for each prompt, detect on sample frames
    # to find which (prompt, frame) pairs yield good detections
    candidates = []  # (prompt, best_frame_idx, best_mask, best_score)

    for p in all_prompts:
        p_best_frame = -1
        p_best_mask = None
        p_best_score = -1

        for frame_idx in sample_frames:
            session = processor.init_video_session(
                video=video_frames,
                inference_device=device,
                processing_device="cpu",
                video_storage_device="cpu",
                dtype=dtype,
            )
            session = processor.add_text_prompt(inference_session=session, text=p)

            for model_outputs in model.propagate_in_video_iterator(
                inference_session=session,
                start_frame_idx=frame_idx,
                max_frame_num_to_track=1,
            ):
                processed = processor.postprocess_outputs(session, model_outputs)
                frame_masks = processed['masks']
                if frame_masks.ndim == 4:
                    frame_masks = frame_masks.squeeze(1)
                if len(frame_masks) > 0:
                    mask = frame_masks[0].cpu().numpy()
                    area_pct = mask.sum() / total_pixels
                    if area_pct < min_mask_area_pct:
                        continue
                    sc = score_detection(mask, method=score_metric)
                    if sc > p_best_score:
                        p_best_score = sc
                        p_best_frame = frame_idx
                        p_best_mask = mask
                break

        if p_best_mask is not None:
            print(f"    Prompt '{p}': best_frame={p_best_frame}, score={p_best_score:.4f}, "
                  f"area={p_best_mask.sum()/total_pixels:.4f}")
            candidates.append((p, p_best_frame, p_best_mask, p_best_score))
        else:
            print(f"    Prompt '{p}': no valid detection")

    if not candidates:
        print("  Warning: No valid detections with any prompt")
        return []

    # Phase 2: De-duplicate candidates by mask IoU (greedy: keep higher scoring first)
    candidates.sort(key=lambda x: x[3], reverse=True)  # sort by score descending
    unique_candidates = []

    for cand in candidates:
        is_dup = False
        for existing in unique_candidates:
            # Compare masks — need to handle different best frames
            # Use the overlapping frame if possible, otherwise compare at each candidate's best frame
            iou = _masks_iou(cand[2], existing[2])
            if iou > iou_dedup_threshold:
                is_dup = True
                break
        if not is_dup:
            unique_candidates.append(cand)
            if len(unique_candidates) >= max_objects:
                break

    print(f"  Unique objects found: {len(unique_candidates)}")

    # Phase 3: Full propagation for each unique object
    results = []

    for obj_idx, (p, best_frame, _, _) in enumerate(unique_candidates):
        print(f"  Propagating object {obj_idx} (prompt='{p}', start_frame={best_frame})...")
        result = consensus_detection(
            model, processor, video_frames, p,
            device, dtype, sample_frames=[best_frame],
            score_metric=score_metric,
        )

        if result.masks:
            # Filter frames below min area
            filtered_masks = {}
            for fidx, mask in result.masks.items():
                if mask.sum() / total_pixels >= min_mask_area_pct:
                    filtered_masks[fidx] = mask
            result.masks = filtered_masks
            result.scores = {k: score_detection(v, method=score_metric) for k, v in filtered_masks.items()}

            if filtered_masks:
                avg_score = float(np.mean(list(result.scores.values())))
                result.strategy_used = "multi_object_consensus"
                result.metadata["prompt_used"] = p
                result.metadata["object_index"] = obj_idx
                result.metadata["avg_score"] = avg_score
                print(f"    Object {obj_idx}: {len(filtered_masks)} frames, avg_score={avg_score:.4f}")
                results.append(result)

    # Sort by average score
    results.sort(key=lambda r: r.metadata.get("avg_score", 0), reverse=True)
    print(f"  Final: {len(results)} objects")
    return results


def combined_detection(model, processor, video_frames, object_prompt: str,
                       device, dtype=torch.bfloat16,
                       bbox_scale: float = 2.0,
                       score_metric: str = "mask_area") -> DetectionResult:
    """
    Strategy 5: Combined two-stage + backward tracking.

    Use hand bbox for spatial constraint, find best frame within that region,
    and track backward to frame 0.

    Args:
        model: Sam3VideoModel
        processor: Sam3VideoProcessor
        video_frames: List of PIL Images
        object_prompt: Text prompt for object detection
        device: Torch device
        dtype: Torch dtype
        bbox_scale: Scale factor for hand bounding box expansion
        score_metric: Metric for selecting best frame

    Returns:
        DetectionResult with masks for all frames
    """
    # Stage 1: Detect hands in all frames
    print("  Stage 1: Detecting hands in all frames...")
    hand_session = processor.init_video_session(
        video=video_frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=dtype,
    )

    hand_outputs = run_sam3_detection(
        model, processor, hand_session, video_frames, "hand"
    )

    # Get hand bboxes for each frame
    hand_bboxes = {}
    img_h, img_w = np.array(video_frames[0]).shape[:2]

    for frame_idx, results in hand_outputs.items():
        frame_masks = results['masks']
        if frame_masks.ndim == 4:
            frame_masks = frame_masks.squeeze(1)

        if len(frame_masks) > 0:
            hand_mask = frame_masks[0].cpu().numpy()
            bbox = get_bbox_from_mask(hand_mask)
            if bbox is not None:
                hand_bboxes[frame_idx] = expand_bbox(bbox, bbox_scale, (img_h, img_w))

    if not hand_bboxes:
        print("  Warning: No hands detected, falling back to backward tracking")
        return backward_tracking(model, processor, video_frames, object_prompt,
                                device, dtype, score_metric)

    print(f"  Found hands in {len(hand_bboxes)} frames")

    # Stage 2: Run object detection in hand regions for each frame
    print("  Stage 2: Detecting objects in hand regions...")
    frame_detections = {}
    frame_scores = {}

    for frame_idx, bbox in hand_bboxes.items():
        # Crop frame to hand region
        from PIL import Image
        frame_np = np.array(video_frames[frame_idx])
        cropped = crop_frame(frame_np, bbox)
        cropped_pil = Image.fromarray(cropped)

        # Run single-frame detection
        session = processor.init_video_session(
            video=[cropped_pil],
            inference_device=device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=dtype,
        )

        session = processor.add_text_prompt(
            inference_session=session,
            text=object_prompt,
        )

        for model_outputs in model.propagate_in_video_iterator(
            inference_session=session,
            max_frame_num_to_track=1,
        ):
            processed_outputs = processor.postprocess_outputs(session, model_outputs)
            frame_masks = processed_outputs['masks']
            if frame_masks.ndim == 4:
                frame_masks = frame_masks.squeeze(1)

            if len(frame_masks) > 0:
                cropped_mask = frame_masks[0].cpu().numpy()
                full_mask = uncrop_mask(cropped_mask, bbox, (img_h, img_w))
                frame_detections[frame_idx] = full_mask
                frame_scores[frame_idx] = score_detection(full_mask, method=score_metric)
            break

    if not frame_scores:
        print("  Warning: No objects detected in hand regions, falling back to backward tracking")
        return backward_tracking(model, processor, video_frames, object_prompt,
                                device, dtype, score_metric)

    # Find best frame
    best_frame_idx = max(frame_scores.keys(), key=lambda k: frame_scores[k])
    print(f"  Best frame: {best_frame_idx} (score: {frame_scores[best_frame_idx]:.4f})")

    # Stage 3: Track from best frame to all frames
    print(f"  Stage 3: Propagating from best frame {best_frame_idx}...")

    best_mask = frame_detections[best_frame_idx]
    best_mask_tensor = torch.from_numpy(best_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    # Forward propagation
    full_session = processor.init_video_session(
        video=video_frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=dtype,
    )

    full_session = processor.add_text_prompt(
        inference_session=full_session,
        text=object_prompt,
    )

    prompt_id = full_session.add_prompt(object_prompt)
    _seed_object_in_session(full_session, prompt_id, best_frame_idx,
                            best_mask_tensor.to(device))

    forward_masks = {}
    for model_outputs in model.propagate_in_video_iterator(
        inference_session=full_session,
        start_frame_idx=best_frame_idx,
        reverse=False,
    ):
        processed_outputs = processor.postprocess_outputs(full_session, model_outputs)
        frame_masks = processed_outputs['masks']
        if frame_masks.ndim == 4:
            frame_masks = frame_masks.squeeze(1)
        if len(frame_masks) > 0:
            forward_masks[model_outputs.frame_idx] = frame_masks[0].cpu().numpy()

    # Backward propagation
    backward_masks = {}
    if best_frame_idx > 0:
        full_session.reset_tracking_data()
        _seed_object_in_session(full_session, prompt_id, best_frame_idx,
                                best_mask_tensor.to(device))

        for model_outputs in model.propagate_in_video_iterator(
            inference_session=full_session,
            start_frame_idx=best_frame_idx,
            reverse=True,
        ):
            processed_outputs = processor.postprocess_outputs(full_session, model_outputs)
            frame_masks = processed_outputs['masks']
            if frame_masks.ndim == 4:
                frame_masks = frame_masks.squeeze(1)
            if len(frame_masks) > 0:
                backward_masks[model_outputs.frame_idx] = frame_masks[0].cpu().numpy()

    # Combine
    final_masks = {**backward_masks, **forward_masks}

    # Constrain to hand regions where available
    for frame_idx in final_masks:
        if frame_idx in hand_bboxes:
            bbox = hand_bboxes[frame_idx]
            # Create hand region mask
            hand_region = np.zeros((img_h, img_w), dtype=bool)
            x_min, y_min, x_max, y_max = bbox
            hand_region[y_min:y_max, x_min:x_max] = True
            # Intersect with object mask
            final_masks[frame_idx] = final_masks[frame_idx] & hand_region

    final_scores = {idx: score_detection(m, method=score_metric) for idx, m in final_masks.items()}

    return DetectionResult(
        masks=final_masks,
        object_ids=[0],
        scores=final_scores,
        best_frame_idx=best_frame_idx,
        strategy_used="combined",
        metadata={
            "prompt": object_prompt,
            "bbox_scale": bbox_scale,
            "score_metric": score_metric,
            "num_hand_frames": len(hand_bboxes),
            "frame_scores": frame_scores
        }
    )


def project_joints_to_2d(joints_3d: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """
    Project 3D camera-space points to 2D pixel coordinates.

    Args:
        joints_3d: (N, 3) array of 3D points in camera space
        intrinsics: (3, 3) camera intrinsic matrix [fx, 0, cx; 0, fy, cy; 0, 0, 1]

    Returns:
        (N, 2) array of 2D pixel coordinates (u, v)
    """
    X, Y, Z = joints_3d[:, 0], joints_3d[:, 1], joints_3d[:, 2]

    # Avoid division by zero
    Z = np.clip(Z, 1e-6, None)

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Perspective projection: u = fx * X/Z + cx, v = fy * Y/Z + cy
    u = fx * X / Z + cx
    v = fy * Y / Z + cy

    return np.stack([u, v], axis=1)


def get_fingertip_centroid(joints_2d: np.ndarray) -> Tuple[float, float]:
    """
    Get centroid of fingertip positions.

    MANO joint indices:
    - 0: Wrist
    - 4: Thumb tip
    - 8: Index fingertip
    - 12: Middle fingertip
    - 16: Ring fingertip
    - 20: Pinky fingertip

    Args:
        joints_2d: (21, 2) array of 2D joint positions

    Returns:
        (x, y) tuple of fingertip centroid coordinates
    """
    fingertip_indices = [4, 8, 12, 16, 20]
    fingertips = joints_2d[fingertip_indices]
    centroid = fingertips.mean(axis=0)
    return float(centroid[0]), float(centroid[1])


def get_hand_center(joints_2d: np.ndarray) -> Tuple[float, float]:
    """
    Get the center of the hand (average of all joint positions).

    Args:
        joints_2d: (21, 2) array of 2D joint positions

    Returns:
        (x, y) tuple of hand center coordinates
    """
    center = joints_2d.mean(axis=0)
    return float(center[0]), float(center[1])


def create_point_mask(point_x: float, point_y: float, img_size: Tuple[int, int],
                      radius: int = 5) -> np.ndarray:
    """
    Create a small circular mask centered at a point.

    This is used to provide a "seed" mask when point prompts aren't directly
    available in the SAM3 API. The small mask serves as initialization for
    object detection/tracking.

    Args:
        point_x: X coordinate of the point
        point_y: Y coordinate of the point
        img_size: (height, width) of the image
        radius: Radius of the circular mask in pixels

    Returns:
        Boolean mask (H, W) with True values in a circle around the point
    """
    h, w = img_size
    mask = np.zeros((h, w), dtype=bool)

    # Create coordinate grids
    y_coords, x_coords = np.ogrid[:h, :w]

    # Create circular mask
    distance_sq = (x_coords - point_x) ** 2 + (y_coords - point_y) ** 2
    mask[distance_sq <= radius ** 2] = True

    return mask


def get_hand_bbox_from_joints(joints_2d: np.ndarray, img_size: Tuple[int, int],
                               padding: int = 50) -> Tuple[int, int, int, int]:
    """
    Compute a bounding box from all 21 projected 2D hand joints with pixel padding.

    Args:
        joints_2d: (21, 2) array of 2D joint positions
        img_size: (height, width) of the image
        padding: Pixel padding around the joint extremes

    Returns:
        (x_min, y_min, x_max, y_max) bounding box
    """
    h, w = img_size
    x_min = max(0, int(np.min(joints_2d[:, 0]) - padding))
    y_min = max(0, int(np.min(joints_2d[:, 1]) - padding))
    x_max = min(w, int(np.max(joints_2d[:, 0]) + padding))
    y_max = min(h, int(np.max(joints_2d[:, 1]) + padding))
    return (x_min, y_min, x_max, y_max)


def score_detection_hand_overlap(mask: np.ndarray, hand_bbox: Tuple[int, int, int, int],
                                  fingertip_centroid: Tuple[float, float],
                                  img_size: Tuple[int, int]) -> float:
    """
    Score a detection mask by how much it overlaps with the hand region.

    Combines:
    - IoU: intersection / union of mask and hand region
    - Mask-in-hand ratio: intersection / mask_area (penalizes masks far from hand)
    - Fingertip bonus: +0.3 if mask contains the fingertip centroid pixel
    - Size quality: penalty for tiny (<0.5% of frame) or huge (>25%) masks

    Args:
        mask: Boolean mask (H, W)
        hand_bbox: (x_min, y_min, x_max, y_max) hand bounding box
        fingertip_centroid: (x, y) fingertip centroid coordinates
        img_size: (height, width) of the image

    Returns:
        Combined score (higher is better, >1.0 is excellent)
    """
    h, w = img_size
    mask_area = mask.sum()
    if mask_area == 0:
        return 0.0

    # Build hand region mask from bbox
    x_min, y_min, x_max, y_max = hand_bbox
    hand_region = np.zeros((h, w), dtype=bool)
    hand_region[y_min:y_max, x_min:x_max] = True
    hand_region_area = hand_region.sum()

    if hand_region_area == 0:
        return 0.0

    # Intersection
    intersection = np.sum(mask & hand_region)

    # Overlap: IoU balances both small and large masks against hand region
    union = np.sum(mask | hand_region)
    overlap_coeff = intersection / union if union > 0 else 0.0

    # Mask-in-hand ratio: penalizes masks far from hand
    mask_in_hand = intersection / mask_area

    # Fingertip bonus
    centroid_bonus = 0.0
    cx, cy = int(round(fingertip_centroid[0])), int(round(fingertip_centroid[1]))
    if 0 <= cy < h and 0 <= cx < w:
        if mask[cy, cx]:
            centroid_bonus = 0.3

    # Size quality
    total_pixels = h * w
    normalized_area = mask_area / total_pixels
    if normalized_area < 0.005:
        quality = normalized_area / 0.005  # Ramp from 0 to 1
    elif normalized_area > 0.25:
        quality = 0.25 / normalized_area  # Penalize huge masks
    else:
        quality = 1.0

    score = (0.6 * overlap_coeff + 0.4 * mask_in_hand) * quality + centroid_bonus
    return score


def hand_point_detection(model, processor, video_frames, prompt: str,
                         hand_joints: np.ndarray, intrinsics: np.ndarray,
                         kept_frames: Optional[np.ndarray] = None,
                         device=None, dtype=torch.bfloat16,
                         use_fingertips: bool = True,
                         score_metric: str = "mask_area",
                         num_sample_frames: int = 10,
                         use_llm: bool = False,
                         early_exit_score: float = 0.8,
                         verbose: bool = True) -> DetectionResult:
    """
    Strategy 6: Use VITRA hand joints + multi-frame/multi-prompt detection.

    Uses SAM3 text detection with a cascade of prompts (specific -> generic),
    VITRA hand joint projections for spatial filtering via hand bounding box
    overlap scoring, and multi-frame sampling to find the best detection.

    Algorithm:
    1. Generate prompt cascade from input prompt (specific -> generic)
    2. Sample frames evenly from all valid hand frames
    3. For each (frame, prompt) pair, run SAM3 text detection and score
       all returned masks by hand-overlap quality
    4. If no text prompt works, try cropping to hand bbox and running
       generic prompts on the crop
    5. Propagate best mask bidirectionally from best frame

    Args:
        model: Sam3VideoModel
        processor: Sam3VideoProcessor
        video_frames: List of PIL Images
        prompt: Text prompt for object detection
        hand_joints: (T, 21, 3) array of 3D hand joints in camera space
        intrinsics: (3, 3) camera intrinsic matrix
        kept_frames: Optional array indicating valid frames (1=valid, 0=invalid)
        device: Torch device
        dtype: Torch dtype
        use_fingertips: If True, use fingertip centroid; if False, use hand center
        score_metric: Metric for scoring detections
        num_sample_frames: Number of frames to sample for detection

    Returns:
        DetectionResult with masks for all frames
    """
    from text_utils import get_detection_prompts
    from PIL import Image

    log = DetectionLog(verbose=verbose)

    if device is None:
        device = next(model.parameters()).device

    n_video_frames = len(video_frames)
    n_joint_frames = hand_joints.shape[0]
    img_h, img_w = np.array(video_frames[0]).shape[:2]
    img_size = (img_h, img_w)

    # Scale intrinsics to match actual frame resolution.
    # VITRA intrinsics are for the original camera resolution (e.g. 1920x1080
    # for EPIC-Kitchens) but video_frames may be resized by TTT3R (e.g. 512x288).
    orig_w = intrinsics[0, 2] * 2  # cx ≈ width/2
    orig_h = intrinsics[1, 2] * 2  # cy ≈ height/2
    if orig_w > 0 and orig_h > 0:
        sx = img_w / orig_w
        sy = img_h / orig_h
        if abs(sx - 1.0) > 0.01 or abs(sy - 1.0) > 0.01:
            intrinsics = intrinsics.copy()
            intrinsics[0, 0] *= sx  # fx
            intrinsics[0, 2] *= sx  # cx
            intrinsics[1, 1] *= sy  # fy
            intrinsics[1, 2] *= sy  # cy
            log.log("setup",
                     f"Scaled intrinsics to frame size: {orig_w:.0f}x{orig_h:.0f} -> {img_w}x{img_h} "
                     f"(sx={sx:.3f}, sy={sy:.3f})")

    # --- Step 0: Generate prompt cascade ---
    prompt_cascade = get_detection_prompts(prompt, use_llm=use_llm)
    log.log("setup", f"Prompt cascade: {prompt_cascade}",
            prompt_cascade=prompt_cascade)

    # --- Step 1: Sample frames from the latter 70% of valid frames ---
    if kept_frames is not None:
        valid_indices = np.where(np.array(kept_frames) == 1)[0]
    else:
        valid_indices = np.arange(n_joint_frames)

    if len(valid_indices) == 0:
        log.log("setup", "Warning: No valid hand frames found, using all joint frames")
        valid_indices = np.arange(n_joint_frames)

    # Use all valid frames as candidates
    candidate_indices = valid_indices

    # Evenly sample num_sample_frames from candidates
    n_samples = min(num_sample_frames, len(candidate_indices))
    if n_samples <= 1:
        sample_positions = [len(candidate_indices) - 1]
    else:
        sample_positions = np.linspace(0, len(candidate_indices) - 1, n_samples, dtype=int)
    sample_frame_indices = [int(candidate_indices[i]) for i in sample_positions]

    # Clamp to video and joint bounds
    sample_frame_indices = [
        min(idx, n_video_frames - 1, n_joint_frames - 1)
        for idx in sample_frame_indices
    ]
    sample_frame_indices = sorted(set(sample_frame_indices))
    log.log("setup", f"Sample frames: {sample_frame_indices}",
            num_valid_frames=len(valid_indices),
            num_sample_frames=len(sample_frame_indices),
            sample_frame_indices=sample_frame_indices)

    # --- Step 2: Project hand joints per frame ---
    frame_hand_data = {}  # frame_idx -> {joints_2d, centroid, bbox}
    for fidx in sample_frame_indices:
        joints_3d = hand_joints[fidx]
        joints_2d = project_joints_to_2d(joints_3d, intrinsics)

        if use_fingertips:
            cx, cy = get_fingertip_centroid(joints_2d)
        else:
            cx, cy = get_hand_center(joints_2d)

        cx = max(0, min(cx, img_w - 1))
        cy = max(0, min(cy, img_h - 1))

        hand_bbox = get_hand_bbox_from_joints(joints_2d, img_size, padding=50)

        frame_hand_data[fidx] = {
            'joints_2d': joints_2d,
            'centroid': (cx, cy),
            'bbox': hand_bbox,
        }

    # --- Step 3: Multi-frame + multi-prompt detection ---
    best_result = {'frame_idx': None, 'mask': None, 'score': -1.0, 'prompt': None}

    for fidx in sample_frame_indices:
        hdata = frame_hand_data[fidx]
        log.log("frame_projection",
                f"Frame {fidx}: centroid=({hdata['centroid'][0]:.0f}, {hdata['centroid'][1]:.0f}), bbox={hdata['bbox']}",
                frame_idx=fidx,
                centroid=list(hdata['centroid']),
                bbox=list(hdata['bbox']))

        for text_prompt in prompt_cascade:
            # Init fresh session for each (frame, prompt) pair
            session = processor.init_video_session(
                video=video_frames,
                inference_device=device,
                processing_device="cpu",
                video_storage_device="cpu",
                dtype=dtype,
            )
            session = processor.add_text_prompt(
                inference_session=session,
                text=text_prompt,
            )

            # Get detection for this frame only
            for model_outputs in model.propagate_in_video_iterator(
                inference_session=session,
                start_frame_idx=fidx,
                max_frame_num_to_track=1,
            ):
                processed = processor.postprocess_outputs(session, model_outputs)
                frame_masks = processed['masks']
                if frame_masks.ndim == 4:
                    frame_masks = frame_masks.squeeze(1)

                # Score ALL returned masks
                for mask_idx in range(len(frame_masks)):
                    mask_np = frame_masks[mask_idx].cpu().numpy()
                    if mask_np.sum() == 0:
                        continue

                    sc = score_detection_hand_overlap(
                        mask_np, hdata['bbox'], hdata['centroid'], img_size
                    )

                    if sc > best_result['score']:
                        best_result = {
                            'frame_idx': fidx,
                            'mask': mask_np,
                            'score': sc,
                            'prompt': text_prompt,
                        }
                        log.log("text_detection",
                                f"  New best: prompt='{text_prompt}', mask_idx={mask_idx}, score={sc:.3f}",
                                frame_idx=fidx, prompt=text_prompt,
                                mask_idx=mask_idx, score=sc, is_new_best=True)
                break  # only need 1 iteration from propagate

            # Early exit if score is excellent (skip remaining prompts for this frame)
            if best_result['score'] > 1.0:
                log.log("text_detection",
                        f"  Excellent score {best_result['score']:.3f}, skipping remaining prompts",
                        score=best_result['score'])
                break

        # Early exit across frames if score exceeds threshold
        if best_result['score'] >= early_exit_score:
            log.log("text_detection",
                    f"  Score {best_result['score']:.3f} >= {early_exit_score}, skipping remaining frames",
                    score=best_result['score'], threshold=early_exit_score)
            break

    log.log("text_detection_summary",
            f"After text detection: best_score={best_result['score']:.3f}, "
            f"prompt='{best_result['prompt']}', frame={best_result['frame_idx']}",
            best_score=best_result['score'],
            best_prompt=best_result['prompt'],
            best_frame=best_result['frame_idx'])

    # --- Step 4: Crop fallback if text detection score is too low ---
    CROP_FALLBACK_THRESHOLD = 0.15
    if best_result['score'] < CROP_FALLBACK_THRESHOLD:
        log.log("crop_fallback",
                f"Step 4: Text detection score {best_result['score']:.3f} < {CROP_FALLBACK_THRESHOLD}, trying crop fallback...")
        # Use specific prompts from cascade first, then generic fallbacks
        crop_prompts = [p for p in prompt_cascade if p not in ("grasped object", "held object")]
        crop_prompts.extend(["object held in hand", "object"])

        for fidx in sample_frame_indices:
            hdata = frame_hand_data[fidx]
            # Expand hand bbox by 1.5x for crop
            expanded_bbox = expand_bbox(hdata['bbox'], 1.5, img_size)
            frame_np = np.array(video_frames[fidx])
            cropped = crop_frame(frame_np, expanded_bbox)
            if cropped.size == 0 or cropped.shape[0] < 2 or cropped.shape[1] < 2:
                log.log("crop_fallback",
                        f"  Skipping frame {fidx}: crop is empty/too small "
                        f"(bbox={expanded_bbox}, shape={cropped.shape})")
                continue
            cropped_pil = Image.fromarray(cropped)

            for cp in crop_prompts:
                crop_session = processor.init_video_session(
                    video=[cropped_pil],
                    inference_device=device,
                    processing_device="cpu",
                    video_storage_device="cpu",
                    dtype=dtype,
                )
                crop_session = processor.add_text_prompt(
                    inference_session=crop_session,
                    text=cp,
                )

                for model_outputs in model.propagate_in_video_iterator(
                    inference_session=crop_session,
                    start_frame_idx=0,
                    max_frame_num_to_track=1,
                ):
                    processed = processor.postprocess_outputs(crop_session, model_outputs)
                    frame_masks = processed['masks']
                    if frame_masks.ndim == 4:
                        frame_masks = frame_masks.squeeze(1)

                    for mask_idx in range(len(frame_masks)):
                        cropped_mask = frame_masks[mask_idx].cpu().numpy()
                        if cropped_mask.sum() == 0:
                            continue
                        # Uncrop back to full frame
                        full_mask = uncrop_mask(cropped_mask, expanded_bbox, img_size)
                        sc = score_detection_hand_overlap(
                            full_mask, hdata['bbox'], hdata['centroid'], img_size
                        )
                        if sc > best_result['score']:
                            best_result = {
                                'frame_idx': fidx,
                                'mask': full_mask,
                                'score': sc,
                                'prompt': f"crop:{cp}",
                            }
                            log.log("crop_fallback",
                                    f"  Crop fallback hit: frame={fidx}, prompt='{cp}', score={sc:.3f}",
                                    frame_idx=fidx, prompt=cp, score=sc)
                    break

        log.log("crop_fallback_summary",
                f"After crop fallback: best_score={best_result['score']:.3f}",
                best_score=best_result['score'])

    # Step 4b: Point prompt fallback (currently disabled)
    # If still nothing, use SAM2 point prompting at the fingertip as fallback.
    # This runs SAM2's mask decoder with a positive point beyond the fingertip
    # to get a proper object segmentation.
    # TODO: re-enable if needed — commented out to simplify the pipeline
    # if best_result['mask'] is None or best_result['score'] <= 0.0:
    #     ... (point prompt fallback code)

    if best_result['mask'] is None or best_result['score'] <= 0.0:
        log.log("detection_failed",
                "Warning: All detection steps failed (text, crop). No mask found.")
        return DetectionResult(
            masks={}, object_ids=[], scores={},
            best_frame_idx=sample_frame_indices[0],
            strategy_used="hand_point",
            metadata={"failure": "no_detection", "detection_log": log.to_dict()},
        )

    # --- Step 5: Propagate bidirectionally using mask-seeded approach ---
    # Use add_text_prompt (to register obj_id_to_prompt_id) + add_mask_inputs
    # (to seed with the verified mask). This ensures propagation tracks the
    # specific object found during hand-scored sampling, not whatever a fresh
    # text detection finds.
    best_frame_idx = best_result['frame_idx']
    best_mask = best_result['mask']
    prop_prompt = best_result['prompt']
    # Strip "crop:" prefix if the winning detection came from crop fallback
    if prop_prompt.startswith("crop:"):
        prop_prompt = prop_prompt[len("crop:"):]
    log.log("propagation_start",
            f"Step 5: Propagating from frame {best_frame_idx} "
            f"(prompt='{prop_prompt}', score={best_result['score']:.3f})...",
            best_frame_idx=best_frame_idx, prompt=prop_prompt,
            score=best_result['score'])

    best_mask_tensor = torch.from_numpy(best_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    def _collect_masks_hp(session, start, reverse=False, target_obj_id=None):
        """Collect masks from propagation, optionally filtering for a specific obj_id.

        When target_obj_id is set, reads raw model output directly to bypass
        SAM3's hotstart filtering which can hide mask-seeded objects.
        """
        masks = {}
        H = session.video_height
        W = session.video_width
        for out in model.propagate_in_video_iterator(
            inference_session=session, start_frame_idx=start, reverse=reverse,
        ):
            if target_obj_id is not None:
                raw = out["obj_id_to_mask"]
                if target_obj_id not in raw:
                    continue
                low_res = raw[target_obj_id]  # (1, H_low, W_low)
                # Upscale to video resolution
                full_res = torch.nn.functional.interpolate(
                    low_res.unsqueeze(0), size=(H, W),
                    mode="bilinear", align_corners=False,
                ).squeeze(0).squeeze(0)
                masks[out.frame_idx] = (full_res > 0).cpu().numpy()
            else:
                processed = processor.postprocess_outputs(session, out)
                fm = processed['masks']
                if fm.ndim == 4:
                    fm = fm.squeeze(1)
                if len(fm) > 0:
                    masks[out.frame_idx] = fm[0].cpu().numpy()
        return masks

    def _create_mask_seed_session(video_frames, best_frame_idx, mask_tensor):
        """Create a session with a non-detecting prompt and seeded mask.

        Uses a dummy prompt so SAM3's text detector finds no new objects,
        while the SAM2 tracker propagates only our seeded mask.
        """
        sess = processor.init_video_session(
            video=video_frames,
            inference_device=device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=dtype,
        )
        # Dummy prompt: detector won't find matches, tracker propagates mask only
        dummy = "_no_detect_placeholder_"
        sess = processor.add_text_prompt(inference_session=sess, text=dummy)
        pid = sess.add_prompt(dummy)
        _seed_object_in_session(sess, pid, best_frame_idx, mask_tensor,
                                keep_alive=10000)  # survive entire video
        return sess

    # Pass 1+2: Mask-seeded bidirectional propagation.
    # Disable hotstart (which would remove our seeded object after N unmatched frames)
    # since we want the tracker to propagate the mask unconditionally.
    saved_hotstart_delay = model.hotstart_delay
    model.hotstart_delay = 0

    log.log("propagation_pass",
            f"  Pass 1: Forward from frame {best_frame_idx} (mask-seeded)...",
            pass_num=1, direction="forward", start_frame=best_frame_idx)
    bidir_session = _create_mask_seed_session(
        video_frames, best_frame_idx, best_mask_tensor.to(device))

    forward_masks = _collect_masks_hp(bidir_session, best_frame_idx, reverse=False,
                                      target_obj_id=0)
    log.log("propagation_pass",
            f"    {len(forward_masks)} frames",
            pass_num=1, num_frames=len(forward_masks))

    backward_masks = {}
    if best_frame_idx > 0:
        log.log("propagation_pass",
                f"  Pass 2: Backward from frame {best_frame_idx}...",
                pass_num=2, direction="backward", start_frame=best_frame_idx)
        bidir_session.hotstart_removed_obj_ids.clear()
        backward_masks = _collect_masks_hp(bidir_session, best_frame_idx, reverse=True,
                                           target_obj_id=0)
        log.log("propagation_pass",
                f"    {len(backward_masks)} frames",
                pass_num=2, num_frames=len(backward_masks))

    model.hotstart_delay = saved_hotstart_delay  # restore

    # Pass 3: Forward from frame 0 as fallback (text-only, may find different object)
    log.log("propagation_pass",
            f"  Pass 3: Forward from frame 0 (fallback)...",
            pass_num=3, direction="forward", start_frame=0)
    fallback_session = processor.init_video_session(
        video=video_frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=dtype,
    )
    fallback_session = processor.add_text_prompt(
        inference_session=fallback_session,
        text=prop_prompt,
    )
    fallback_masks = _collect_masks_hp(fallback_session, 0, reverse=False)
    log.log("propagation_pass",
            f"    {len(fallback_masks)} frames",
            pass_num=3, num_frames=len(fallback_masks))

    # Merge: forward-from-best > backward > forward-from-0
    final_masks = {**fallback_masks, **backward_masks, **forward_masks}
    final_scores = {idx: score_detection(m, method=score_metric) for idx, m in final_masks.items()}

    log.log("propagation_complete",
            f"Detection complete: {len(final_masks)} frames with masks",
            total_frames_with_masks=len(final_masks))

    return DetectionResult(
        masks=final_masks,
        object_ids=[0],
        scores=final_scores,
        best_frame_idx=best_frame_idx,
        strategy_used="hand_point",
        metadata={
            "prompt": prompt,
            "winning_prompt": best_result['prompt'],
            "winning_score": best_result['score'],
            "best_frame": best_frame_idx,
            "sample_frames": sample_frame_indices,
            "prompt_cascade": prompt_cascade,
            "num_sample_frames": num_sample_frames,
            "n_joint_frames": n_joint_frames,
            "detection_log": log.to_dict(),
        }
    )


def grounding_dino_detection(model, processor, video_frames, prompt: str,
                              device, dtype=torch.bfloat16,
                              num_sample_frames: int = 5,
                              box_threshold: float = 0.25,
                              text_threshold: float = 0.25,
                              score_metric: str = "mask_area") -> DetectionResult:
    """
    Strategy 7: GroundingDINO open-vocabulary detection with bounding box prompts.

    Uses GroundingDINO to detect objects by text prompt across sampled frames,
    scores detections by confidence and hand proximity, then uses the best
    bounding box as a SAM3 box prompt for mask propagation.

    Falls back to simple_detection if GroundingDINO finds no detections.

    Args:
        model: Sam3VideoModel
        processor: Sam3VideoProcessor
        video_frames: List of PIL Images
        prompt: Text prompt for object detection
        device: Torch device
        dtype: Torch dtype
        num_sample_frames: Number of frames to run GroundingDINO on
        box_threshold: GroundingDINO box confidence threshold
        text_threshold: GroundingDINO text confidence threshold
        score_metric: Metric for scoring final masks

    Returns:
        DetectionResult with masks for all frames
    """
    from grounding_dino_detector import GroundingDINODetector, compute_hand_proximity_score

    img_h, img_w = np.array(video_frames[0]).shape[:2]

    # Initialize GroundingDINO
    print("  Initializing GroundingDINO detector...")
    try:
        detector = GroundingDINODetector(device=str(device))
    except Exception as e:
        print(f"  Warning: Failed to load GroundingDINO: {e}")
        print(f"  Falling back to simple detection.")
        return simple_detection(model, processor, video_frames, prompt, device, dtype)

    # Run detection on sampled frames
    print(f"  Running GroundingDINO on {num_sample_frames} sampled frames...")
    multi_frame_detections = detector.detect_multi_frame(
        video_frames, prompt,
        num_samples=num_sample_frames,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    if not multi_frame_detections:
        print("  Warning: GroundingDINO found no detections, falling back to simple detection")
        return simple_detection(model, processor, video_frames, prompt, device, dtype)

    # Optionally detect hands for proximity scoring
    # Run a quick hand detection on the best frame to get hand bboxes
    hand_bboxes = []
    try:
        best_gdino_frame = max(multi_frame_detections.keys(),
                               key=lambda k: max(d.confidence for d in multi_frame_detections[k]))
        hand_detections = detector.detect(video_frames[best_gdino_frame], "hand.", box_threshold=0.3)
        hand_bboxes = [d.bbox for d in hand_detections]
        if hand_bboxes:
            print(f"  Found {len(hand_bboxes)} hand(s) in frame {best_gdino_frame}")
    except Exception:
        pass  # Hand detection is optional

    # Score all detections by confidence * hand proximity
    best_detection = None
    best_score = -1
    best_frame_idx = 0

    for frame_idx, detections in multi_frame_detections.items():
        for det in detections:
            proximity = compute_hand_proximity_score(det, hand_bboxes if hand_bboxes else None)
            combined_score = det.confidence * proximity
            if combined_score > best_score:
                best_score = combined_score
                best_detection = det
                best_frame_idx = frame_idx

    print(f"  Best detection: frame {best_frame_idx}, "
          f"confidence={best_detection.confidence:.3f}, "
          f"label='{best_detection.label}', "
          f"bbox={best_detection.bbox}")

    # Use SAM3 with box prompt from GroundingDINO
    print(f"  Initializing SAM3 with bounding box prompt...")
    inference_session = processor.init_video_session(
        video=video_frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=dtype,
    )

    # Add box prompt using the best detection's bounding box
    x_min, y_min, x_max, y_max = best_detection.bbox
    box_tensor = torch.tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float32)

    inference_session = processor.add_box_prompt(
        inference_session=inference_session,
        frame_idx=best_frame_idx,
        box=box_tensor,
    )

    # Forward propagation from best frame
    forward_masks = {}
    for model_outputs in model.propagate_in_video_iterator(
        inference_session=inference_session,
        start_frame_idx=best_frame_idx,
        reverse=False,
    ):
        processed_outputs = processor.postprocess_outputs(inference_session, model_outputs)
        frame_masks = processed_outputs['masks']
        if frame_masks.ndim == 4:
            frame_masks = frame_masks.squeeze(1)
        if len(frame_masks) > 0:
            forward_masks[model_outputs.frame_idx] = frame_masks[0].cpu().numpy()

    # Backward propagation from best frame (if not frame 0)
    backward_masks = {}
    if best_frame_idx > 0:
        inference_session.reset_tracking_data()

        inference_session = processor.add_box_prompt(
            inference_session=inference_session,
            frame_idx=best_frame_idx,
            box=box_tensor,
        )

        for model_outputs in model.propagate_in_video_iterator(
            inference_session=inference_session,
            start_frame_idx=best_frame_idx,
            reverse=True,
        ):
            processed_outputs = processor.postprocess_outputs(inference_session, model_outputs)
            frame_masks = processed_outputs['masks']
            if frame_masks.ndim == 4:
                frame_masks = frame_masks.squeeze(1)
            if len(frame_masks) > 0:
                backward_masks[model_outputs.frame_idx] = frame_masks[0].cpu().numpy()

    # Combine forward and backward
    final_masks = {**backward_masks, **forward_masks}
    final_scores = {idx: score_detection(m, method=score_metric) for idx, m in final_masks.items()}

    print(f"  GroundingDINO detection complete: {len(final_masks)} frames with masks")

    return DetectionResult(
        masks=final_masks,
        object_ids=[0],
        scores=final_scores,
        best_frame_idx=best_frame_idx,
        strategy_used="grounding_dino",
        metadata={
            "prompt": prompt,
            "best_detection_bbox": best_detection.bbox,
            "best_detection_confidence": best_detection.confidence,
            "best_detection_label": best_detection.label,
            "num_sample_frames": num_sample_frames,
            "hand_bboxes": hand_bboxes,
        }
    )


def detect_object(model, processor, video_frames, prompt: str,
                  strategy: str = "simple", device=None, dtype=torch.bfloat16,
                  verbose: bool = True, **kwargs) -> DetectionResult:
    """
    Main entry point for object detection.

    Args:
        model: Sam3VideoModel
        processor: Sam3VideoProcessor
        video_frames: List of PIL Images
        prompt: Text prompt for object detection
        strategy: Detection strategy - "simple", "two_stage", "backward",
                  "consensus", "combined", "hand_point", "grounding_dino"
        device: Torch device
        dtype: Torch dtype
        **kwargs: Additional arguments for specific strategies:
            - bbox_scale: For two_stage and combined (default: 2.0)
            - score_metric: For backward, consensus, combined, hand_point (default: "mask_area")
            - sample_frames: For consensus (default: auto)
            - hand_joints: For hand_point - (T, 21, 3) array of 3D hand joints
            - intrinsics: For hand_point - (3, 3) camera intrinsic matrix
            - kept_frames: For hand_point - Optional array of valid frame indicators
            - use_fingertips: For hand_point - Use fingertip centroid vs hand center (default: True)
            - num_sample_frames: For hand_point and grounding_dino - Number of frames to sample (default: 5)
            - max_objects: For multi-object detection - max objects to detect (default: 1)
            - min_mask_area_pct: For multi-object - min mask area filter (default: 0.005)

    Returns:
        DetectionResult with masks for all frames
    """
    if device is None:
        device = next(model.parameters()).device

    max_objects = kwargs.get("max_objects", 1)

    # Multi-object detection: wraps consensus strategy with prompt cascade
    if max_objects > 1:
        if verbose:
            print(f"Running multi-object detection (max_objects={max_objects})...")
        score_metric = kwargs.get("score_metric", "mask_area")
        sample_frames = kwargs.get("sample_frames", None)
        min_mask_area_pct = kwargs.get("min_mask_area_pct", 0.005)
        additional_prompts = kwargs.get("additional_prompts", None)
        results = multi_object_consensus_detection(
            model, processor, video_frames, prompt,
            device, dtype, max_objects, sample_frames,
            score_metric, min_mask_area_pct,
            additional_prompts=additional_prompts,
        )
        if results:
            return results  # Returns List[DetectionResult]
        if verbose:
            print("  Multi-object detection found nothing, falling back to single object")

    if verbose:
        print(f"Running {strategy} detection strategy...")

    if strategy == "simple":
        return simple_detection(model, processor, video_frames, prompt, device, dtype)

    elif strategy == "two_stage":
        bbox_scale = kwargs.get("bbox_scale", 2.0)
        return two_stage_detection(model, processor, video_frames, prompt,
                                   device, dtype, bbox_scale)

    elif strategy == "backward":
        score_metric = kwargs.get("score_metric", "mask_area")
        return backward_tracking(model, processor, video_frames, prompt,
                                device, dtype, score_metric)

    elif strategy == "consensus":
        score_metric = kwargs.get("score_metric", "mask_area")
        sample_frames = kwargs.get("sample_frames", None)
        return consensus_detection(model, processor, video_frames, prompt,
                                  device, dtype, sample_frames, score_metric)

    elif strategy == "combined":
        bbox_scale = kwargs.get("bbox_scale", 2.0)
        score_metric = kwargs.get("score_metric", "mask_area")
        return combined_detection(model, processor, video_frames, prompt,
                                 device, dtype, bbox_scale, score_metric)

    elif strategy == "hand_point":
        hand_joints = kwargs.get("hand_joints")
        intrinsics = kwargs.get("intrinsics")

        if hand_joints is None or intrinsics is None:
            if verbose:
                print("  Warning: hand_point strategy requires hand_joints and intrinsics.")
                print("  Falling back to simple detection.")
            return simple_detection(model, processor, video_frames, prompt, device, dtype)

        score_metric = kwargs.get("score_metric", "mask_area")
        kept_frames = kwargs.get("kept_frames", None)
        use_fingertips = kwargs.get("use_fingertips", True)

        num_sample_frames = kwargs.get("num_sample_frames", 5)
        use_llm = kwargs.get("use_llm", False)
        early_exit_score = kwargs.get("early_exit_score", 0.8)

        return hand_point_detection(
            model, processor, video_frames, prompt,
            hand_joints=hand_joints,
            intrinsics=intrinsics,
            kept_frames=kept_frames,
            device=device,
            dtype=dtype,
            use_fingertips=use_fingertips,
            score_metric=score_metric,
            num_sample_frames=num_sample_frames,
            use_llm=use_llm,
            early_exit_score=early_exit_score,
            verbose=verbose,
        )

    elif strategy == "grounding_dino":
        num_sample_frames = kwargs.get("num_sample_frames", 5)
        box_threshold = kwargs.get("box_threshold", 0.25)
        text_threshold = kwargs.get("text_threshold", 0.25)
        score_metric = kwargs.get("score_metric", "mask_area")
        return grounding_dino_detection(model, processor, video_frames, prompt,
                                         device, dtype, num_sample_frames,
                                         box_threshold, text_threshold, score_metric)

    else:
        raise ValueError(f"Unknown detection strategy: {strategy}. "
                        f"Choose from: simple, two_stage, backward, consensus, combined, hand_point, grounding_dino")
