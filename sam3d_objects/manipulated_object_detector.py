"""
Manipulated object detection module — multi-method comparison.

Provides 5 detection methods (2 text-dependent, 3 text-independent) for finding
objects being manipulated in egocentric video, all using SAM3 text or mask
conditioning. Each method returns a single-frame detection scored by hand-overlap.

Methods:
1. text_fullframe   — prompt cascade on full frame (text-dependent)
2. text_crop        — prompt cascade on hand-region crop (text-dependent)
3. generic_fullframe — fixed generic prompts on full frame (text-independent)
4. generic_crop     — fixed generic prompts on hand-region crop (text-independent)
5. mask_refine      — seed mask at fingertip centroid, SAM3 mask conditioning (text-independent)
"""

import numpy as np
import torch
from PIL import Image
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from detection_strategies import (
    project_joints_to_2d,
    get_fingertip_centroid,
    get_hand_bbox_from_joints,
    score_detection_hand_overlap,
    expand_bbox,
    crop_frame,
    uncrop_mask,
    create_point_mask,
)
from text_utils import get_detection_prompts, extract_object_noun


@dataclass
class SingleFrameDetection:
    """Result from a single detection method on a single frame."""
    mask: np.ndarray            # (H, W) boolean
    score: float                # hand-overlap score
    frame_idx: int
    method: str
    prompt_used: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionComparison:
    """Comparison of all detection methods on a single frame."""
    annotation_path: str
    action_text: str
    frame_idx: int
    frame_image: np.ndarray     # (H, W, 3) RGB
    results: Dict[str, SingleFrameDetection]  # method_name -> result
    hand_data: Dict             # joints_2d, centroid, bbox


GENERIC_FULLFRAME_PROMPTS = ["object held in hand", "object in hand", "held object"]
GENERIC_CROP_PROMPTS = ["object", "object held in hand"]


class ManipulatedObjectDetector:
    """Multi-method manipulated object detector using SAM3."""

    def __init__(self, model, processor, device, dtype=torch.bfloat16):
        self.model = model
        self.processor = processor
        self.device = device
        self.dtype = dtype

    # ------------------------------------------------------------------
    # Frame selection & hand data
    # ------------------------------------------------------------------

    def select_frame(self, annotation: dict, frame_idx: Optional[int] = None) -> int:
        """Select a detection frame from the latter 70% of valid hand frames."""
        if frame_idx is not None:
            return frame_idx

        # Find active hand
        text_data = annotation.get("text", {})
        active_hand = None
        for hand in ["left", "right"]:
            if text_data.get(hand):
                active_hand = hand
                break
        if active_hand is None:
            for hand in ["left", "right"]:
                hd = annotation.get(hand)
                if isinstance(hd, dict) and "kept_frames" in hd and np.any(hd["kept_frames"] == 1):
                    active_hand = hand
                    break

        if active_hand is None:
            # Fallback: middle frame
            n = len(annotation.get("video_decode_frame", [0]))
            return n // 2

        hd = annotation[active_hand]
        kept = hd["kept_frames"]
        valid = np.where(np.array(kept) == 1)[0]
        if len(valid) == 0:
            return len(kept) // 2

        # Latter 70%
        start = int(len(valid) * 0.3)
        candidates = valid[start:]
        if len(candidates) == 0:
            candidates = valid
        # Pick middle of candidates
        return int(candidates[len(candidates) // 2])

    def get_hand_data(self, annotation: dict, frame_idx: int) -> Dict:
        """Project hand joints to 2D and compute centroid + bbox.

        Returns dict with keys: joints_2d, centroid, bbox, hand_side
        """
        intrinsics = annotation["intrinsics"]
        # Determine image size from first video frame is caller's job;
        # but we need it for bbox clamping. We'll compute it later if needed.
        # For now, store raw data and let callers pass img_size.

        text_data = annotation.get("text", {})
        active_hand = None
        for hand in ["left", "right"]:
            if text_data.get(hand):
                active_hand = hand
                break
        if active_hand is None:
            for hand in ["left", "right"]:
                hd = annotation.get(hand)
                if isinstance(hd, dict) and "kept_frames" in hd and np.any(hd["kept_frames"] == 1):
                    active_hand = hand
                    break
        if active_hand is None:
            raise ValueError("No valid hand data in annotation")

        hd = annotation[active_hand]
        joints_3d = hd["joints_camspace"][frame_idx]  # (21, 3)
        joints_2d = project_joints_to_2d(joints_3d, intrinsics)
        cx, cy = get_fingertip_centroid(joints_2d)

        return {
            "joints_2d": joints_2d,
            "centroid": (cx, cy),
            "hand_side": active_hand,
        }

    def _finalize_hand_data(self, hand_data: Dict, img_size: Tuple[int, int]) -> Dict:
        """Clamp centroid and compute bbox given image size."""
        h, w = img_size
        cx, cy = hand_data["centroid"]
        cx = max(0, min(cx, w - 1))
        cy = max(0, min(cy, h - 1))
        bbox = get_hand_bbox_from_joints(hand_data["joints_2d"], img_size, padding=50)
        return {
            **hand_data,
            "centroid": (cx, cy),
            "bbox": bbox,
        }

    # ------------------------------------------------------------------
    # SAM3 primitives
    # ------------------------------------------------------------------

    def _detect_text_single_frame(
        self, video_frames: list, frame_idx: int, text: str
    ) -> List[np.ndarray]:
        """Run SAM3 text detection on a single frame. Returns list of masks."""
        session = self.processor.init_video_session(
            video=video_frames,
            inference_device=self.device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=self.dtype,
        )
        session = self.processor.add_text_prompt(
            inference_session=session,
            text=text,
        )

        masks_out = []
        for model_outputs in self.model.propagate_in_video_iterator(
            inference_session=session,
            start_frame_idx=frame_idx,
            max_frame_num_to_track=1,
        ):
            processed = self.processor.postprocess_outputs(session, model_outputs)
            frame_masks = processed["masks"]
            if frame_masks.ndim == 4:
                frame_masks = frame_masks.squeeze(1)
            for i in range(len(frame_masks)):
                m = frame_masks[i].cpu().numpy()
                if m.sum() > 0:
                    masks_out.append(m)
            break  # only need first iteration
        return masks_out

    def _refine_mask_single_frame(
        self, video_frames: list, frame_idx: int, seed_mask: np.ndarray
    ) -> Optional[np.ndarray]:
        """Refine a seed mask using SAM3 mask conditioning. Returns refined mask or None."""
        session = self.processor.init_video_session(
            video=video_frames,
            inference_device=self.device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=self.dtype,
        )
        mask_tensor = torch.from_numpy(seed_mask.astype(np.float32)).unsqueeze(0)
        session.obj_id_to_idx(obj_id=0)
        session.add_mask_inputs(
            obj_idx=0,
            frame_idx=frame_idx,
            inputs=mask_tensor.to(self.device),
        )

        for model_outputs in self.model.propagate_in_video_iterator(
            inference_session=session,
            start_frame_idx=frame_idx,
            max_frame_num_to_track=1,
        ):
            processed = self.processor.postprocess_outputs(session, model_outputs)
            frame_masks = processed["masks"]
            if frame_masks.ndim == 4:
                frame_masks = frame_masks.squeeze(1)
            if len(frame_masks) > 0:
                return frame_masks[0].cpu().numpy()
            break
        return None

    # ------------------------------------------------------------------
    # 5 detection methods
    # ------------------------------------------------------------------

    def text_fullframe(
        self,
        video_frames: list,
        frame_idx: int,
        action_text: str,
        hand_data: Dict,
    ) -> SingleFrameDetection:
        """Method 1: Text prompt cascade on full frame."""
        img_size = np.array(video_frames[0]).shape[:2]
        prompts = get_detection_prompts(action_text)
        best_mask = None
        best_score = -1.0
        best_prompt = None

        for prompt in prompts:
            masks = self._detect_text_single_frame(video_frames, frame_idx, prompt)
            for m in masks:
                sc = score_detection_hand_overlap(
                    m, hand_data["bbox"], hand_data["centroid"], img_size
                )
                if sc > best_score:
                    best_score = sc
                    best_mask = m
                    best_prompt = prompt
            if best_score > 1.0:
                break

        if best_mask is None:
            best_mask = np.zeros(img_size, dtype=bool)
            best_score = 0.0

        return SingleFrameDetection(
            mask=best_mask.astype(bool),
            score=best_score,
            frame_idx=frame_idx,
            method="text_fullframe",
            prompt_used=best_prompt,
            metadata={"prompts_tried": prompts},
        )

    def text_crop(
        self,
        video_frames: list,
        frame_idx: int,
        action_text: str,
        hand_data: Dict,
    ) -> SingleFrameDetection:
        """Method 2: Text prompt cascade on hand-region crop."""
        img_size = np.array(video_frames[0]).shape[:2]
        prompts = get_detection_prompts(action_text)
        expanded = expand_bbox(hand_data["bbox"], 1.5, img_size)

        frame_np = np.array(video_frames[frame_idx])
        cropped = crop_frame(frame_np, expanded)
        cropped_pil = Image.fromarray(cropped)

        best_mask = None
        best_score = -1.0
        best_prompt = None

        for prompt in prompts:
            crop_masks = self._detect_text_single_frame([cropped_pil], 0, prompt)
            for cm in crop_masks:
                full_mask = uncrop_mask(cm, expanded, img_size)
                sc = score_detection_hand_overlap(
                    full_mask, hand_data["bbox"], hand_data["centroid"], img_size
                )
                if sc > best_score:
                    best_score = sc
                    best_mask = full_mask
                    best_prompt = prompt
            if best_score > 1.0:
                break

        if best_mask is None:
            best_mask = np.zeros(img_size, dtype=bool)
            best_score = 0.0

        return SingleFrameDetection(
            mask=best_mask.astype(bool),
            score=best_score,
            frame_idx=frame_idx,
            method="text_crop",
            prompt_used=best_prompt,
            metadata={"crop_bbox": expanded, "prompts_tried": prompts},
        )

    def generic_fullframe(
        self,
        video_frames: list,
        frame_idx: int,
        hand_data: Dict,
    ) -> SingleFrameDetection:
        """Method 3: Fixed generic prompts on full frame."""
        img_size = np.array(video_frames[0]).shape[:2]
        best_mask = None
        best_score = -1.0
        best_prompt = None

        for prompt in GENERIC_FULLFRAME_PROMPTS:
            masks = self._detect_text_single_frame(video_frames, frame_idx, prompt)
            for m in masks:
                sc = score_detection_hand_overlap(
                    m, hand_data["bbox"], hand_data["centroid"], img_size
                )
                if sc > best_score:
                    best_score = sc
                    best_mask = m
                    best_prompt = prompt
            if best_score > 1.0:
                break

        if best_mask is None:
            best_mask = np.zeros(img_size, dtype=bool)
            best_score = 0.0

        return SingleFrameDetection(
            mask=best_mask.astype(bool),
            score=best_score,
            frame_idx=frame_idx,
            method="generic_fullframe",
            prompt_used=best_prompt,
            metadata={"prompts_tried": GENERIC_FULLFRAME_PROMPTS},
        )

    def generic_crop(
        self,
        video_frames: list,
        frame_idx: int,
        hand_data: Dict,
    ) -> SingleFrameDetection:
        """Method 4: Fixed generic prompts on hand-region crop."""
        img_size = np.array(video_frames[0]).shape[:2]
        expanded = expand_bbox(hand_data["bbox"], 1.5, img_size)

        frame_np = np.array(video_frames[frame_idx])
        cropped = crop_frame(frame_np, expanded)
        cropped_pil = Image.fromarray(cropped)

        best_mask = None
        best_score = -1.0
        best_prompt = None

        for prompt in GENERIC_CROP_PROMPTS:
            crop_masks = self._detect_text_single_frame([cropped_pil], 0, prompt)
            for cm in crop_masks:
                full_mask = uncrop_mask(cm, expanded, img_size)
                sc = score_detection_hand_overlap(
                    full_mask, hand_data["bbox"], hand_data["centroid"], img_size
                )
                if sc > best_score:
                    best_score = sc
                    best_mask = full_mask
                    best_prompt = prompt
            if best_score > 1.0:
                break

        if best_mask is None:
            best_mask = np.zeros(img_size, dtype=bool)
            best_score = 0.0

        return SingleFrameDetection(
            mask=best_mask.astype(bool),
            score=best_score,
            frame_idx=frame_idx,
            method="generic_crop",
            prompt_used=best_prompt,
            metadata={"crop_bbox": expanded, "prompts_tried": GENERIC_CROP_PROMPTS},
        )

    def mask_refine(
        self,
        video_frames: list,
        frame_idx: int,
        hand_data: Dict,
        seed_radius: int = 20,
    ) -> SingleFrameDetection:
        """Method 5: Seed mask at fingertip centroid + SAM3 mask conditioning."""
        img_size = np.array(video_frames[0]).shape[:2]
        cx, cy = hand_data["centroid"]
        seed = create_point_mask(cx, cy, img_size, radius=seed_radius)

        refined = self._refine_mask_single_frame(video_frames, frame_idx, seed)
        if refined is None or refined.sum() == 0:
            # Fallback: return the seed itself
            refined = seed

        sc = score_detection_hand_overlap(
            refined, hand_data["bbox"], hand_data["centroid"], img_size
        )

        return SingleFrameDetection(
            mask=refined.astype(bool),
            score=sc,
            frame_idx=frame_idx,
            method="mask_refine",
            prompt_used=None,
            metadata={"seed_radius": seed_radius, "centroid": (cx, cy)},
        )

    # ------------------------------------------------------------------
    # Compare all
    # ------------------------------------------------------------------

    def compare_all(
        self,
        video_frames: list,
        annotation: dict,
        frame_idx: Optional[int] = None,
        annotation_path: str = "",
    ) -> DetectionComparison:
        """Run all 5 detection methods on a single frame and return comparison."""
        frame_idx = self.select_frame(annotation, frame_idx)
        raw_hand = self.get_hand_data(annotation, frame_idx)
        img_size = np.array(video_frames[0]).shape[:2]
        hand_data = self._finalize_hand_data(raw_hand, img_size)

        # Get action text
        text_data = annotation.get("text", {})
        action_text = ""
        for hand in ["left", "right"]:
            actions = text_data.get(hand, [])
            if actions:
                action_text = actions[0][0]
                break
        if not action_text:
            action_text = "object held in hand"

        frame_image = np.array(video_frames[frame_idx])

        print(f"  Frame {frame_idx}: action='{action_text}', "
              f"centroid=({hand_data['centroid'][0]:.0f}, {hand_data['centroid'][1]:.0f})")

        results = {}

        # Text-dependent methods
        print("    Running text_fullframe...")
        results["text_fullframe"] = self.text_fullframe(
            video_frames, frame_idx, action_text, hand_data
        )
        print(f"      score={results['text_fullframe'].score:.3f}, "
              f"prompt='{results['text_fullframe'].prompt_used}'")

        print("    Running text_crop...")
        results["text_crop"] = self.text_crop(
            video_frames, frame_idx, action_text, hand_data
        )
        print(f"      score={results['text_crop'].score:.3f}, "
              f"prompt='{results['text_crop'].prompt_used}'")

        # Text-independent methods
        print("    Running generic_fullframe...")
        results["generic_fullframe"] = self.generic_fullframe(
            video_frames, frame_idx, hand_data
        )
        print(f"      score={results['generic_fullframe'].score:.3f}, "
              f"prompt='{results['generic_fullframe'].prompt_used}'")

        print("    Running generic_crop...")
        results["generic_crop"] = self.generic_crop(
            video_frames, frame_idx, hand_data
        )
        print(f"      score={results['generic_crop'].score:.3f}, "
              f"prompt='{results['generic_crop'].prompt_used}'")

        print("    Running mask_refine...")
        results["mask_refine"] = self.mask_refine(
            video_frames, frame_idx, hand_data
        )
        print(f"      score={results['mask_refine'].score:.3f}")

        return DetectionComparison(
            annotation_path=annotation_path,
            action_text=action_text,
            frame_idx=frame_idx,
            frame_image=frame_image,
            results=results,
            hand_data=hand_data,
        )
