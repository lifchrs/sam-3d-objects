"""
GroundingDINO wrapper for open-vocabulary object detection.

Uses HuggingFace transformers AutoModelForZeroShotObjectDetection
to detect objects by text prompt with confidence scores and bounding boxes.
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Detection:
    """A single object detection result."""
    bbox: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
    confidence: float
    label: str


class GroundingDINODetector:
    """Wrapper for GroundingDINO zero-shot object detection."""

    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-base", device: str = "cuda"):
        """
        Initialize GroundingDINO detector.

        Args:
            model_id: HuggingFace model ID
            device: Device to run inference on
        """
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.model.eval()

    def detect(self, image: Image.Image, text_prompt: str,
               box_threshold: float = 0.25, text_threshold: float = 0.25) -> List[Detection]:
        """
        Detect objects in a single image.

        Args:
            image: PIL Image
            text_prompt: Text description of object to detect (e.g., "cup. bottle. kettle.")
                         Multiple objects should be separated by periods.
            box_threshold: Minimum confidence for box detection
            text_threshold: Minimum confidence for text matching

        Returns:
            List of Detection objects sorted by confidence (highest first)
        """
        # Ensure prompt ends with period for GroundingDINO format
        if not text_prompt.endswith("."):
            text_prompt = text_prompt + "."

        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]]  # (height, width)
        )[0]

        detections = []
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"]

        for box, score, label in zip(boxes, scores, labels):
            x_min, y_min, x_max, y_max = box.astype(int)
            detections.append(Detection(
                bbox=(int(x_min), int(y_min), int(x_max), int(y_max)),
                confidence=float(score),
                label=label
            ))

        # Sort by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def detect_multi_frame(self, frames: List[Image.Image], text_prompt: str,
                           num_samples: int = 5,
                           box_threshold: float = 0.25,
                           text_threshold: float = 0.25) -> dict:
        """
        Run detection on sampled frames from a video.

        Args:
            frames: List of PIL Images (full video frames)
            text_prompt: Text description of object to detect
            num_samples: Number of frames to sample
            box_threshold: Minimum confidence for box detection
            text_threshold: Minimum confidence for text matching

        Returns:
            Dict mapping frame_idx -> List[Detection]
        """
        n_frames = len(frames)

        # Sample frames evenly across the video
        if n_frames <= num_samples:
            sample_indices = list(range(n_frames))
        else:
            sample_indices = [int(i * (n_frames - 1) / (num_samples - 1)) for i in range(num_samples)]

        results = {}
        for idx in sample_indices:
            detections = self.detect(frames[idx], text_prompt, box_threshold, text_threshold)
            if detections:
                results[idx] = detections

        return results


def compute_hand_proximity_score(detection: Detection, hand_bboxes: Optional[List[Tuple[int, int, int, int]]] = None) -> float:
    """
    Score a detection by proximity to hand bounding boxes.

    Args:
        detection: Detection result
        hand_bboxes: List of hand bounding boxes (x_min, y_min, x_max, y_max)

    Returns:
        Proximity score (0-1, higher = closer to hands)
    """
    if not hand_bboxes:
        return 1.0  # No hand info, don't penalize

    det_cx = (detection.bbox[0] + detection.bbox[2]) / 2
    det_cy = (detection.bbox[1] + detection.bbox[3]) / 2

    min_dist = float('inf')
    for hbox in hand_bboxes:
        hand_cx = (hbox[0] + hbox[2]) / 2
        hand_cy = (hbox[1] + hbox[3]) / 2
        dist = np.sqrt((det_cx - hand_cx)**2 + (det_cy - hand_cy)**2)
        min_dist = min(min_dist, dist)

    # Normalize: closer is better. Use sigmoid-like decay.
    # Objects within ~100 pixels of hand center get high scores
    score = 1.0 / (1.0 + min_dist / 100.0)
    return score
