import cv2
import torch
import numpy as np
from PIL import Image
from transformers import Sam3VideoModel, Sam3VideoProcessor
from transformers.video_utils import load_video
from accelerate import Accelerator
import sys
import os
import copy

import argparse

# Ensure we can import internal SAM3D utilities regardless of CWD.
# We avoid importing `sam3d_objects.*` as a package because `sam3d_objects/__init__.py`
# references a missing `sam3d_objects.init` in this checkout.
THIS_DIR = os.path.dirname(__file__)
SAM3D_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
NOTEBOOK_DIR = os.path.join(SAM3D_ROOT, "notebook")
if NOTEBOOK_DIR not in sys.path:
    sys.path.insert(0, NOTEBOOK_DIR)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from inference import Inference
from scipy.spatial.transform import Rotation as R
from detection_strategies import detect_object, DetectionResult
from text_utils import extract_object_noun, get_detection_prompts


def _wxyz_to_xyzw(quat_wxyz):
    """Convert quaternion from wxyz (pytorch3d) to xyzw (scipy) format."""
    if torch.is_tensor(quat_wxyz):
        quat_wxyz = quat_wxyz.detach().cpu().numpy()
    quat_wxyz = np.asarray(quat_wxyz).squeeze()
    return np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])


def _xyzw_to_wxyz(quat_xyzw):
    """Convert quaternion from xyzw (scipy) to wxyz (pytorch3d) format."""
    quat_xyzw = np.asarray(quat_xyzw).squeeze()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def _quaternion_to_matrix_scipy(quat_wxyz):
    """Convert wxyz quaternion to rotation matrix using scipy."""
    quat_xyzw = _wxyz_to_xyzw(quat_wxyz)
    return R.from_quat(quat_xyzw).as_matrix()


def _matrix_to_quaternion_scipy(rot_matrix):
    """Convert rotation matrix to wxyz quaternion using scipy."""
    if torch.is_tensor(rot_matrix):
        rot_matrix = rot_matrix.detach().cpu().numpy()
    rot_matrix = np.asarray(rot_matrix).squeeze()
    if rot_matrix.shape == (1, 3, 3):
        rot_matrix = rot_matrix[0]
    quat_xyzw = R.from_matrix(rot_matrix).as_quat()
    return _xyzw_to_wxyz(quat_xyzw)


def _quaternion_multiply_scipy(q1_wxyz, q2_wxyz):
    """Multiply two wxyz quaternions using scipy (q1 * q2)."""
    q1_xyzw = _wxyz_to_xyzw(q1_wxyz)
    q2_xyzw = _wxyz_to_xyzw(q2_wxyz)
    r1 = R.from_quat(q1_xyzw)
    r2 = R.from_quat(q2_xyzw)
    result = r1 * r2
    return _xyzw_to_wxyz(result.as_quat())


def _as_pose_tensors(rotation, translation, scale, *, device, dtype):
    """
    Normalize pose inputs to torch tensors:
      - quat_wxyz: (1, 4)
      - trans: (1, 3)
      - scale3: (1, 3)
    """
    rot = rotation.detach() if torch.is_tensor(rotation) else torch.from_numpy(np.asarray(rotation))
    trans = translation.detach() if torch.is_tensor(translation) else torch.from_numpy(np.asarray(translation))
    sc = scale.detach() if torch.is_tensor(scale) else torch.from_numpy(np.asarray(scale))

    rot = rot.squeeze().to(device=device, dtype=dtype)
    trans = trans.squeeze().to(device=device, dtype=dtype)
    sc = sc.squeeze().to(device=device, dtype=dtype)

    if rot.shape == (4,):
        quat = rot.view(1, 4)
    elif rot.shape == (3, 3):
        # Use scipy to convert rotation matrix to quaternion (consistent with run_full_pipeline.py)
        quat_wxyz = _matrix_to_quaternion_scipy(rot.cpu().numpy())
        quat = torch.from_numpy(quat_wxyz.astype(np.float32)).view(1, 4).to(device=device, dtype=dtype)
    else:
        raise ValueError(f"Unexpected rotation shape: {tuple(rot.shape)}")

    if trans.shape != (3,):
        raise ValueError(f"Unexpected translation shape: {tuple(trans.shape)}")
    trans = trans.view(1, 3)

    if sc.numel() == 1:
        scale3 = sc.view(1, 1).expand(1, 3)
    else:
        if tuple(sc.shape) != (3,):
            raise ValueError(f"Unexpected scale shape: {tuple(sc.shape)}")
        scale3 = sc.view(1, 3)

    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(quat.dtype).eps)
    return quat, trans, scale3


def _yaw180_about_camera_y(points_cam_p3d: torch.Tensor) -> torch.Tensor:
    """
    Rotate points 180 degrees about camera Y axis (PyTorch3D camera coords),
    around the centroid so the object stays in place.
    """
    center = points_cam_p3d.mean(dim=0, keepdim=True)
    pts0 = points_cam_p3d - center
    R_y = torch.tensor(
        [[-1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0],
         [0.0, 0.0, -1.0]],
        device=points_cam_p3d.device,
        dtype=points_cam_p3d.dtype,
    )
    return (pts0 @ R_y.T) + center


def _apply_pose_to_points_l2c(points_local: torch.Tensor, rotation, translation, scale, *, flip_facing: bool):
    """
    Apply SAM3D's l2c pose to points in canonical/local coordinates.
    Returns points in PyTorch3D camera coordinates and the (1,4) pose quaternion (wxyz).

    Uses scipy for rotation matrix conversion, consistent with run_full_pipeline.py.
    """
    quat_l2c, trans_l2c, scale_l2c = _as_pose_tensors(
        rotation, translation, scale, device=points_local.device, dtype=points_local.dtype
    )

    # Convert quaternion to rotation matrix using scipy (consistent with run_full_pipeline.py)
    rot_matrix = _quaternion_to_matrix_scipy(quat_l2c)
    rot_matrix_t = torch.from_numpy(rot_matrix.astype(np.float32)).to(
        device=points_local.device, dtype=points_local.dtype
    )

    # Apply transformation: scale -> rotate -> translate
    # Note: no transpose on rotation matrix - matches run_full_pipeline.py convention
    scale_mean = scale_l2c.mean()  # Use mean scale for uniform scaling
    pts_cam = points_local * scale_mean
    pts_cam = pts_cam @ rot_matrix_t
    pts_cam = pts_cam + trans_l2c.squeeze(0)

    if flip_facing:
        pts_cam = _yaw180_about_camera_y(pts_cam)
    return pts_cam, quat_l2c



def load_camera_pose(frame_idx, camera_dir):
    """
    Load camera pose for a specific frame.

    Args:
        frame_idx: Frame index
        camera_dir: Directory containing camera NPZ files

    Returns:
        camera_to_world: 4x4 transformation matrix from camera to world coordinates
        intrinsics: 3x3 camera intrinsics matrix
    """
    camera_file = os.path.join(camera_dir, f"{frame_idx:06d}.npz")
    data = np.load(camera_file)
    return data['pose'], data['intrinsics']


def transform_gaussian_to_world(gs_camera, camera_to_world):
    """
    Transform Gaussian splat from camera coordinates to world coordinates.

    Args:
        gs_camera: Gaussian splat in camera coordinates
        camera_to_world: 4x4 camera-to-world transformation matrix

    Returns:
        Transformed Gaussian splat in world coordinates
    """
    gs_world = copy.deepcopy(gs_camera)

    # IMPORTANT:
    # - SAM3D pose/pointmaps are in PyTorch3D camera coords (x left, y up, z inward).
    # - TTT3R camera poses are OpenCV-style. Convert camera coords via M = diag(-1, -1, 1),
    #   then apply camera_to_world (OpenCV).
    device = gs_camera._xyz.device
    dtype = gs_camera._xyz.dtype

    xyz_camera_p3d = gs_world.get_xyz  # (N,3) torch
    Mv = torch.tensor([-1.0, -1.0, 1.0], device=device, dtype=dtype)
    xyz_camera_cv = xyz_camera_p3d * Mv

    c2w = torch.from_numpy(camera_to_world).to(device=device, dtype=dtype)  # (4,4)
    xyz_h = torch.cat([xyz_camera_cv, torch.ones((xyz_camera_cv.shape[0], 1), device=device, dtype=dtype)], dim=1)
    xyz_world = (xyz_h @ c2w.T)[:, :3]
    gs_world.from_xyz(xyz_world)

    # Rotations: change basis (p3d->cv) then apply camera rotation.
    R_c2w = c2w[:3, :3]
    M = torch.diag(Mv)
    rots_cam = gs_world.get_rotation  # (N,4) wxyz

    # Convert quaternions to rotation matrices using scipy (batched)
    rots_cam_np = rots_cam.detach().cpu().numpy()  # (N, 4) wxyz
    # Convert wxyz to xyzw for scipy
    rots_cam_xyzw = np.stack([rots_cam_np[:, 1], rots_cam_np[:, 2], rots_cam_np[:, 3], rots_cam_np[:, 0]], axis=1)
    R_gauss_p3d_np = R.from_quat(rots_cam_xyzw).as_matrix()  # (N, 3, 3)
    R_gauss_p3d = torch.from_numpy(R_gauss_p3d_np.astype(np.float32)).to(device=device, dtype=dtype)

    R_gauss_cv = M @ R_gauss_p3d @ M
    R_gauss_world = R_c2w.unsqueeze(0) @ R_gauss_cv

    # Convert rotation matrices back to quaternions using scipy
    R_gauss_world_np = R_gauss_world.detach().cpu().numpy()  # (N, 3, 3)
    rots_world_xyzw = R.from_matrix(R_gauss_world_np).as_quat()  # (N, 4) xyzw
    # Convert xyzw back to wxyz
    rots_world_np = np.stack([rots_world_xyzw[:, 3], rots_world_xyzw[:, 0], rots_world_xyzw[:, 1], rots_world_xyzw[:, 2]], axis=1)
    rots_world = torch.from_numpy(rots_world_np.astype(np.float32)).to(device=device, dtype=dtype)
    gs_world.from_rotation(rots_world)

    return gs_world


def apply_pose_to_gaussian(gs_original, rotation, translation, scale, *, flip_facing: bool = False):
    """
    Apply pose transformation to a Gaussian splat.

    Args:
        gs_original: Original Gaussian splat object
        rotation: Rotation matrix or quaternion (3x3 or 4-element array/tensor)
        translation: Translation vector (3-element array/tensor)
        scale: Scale factor (scalar or 3-element array/tensor)

    Returns:
        Transformed Gaussian splat object
    """
    # Create a copy to avoid modifying the original
    gs_transformed = copy.deepcopy(gs_original)

    # Apply pose using SAM3D's convention (PyTorch3D camera coords) via scipy transforms.
    xyz_local = gs_transformed.get_xyz  # (N,3) canonical
    xyz_cam, quat_l2c = _apply_pose_to_points_l2c(
        xyz_local, rotation, translation, scale, flip_facing=flip_facing
    )
    gs_transformed.from_xyz(xyz_cam)

    # Rotate gaussians: new = pose * gaussian
    # Using scipy for quaternion multiplication (consistent with run_full_pipeline.py)
    rots = gs_transformed.get_rotation  # (N,4) wxyz
    device = rots.device
    dtype = rots.dtype

    # Convert to numpy and xyzw format
    rots_np = rots.detach().cpu().numpy()  # (N, 4) wxyz
    rots_xyzw = np.stack([rots_np[:, 1], rots_np[:, 2], rots_np[:, 3], rots_np[:, 0]], axis=1)

    quat_l2c_np = quat_l2c.detach().cpu().numpy().squeeze()  # (4,) wxyz
    quat_pose_xyzw = np.array([quat_l2c_np[1], quat_l2c_np[2], quat_l2c_np[3], quat_l2c_np[0]])

    # Quaternion multiply: pose * gaussian for each gaussian
    r_pose = R.from_quat(quat_pose_xyzw)
    r_gauss = R.from_quat(rots_xyzw)
    r_new = r_pose * r_gauss  # Compose rotations
    rots_new_xyzw = r_new.as_quat()  # (N, 4) xyzw

    if flip_facing:
        # 180 degree yaw around Y axis: quaternion (w=0, x=0, y=1, z=0) in wxyz -> (0, 1, 0, 0) in xyzw
        quat_yaw180_xyzw = np.array([0.0, 1.0, 0.0, 0.0])
        r_yaw = R.from_quat(quat_yaw180_xyzw)
        r_new = r_yaw * R.from_quat(rots_new_xyzw)
        rots_new_xyzw = r_new.as_quat()

    # Convert back to wxyz and torch
    rots_new_wxyz = np.stack([rots_new_xyzw[:, 3], rots_new_xyzw[:, 0], rots_new_xyzw[:, 1], rots_new_xyzw[:, 2]], axis=1)
    rots_new = torch.from_numpy(rots_new_wxyz.astype(np.float32)).to(device=device, dtype=dtype)
    gs_transformed.from_rotation(rots_new)

    # Scale gaussian sizes consistently with pose scale (helps appearance/GLB match).
    _, _, scale_l2c = _as_pose_tensors(
        rotation, translation, scale, device=xyz_local.device, dtype=xyz_local.dtype
    )
    scale_factor = scale_l2c.mean()
    gs_transformed.from_scaling(gs_transformed.get_scaling * scale_factor)
    gs_transformed.mininum_kernel_size *= float(scale_factor)

    return gs_transformed


def main():
    parser = argparse.ArgumentParser(description="Track and reconstruct objects from video using SAM3")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--frames_dir", type=str, default=None,
                        help="Directory of pre-extracted frames (PNG/JPG). If provided, "
                             "loads frames from here instead of decoding --video.")
    parser.add_argument("--prompt", type=str, default="object held in either hand", help="Text prompt for SAM3")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for reconstructions")
    parser.add_argument("--camera_dir", type=str, required=True, help="Directory containing TTT3R camera poses")
    parser.add_argument("--stride", type=int, default=1, help="Stride for frame processing")
    parser.add_argument(
        "--flip_facing",
        action="store_true",
        help="Flip object facing (180° yaw in camera space) for thin objects (e.g., TVs).",
    )
    # Detection strategy arguments
    parser.add_argument(
        "--detection_strategy",
        type=str,
        choices=["simple", "two_stage", "backward", "consensus", "combined", "hand_point", "grounding_dino"],
        default="simple",
        help="Object detection strategy: "
             "simple (text prompt from frame 0), "
             "two_stage (hand -> object in hand region), "
             "backward (find best frame, track backward), "
             "consensus (multi-frame sampling), "
             "combined (two_stage + backward), "
             "hand_point (use VITRA hand joints as point prompts), "
             "grounding_dino (GroundingDINO open-vocab detector for bbox prompts)"
    )
    parser.add_argument(
        "--hand_bbox_scale",
        type=float,
        default=2.0,
        help="Scale factor for hand bounding box expansion (for two_stage/combined strategies)"
    )
    parser.add_argument(
        "--best_frame_metric",
        type=str,
        choices=["mask_area", "confidence", "hand_overlap"],
        default="mask_area",
        help="Metric for selecting best frame (for backward/consensus/combined strategies)"
    )
    parser.add_argument(
        "--vitra_annotation",
        type=str,
        default=None,
        help="Path to VITRA .npy annotation file. If provided, extracts object "
             "noun from annotation text to use as prompt (overrides --prompt)"
    )
    parser.add_argument(
        "--num_sample_frames",
        type=int,
        default=5,
        help="Number of frames to sample for hand_point detection strategy (default: 5)"
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=None,
        help="Number of diffusion steps for 3D reconstruction (default: 25 per stage). "
             "Lower values (e.g. 12) are faster with modest quality loss."
    )
    parser.add_argument(
        "--early_exit_score",
        type=float,
        default=0.8,
        help="Stop sampling frames once detection score exceeds this threshold (default: 0.8)"
    )
    parser.add_argument(
        "--use_llm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use Qwen2.5-7B to extract object noun for detection prompts. "
             "Default: True for hand_point strategy, False otherwise. "
             "Use --no-use_llm to disable."
    )
    args = parser.parse_args()

    # Auto-extract prompt and hand joints from VITRA annotation if provided
    hand_joints = None
    intrinsics = None
    kept_frames = None
    active_hand = None

    if args.vitra_annotation:
        annotation = np.load(args.vitra_annotation, allow_pickle=True).item()

        # Get action text (left hand first, then right)
        text_data = annotation.get('text', {})
        action_text = None
        for hand in ['left', 'right']:
            actions = text_data.get(hand, [])
            if actions:
                action_text = actions[0][0]  # First action's text
                active_hand = hand
                break

        if action_text:
            extracted_prompt = extract_object_noun(action_text)
            print(f"Extracted prompt from VITRA: '{action_text}' → '{extracted_prompt}'")
            args.prompt = extracted_prompt
        else:
            print(f"Warning: No action text found in VITRA annotation, using default prompt: '{args.prompt}'")

        # Extract hand joint data for hand_point detection
        intrinsics = annotation.get('intrinsics')  # (3, 3)

        # Try to get hand joints from the active hand (the one performing the action)
        if active_hand and active_hand in annotation:
            hand_data = annotation[active_hand]
            hand_joints = hand_data.get('joints_camspace')  # (T, 21, 3)
            kept_frames = hand_data.get('kept_frames')  # Array of 0s and 1s
            if hand_joints is not None:
                print(f"Loaded {active_hand} hand joints: shape {hand_joints.shape}")
                if kept_frames is not None:
                    valid_count = np.sum(kept_frames == 1) if isinstance(kept_frames, np.ndarray) else sum(kept_frames)
                    print(f"  Valid hand frames: {valid_count}/{len(kept_frames)}")

        # If no active hand or no joints, try the other hand
        if hand_joints is None:
            for hand in ['left', 'right']:
                if hand in annotation:
                    hand_data = annotation[hand]
                    hand_joints = hand_data.get('joints_camspace')
                    kept_frames = hand_data.get('kept_frames')
                    if hand_joints is not None:
                        print(f"Using {hand} hand joints (fallback): shape {hand_joints.shape}")
                        active_hand = hand
                        break

        # Auto-select hand_point strategy if we have valid hand data
        if hand_joints is not None and intrinsics is not None:
            if args.detection_strategy == "simple":
                print("Auto-selecting 'hand_point' detection strategy (VITRA hand data available)")
                args.detection_strategy = "hand_point"
        else:
            if args.detection_strategy == "hand_point":
                print("Warning: hand_point strategy requested but VITRA hand data not available")
                print("  Falling back to 'simple' detection strategy")
                args.detection_strategy = "simple"

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize accelerator
    accelerator = Accelerator()
    device = accelerator.device
    print(f"Using device: {device}")

    # Initialize Inference model
    print("Initializing Inference model...")
    tag = "hf"
    config_path = f"checkpoints/{tag}/pipeline.yaml"
    inference = Inference(config_path, compile=False)

    # Load SAM3 video model and processor
    print("Loading SAM3 video model...")
    try:
        model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
        processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load video frames
    if args.frames_dir is not None:
        # Load pre-extracted frames from directory
        print(f"Loading frames from directory: {args.frames_dir}")
        import glob as _glob
        frame_paths = sorted(_glob.glob(os.path.join(args.frames_dir, "*.png"))
                             + _glob.glob(os.path.join(args.frames_dir, "*.jpg")))
        if not frame_paths:
            print(f"ERROR: No PNG/JPG frames found in {args.frames_dir}")
            return
        video_frames = [Image.open(p).convert("RGB") for p in frame_paths]
        print(f"  Loaded {len(video_frames)} frames from directory")
    else:
        print(f"Loading video: {args.video}")
        try:
            video_frames, _ = load_video(args.video)
        except Exception as e:
            print(f"  transformers load_video failed ({e}), trying OpenCV...")
            cap = cv2.VideoCapture(args.video)
            video_frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                video_frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            cap.release()
            if not video_frames:
                print(f"ERROR: Could not load any frames from {args.video}")
                return
            print(f"  Loaded {len(video_frames)} frames via OpenCV")

    # Run object detection using selected strategy
    print(f"\n=== Running {args.detection_strategy} detection strategy ===")
    print(f"Text prompt: {args.prompt}")

    # Resolve use_llm default: True for hand_point, False otherwise
    use_llm = args.use_llm
    if use_llm is None:
        use_llm = (args.detection_strategy == "hand_point")

    # Build kwargs for detection
    detection_kwargs = {
        "bbox_scale": args.hand_bbox_scale,
        "score_metric": args.best_frame_metric,
        "num_sample_frames": args.num_sample_frames,
        "early_exit_score": args.early_exit_score,
        "use_llm": use_llm,
    }

    # Add hand joint data if available (for hand_point strategy)
    if hand_joints is not None:
        detection_kwargs["hand_joints"] = hand_joints
    if intrinsics is not None:
        detection_kwargs["intrinsics"] = intrinsics
    if kept_frames is not None:
        detection_kwargs["kept_frames"] = kept_frames

    detection_result = detect_object(
        model=model,
        processor=processor,
        video_frames=video_frames,
        prompt=args.prompt,
        strategy=args.detection_strategy,
        device=device,
        dtype=torch.bfloat16,
        **detection_kwargs,
    )

    print(f"\nDetection complete:")
    print(f"  Strategy: {detection_result.strategy_used}")
    print(f"  Best frame: {detection_result.best_frame_idx}")
    print(f"  Frames with masks: {len(detection_result.masks)}")
    if detection_result.scores:
        avg_score = sum(detection_result.scores.values()) / len(detection_result.scores)
        print(f"  Average score: {avg_score:.4f}")

    # Convert detection result to outputs_per_frame format for compatibility
    outputs_per_frame = {}
    for frame_idx, mask in detection_result.masks.items():
        # Create a mock output structure matching the original format
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        outputs_per_frame[frame_idx] = {
            'masks': mask_tensor,
            'object_ids': torch.tensor(detection_result.object_ids if detection_result.object_ids else [0]),
            'scores': torch.tensor([detection_result.scores.get(frame_idx, 1.0)]),
        }

    print(f"\nProcessed {len(outputs_per_frame)} frames")

    # Reconstruct objects for each frame
    print("Reconstructing objects for each frame...")

    # Dictionary to track which objects we've seen
    object_seen = {}

    # Iterate through frames in order with stride
    sorted_frames = sorted(outputs_per_frame.keys())
    # Subsample frames based on stride
    frames_to_process = sorted_frames[::args.stride]

    # Prefer the best detection frame for reconstruction
    best_frame = detection_result.best_frame_idx
    if best_frame is not None and best_frame in outputs_per_frame:
        frames_to_process = [best_frame]
    else:
        frames_to_process = frames_to_process[:1]

    if not frames_to_process:
        print("No frames with masks to reconstruct. Detection may have failed.")
        return

    print(f"Processing every {args.stride}th frame. Total frames to reconstruct: {len(frames_to_process)}")

    # Helper class for slicing Gaussian Splats
    for frame_idx in frames_to_process:
        results = outputs_per_frame[frame_idx]
        masks = results['masks']

        if masks.ndim == 4:
            masks = masks.squeeze(1)

        if len(masks) > 0:
            # Get image for this frame
            image_pil = video_frames[frame_idx]
            image_np = np.array(image_pil)  # RGB

            for obj_idx, mask_tensor in enumerate(masks):
                # Check if object is present in this frame (mask area > 0)
                if mask_tensor.sum() > 0:
                    mask_np = mask_tensor.cpu().numpy().astype(bool)  # Ensure boolean

                    # Track first frame for this object
                    if obj_idx not in object_seen:
                        object_seen[obj_idx] = frame_idx

                    print(f"Reconstructing Object {obj_idx} frame {frame_idx}")

                    # Do full reconstruction for this frame
                    output = inference(
                        image_np, mask_np, seed=42,
                        stage1_inference_steps=args.inference_steps,
                        stage2_inference_steps=args.inference_steps,
                    )

                    # Load camera pose for this frame
                    # TTT3R saves camera poses sequentially (0, 1, 2...) even if stride is used
                    # So frame 0 -> 000000.npz, frame 8 -> 000001.npz (if stride 8)
                    ttt3r_idx = frame_idx // args.stride
                    camera_to_world, intrinsics = load_camera_pose(ttt3r_idx, args.camera_dir)

                    # Transform reconstruction to world coordinates
                    obj_dir = os.path.join(args.output_dir, f"object_{obj_idx}")
                    os.makedirs(obj_dir, exist_ok=True)

                    # Apply pose to Gaussian splat (canonical -> camera space)
                    gs_camera = apply_pose_to_gaussian(
                        output['gs'],
                        output['rotation'],
                        output['translation'],
                        output['scale'],
                        flip_facing=args.flip_facing,
                    )
                    
                    camera_coords_ply_path = os.path.join(obj_dir, f"frame_{frame_idx:04d}_camera_coords.ply")
                    gs_camera.save_ply(camera_coords_ply_path)
                    print(f"Saved camera coordinates to {camera_coords_ply_path}")

                    # Transform to world coordinates (camera -> world)
                    gs_world = transform_gaussian_to_world(gs_camera, camera_to_world)

                    # Save the reconstruction in world coordinates
                    ply_path = os.path.join(obj_dir, f"frame_{frame_idx:04d}.ply")
                    gs_world.save_ply(ply_path)

                    glb_path = os.path.join(obj_dir, f"frame_{frame_idx:04d}.glb")

                    # Export as GLB
                    print(f"Exporting GLB for Object {obj_idx} frame {frame_idx}...")
                    # Check if mesh is available in output

                    # Export GLB with the Gaussian representation
                    glb_mesh = output['glb']

                    # Transform GLB mesh to match the Gaussian splat coordinate system
                    # The mesh is in canonical space and needs:
                    # 0. Undo the Y-up rotation applied by to_glb() function
                    # 1. Pose transformation (canonical -> camera space)
                    # 2. Coordinate conversion (OpenGL -> OpenCV)
                    # 3. Camera-to-world transformation

                    # Transform the GLB mesh into the same camera/world frames as the Gaussian.
                    vertices = glb_mesh.vertices.copy()

                    # Step 1: Undo Y-up rotation to get back to Z-up canonical space
                    # to_glb() applies: Z-up -> Y-up via [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
                    # Inverse is: Y-up -> Z-up via [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
                    y_up_to_z_up = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
                    vertices = vertices @ y_up_to_z_up

                    # Step 2: Apply SAM3D pose (local->camera) using the same math as gaussians.
                    verts_t = torch.from_numpy(vertices.astype(np.float32))
                    verts_cam, _ = _apply_pose_to_points_l2c(
                        verts_t, output["rotation"], output["translation"], output["scale"], flip_facing=args.flip_facing
                    )
                    vertices = verts_cam.detach().cpu().numpy()

                    # Step 3: Convert from PyTorch3D camera convention (x left, y up, z inwards)
                    # to OpenCV camera convention (x right, y down, z forward).
                    vertices[:, 0] *= -1
                    vertices[:, 1] *= -1
                    vertices[:, 2] *= 1

                    camera_coords_glb_path = os.path.join(obj_dir, f"frame_{frame_idx:04d}_camera_coords.glb")
                    glb_mesh.vertices = vertices
                    glb_mesh.export(camera_coords_glb_path)
                    print(f"Saved camera coordinates GLB to {camera_coords_glb_path}")

                    # Step 4: Transform to world coordinates
                    vertices_homogeneous = np.concatenate([vertices, np.ones((vertices.shape[0], 1))], axis=1)
                    vertices_world = (vertices_homogeneous @ camera_to_world.T)[:, :3]

                    # Update and save the GLB mesh
                    glb_mesh.vertices = vertices_world
                    glb_mesh.export(glb_path)
                    print(f"Saved GLB to {glb_path}")

                    # Save mask visualization
                    mask_vis = image_np.copy()
                    mask_overlay = np.zeros_like(mask_vis)
                    mask_overlay[mask_np] = [255, 0, 0]  # Red highlight
                    mask_vis = cv2.addWeighted(mask_vis, 0.6, mask_overlay, 0.4, 0)

                    # Draw bounding box
                    y_indices, x_indices = np.where(mask_np)
                    if len(y_indices) > 0:
                        y_min, y_max = y_indices.min(), y_indices.max()
                        x_min, x_max = x_indices.min(), x_indices.max()
                        cv2.rectangle(mask_vis, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    mask_path = os.path.join(obj_dir, f"frame_{frame_idx:04d}_mask.jpg")
                    cv2.imwrite(mask_path, cv2.cvtColor(mask_vis, cv2.COLOR_RGB2BGR))

                    # Save binary mask for depth sampling
                    mask_binary_path = os.path.join(obj_dir, f"frame_{frame_idx:04d}_mask.npy")
                    np.save(mask_binary_path, mask_np)

                    # print(f"Object {obj_idx} pose - Translation: {output['translation']}, Rotation: {output['rotation']}, Scale: {output['scale']}")
                    # np.save(os.path.join(obj_dir, "translation.npy"), output["translation"].detach().cpu().numpy())
                    # np.save(os.path.join(obj_dir, "rotation.npy"), output["rotation"].detach().cpu().numpy())
                    # np.save(os.path.join(obj_dir, "scale.npy"), output["scale"].detach().cpu().numpy())

                    print(f"Saved reconstruction to {ply_path}")

    print(f"\n\nFinished! Reconstructions saved to {args.output_dir}/")
    print(f"Total objects tracked: {len(object_seen)}")
    for obj_idx, first_frame in object_seen.items():
        obj_dir = os.path.join(args.output_dir, f"object_{obj_idx}")
        num_frames = len([f for f in os.listdir(obj_dir) if f.endswith('.ply')])
        print(f"  Object {obj_idx}: {num_frames} frames (first frame: {first_frame})")


if __name__ == "__main__":
    main()
