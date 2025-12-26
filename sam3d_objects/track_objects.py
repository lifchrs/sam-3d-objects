import cv2
import torch
import numpy as np
from PIL import Image
from transformers import Sam3VideoModel, Sam3VideoProcessor
from transformers.video_utils import load_video
from accelerate import Accelerator
import sys
import os
from scipy.spatial.transform import Rotation as R
import copy

import argparse

# Add notebook to path for inference imports
sys.path.append("notebook")
from inference import Inference



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

    # Get the Gaussian centers (xyz positions) in camera coordinates
    if hasattr(gs_world, '_xyz'):
        xyz_camera = gs_world._xyz.detach().cpu().numpy()
    elif hasattr(gs_world, 'xyz'):
        xyz_camera = gs_world.xyz.detach().cpu().numpy()
    else:
        raise AttributeError("Gaussian splat object doesn't have xyz attribute")

    # Transform positions to world coordinates
    # SAM3D outputs in OpenGL convention (Y Up, -Z Forward)
    # TTT3R uses OpenCV convention (Y Down, Z Forward) - X Right.
    # We need to convert from OpenGL (R, U, B) to OpenCV (R, D, F).
    # Transformation: diag(1, -1, -1) [Scale Y by -1, Scale Z by -1]
    
    # 1. Apply Coordinate conversion
    # x_new = x_old, y_new = -y_old, z_new = -z_old
    xyz_camera_cv = xyz_camera.copy()
    xyz_camera_cv[:, 0] *= 1   # X stays same (Right -> Right)
    xyz_camera_cv[:, 1] *= -1  # Y flips (Up -> Down)
    xyz_camera_cv[:, 2] *= -1  # Z flips (Back -> Forward)
    
    # 2. Add homogeneous coordinate
    xyz_homogeneous = np.concatenate([xyz_camera_cv, np.ones((xyz_camera_cv.shape[0], 1))], axis=1)
    
    # 3. Apply Camera-to-World transformation
    xyz_world_homogeneous = xyz_homogeneous @ camera_to_world.T
    xyz_world = xyz_world_homogeneous[:, :3]

    # Update the Gaussian centers
    if hasattr(gs_world, '_xyz'):
        gs_world._xyz = torch.from_numpy(xyz_world).to(gs_camera._xyz.device).to(gs_camera._xyz.dtype)
    elif hasattr(gs_world, 'xyz'):
        gs_world.xyz = torch.from_numpy(xyz_world).to(gs_camera.xyz.device).to(gs_camera.xyz.dtype)

    # Transform rotation quaternions
    if hasattr(gs_world, '_rotation'):
        # Extract rotation from camera_to_world (upper-left 3x3)
        camera_rotation_matrix = camera_to_world[:3, :3]

        original_rots = gs_world._rotation.detach().cpu().numpy()

        # Transform each Gaussian's rotation
        transformed_rots = []
        for quat in original_rots:
            # Convert Gaussian rotation to matrix (quat format: w, x, y, z)
            gs_rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
            
            # Apply Coordinate conversion to rotation
            # M = diag(1, -1, -1)
            M = np.diag([1, -1, -1])
            gs_rot_cv = M @ gs_rot
            
            # Compose: world_rotation = camera_rotation @ gaussian_rotation_cv
            new_rot = camera_rotation_matrix @ gs_rot_cv
            # Convert back to quaternion
            new_quat = R.from_matrix(new_rot).as_quat()
            # Convert to (w, x, y, z) format
            transformed_rots.append([new_quat[3], new_quat[0], new_quat[1], new_quat[2]])

        gs_world._rotation = torch.from_numpy(np.array(transformed_rots)).to(gs_camera._rotation.device).to(gs_camera._rotation.dtype)

    return gs_world


def apply_pose_to_gaussian(gs_original, rotation, translation, scale):
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

    # Convert all inputs to numpy arrays if they're tensors
    if torch.is_tensor(rotation):
        rotation = rotation.detach().cpu().numpy()
    if torch.is_tensor(translation):
        translation = translation.detach().cpu().numpy()
    if torch.is_tensor(scale):
        scale = scale.detach().cpu().numpy()

    # Squeeze out any batch dimensions
    rotation = np.squeeze(rotation)
    translation = np.squeeze(translation)
    if hasattr(scale, 'squeeze'):
        scale = np.squeeze(scale)

    # Ensure scale is numeric
    if hasattr(scale, 'item') and scale.size == 1:
        scale = scale.item()

    # Convert rotation to matrix if needed
    rotation_shape = tuple(rotation.shape)
    if rotation_shape == (4,):
        # Quaternion format (w, x, y, z)
        rot_matrix = R.from_quat([rotation[1], rotation[2], rotation[3], rotation[0]]).as_matrix()
    elif rotation_shape == (3, 3):
        rot_matrix = rotation
    else:
        raise ValueError(f"Unexpected rotation shape: {rotation_shape}")

    # Get the Gaussian centers (xyz positions)
    if hasattr(gs_transformed, '_xyz'):
        xyz = gs_transformed._xyz.detach().cpu().numpy()
    elif hasattr(gs_transformed, 'xyz'):
        xyz = gs_transformed.xyz.detach().cpu().numpy()
    else:
        raise AttributeError("Gaussian splat object doesn't have xyz attribute")

    # Apply transformations: first scale, then rotate, then translate
    # Scale
    if isinstance(scale, (int, float, np.integer, np.floating)):
        xyz_transformed = xyz * scale
    else:
        xyz_transformed = xyz * scale.reshape(1, 3)

    # Rotate
    xyz_transformed = xyz_transformed @ rot_matrix.T

    # Translate
    xyz_transformed = xyz_transformed + translation.reshape(1, 3)

    # Update the Gaussian centers
    if hasattr(gs_transformed, '_xyz'):
        gs_transformed._xyz = torch.from_numpy(xyz_transformed).to(gs_original._xyz.device).to(gs_original._xyz.dtype)
    elif hasattr(gs_transformed, 'xyz'):
        gs_transformed.xyz = torch.from_numpy(xyz_transformed).to(gs_original.xyz.device).to(gs_original.xyz.dtype)

    # Also transform rotation quaternions if they exist
    if hasattr(gs_transformed, '_rotation'):
        # The Gaussian splat rotations need to be composed with the pose rotation
        original_rots = gs_transformed._rotation.detach().cpu().numpy()

        # Convert each Gaussian's rotation quaternion and compose with pose rotation
        transformed_rots = []
        for quat in original_rots:
            # Convert Gaussian rotation to matrix
            gs_rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
            # Compose: new_rotation = pose_rotation @ gaussian_rotation
            new_rot = rot_matrix @ gs_rot
            # Convert back to quaternion
            new_quat = R.from_matrix(new_rot).as_quat()
            # Convert to (w, x, y, z) format
            transformed_rots.append([new_quat[3], new_quat[0], new_quat[1], new_quat[2]])

        gs_transformed._rotation = torch.from_numpy(np.array(transformed_rots)).to(gs_original._rotation.device).to(gs_original._rotation.dtype)

    # Scale the Gaussian scales if they exist
    if hasattr(gs_transformed, '_scaling'):
        if isinstance(scale, (int, float)):
            gs_transformed._scaling = gs_original._scaling * scale
        else:
            # If scale is a 3D vector, apply it to the Gaussian scales
            scale_factor = np.mean(scale)  # Use average scale
            gs_transformed._scaling = gs_original._scaling * scale_factor

    return gs_transformed


def main():
    parser = argparse.ArgumentParser(description="Track and reconstruct objects from video using SAM3")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--prompt", type=str, default="object held in either hand", help="Text prompt for SAM3")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for reconstructions")
    parser.add_argument("--camera_dir", type=str, required=True, help="Directory containing TTT3R camera poses")
    parser.add_argument("--stride", type=int, default=1, help="Stride for frame processing")
    args = parser.parse_args()

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
    print(f"Loading video: {args.video}")
    try:
        video_frames, _ = load_video(args.video)
    except Exception as e:
        print(f"Error loading video frames: {e}")
        return

    # Initialize video inference session
    print("Initializing video inference session...")
    inference_session = processor.init_video_session(
        video=video_frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=torch.bfloat16,
    )

    # Add text prompt
    print(f"Adding text prompt: {args.prompt}")
    inference_session = processor.add_text_prompt(
        inference_session=inference_session,
        text=args.prompt,
    )

    # Process frames
    print("Processing frames with SAM3...")
    outputs_per_frame = {}

    for model_outputs in model.propagate_in_video_iterator(
        inference_session=inference_session, max_frame_num_to_track=None
    ):
        processed_outputs = processor.postprocess_outputs(inference_session, model_outputs)
        outputs_per_frame[model_outputs.frame_idx] = processed_outputs
        print(f"Processed frame {model_outputs.frame_idx}", end='\r')

    print(f"\nProcessed {len(outputs_per_frame)} frames")

    # DEBUG MODE: Reconstruct each frame independently
    print("Reconstructing objects for each frame (DEBUG MODE - no pose tracking)...")

    # Dictionary to track which objects we've seen
    object_seen = {}

    # Iterate through frames in order with stride
    sorted_frames = sorted(outputs_per_frame.keys())
    # Subsample frames based on stride
    frames_to_process = sorted_frames[::args.stride]
    
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
                    mask_np = mask_tensor.cpu().numpy()  # Boolean

                    # Track first frame for this object
                    if obj_idx not in object_seen:
                        object_seen[obj_idx] = frame_idx

                    print(f"Reconstructing Object {obj_idx} frame {frame_idx}")

                    # Do full reconstruction for this frame
                    output = inference(image_np, mask_np, seed=42)

                    # Load camera pose for this frame
                    # TTT3R saves camera poses sequentially (0, 1, 2...) even if stride is used
                    # So frame 0 -> 000000.npz, frame 8 -> 000001.npz (if stride 8)
                    ttt3r_idx = frame_idx // args.stride
                    camera_to_world, intrinsics = load_camera_pose(ttt3r_idx, args.camera_dir)

                    # Transform reconstruction to world coordinates
                    gs_world = transform_gaussian_to_world(output['gs'], camera_to_world)

                    # Save the reconstruction in world coordinates
                    obj_dir = os.path.join(args.output_dir, f"object_{obj_idx}")
                    os.makedirs(obj_dir, exist_ok=True)

                    ply_path = os.path.join(obj_dir, f"frame_{frame_idx:04d}.ply")
                    gs_world.save_ply(ply_path)

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

                    print(f"Saved reconstruction to {ply_path}")

    print(f"\n\nFinished! Reconstructions saved to {args.output_dir}/")
    print(f"Total objects tracked: {len(object_seen)}")
    for obj_idx, first_frame in object_seen.items():
        obj_dir = os.path.join(args.output_dir, f"object_{obj_idx}")
        num_frames = len([f for f in os.listdir(obj_dir) if f.endswith('.ply')])
        print(f"  Object {obj_idx}: {num_frames} frames (first frame: {first_frame})")


if __name__ == "__main__":
    main()
