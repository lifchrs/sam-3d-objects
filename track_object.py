import cv2
import torch
import numpy as np
from PIL import Image
from transformers import Sam3VideoModel, Sam3VideoProcessor
from transformers.video_utils import load_video
from accelerate import Accelerator

# Hardcoded paths and prompts
VIDEO_PATH = "/home/ap977/GRILL/new_project/factory_005_worker_001_00000_2sec.mp4"  # Replace with actual video path
VIDEO_PATH = "/home/ap977/GRILL/new_project/factory_003_worker_001_00000_2sec.mp4"  # Replace with actual video path
TEXT_PROMPT = "object held in either hand"          # Replace with actual text prompt
OUTPUT_VIDEO_PATH = "output_video_sam/{}".format(VIDEO_PATH.split('/')[-1])

def main():
    # Initialize accelerator
    accelerator = Accelerator()
    device = accelerator.device
    print(f"Using device: {device}")

    # Load SAM3 video model and processor
    print("Loading SAM3 video model...")
    try:
        model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
        processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load video frames
    print(f"Loading video: {VIDEO_PATH}")
    try:
        video_frames, _ = load_video(VIDEO_PATH)
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
    print(f"Adding text prompt: {TEXT_PROMPT}")
    inference_session = processor.add_text_prompt(
        inference_session=inference_session,
        text=TEXT_PROMPT,
    )

    # Process frames
    print("Processing frames...")
    outputs_per_frame = {}
    for model_outputs in model.propagate_in_video_iterator(
        inference_session=inference_session, max_frame_num_to_track=None # Track all frames
    ):
        processed_outputs = processor.postprocess_outputs(inference_session, model_outputs)
        outputs_per_frame[model_outputs.frame_idx] = processed_outputs
        print(f"Processed frame {model_outputs.frame_idx}", end='\r')
    
    print(f"\nProcessed {len(outputs_per_frame)} frames")

    # Visualization and saving
    print("Creating output video...")
    
    # Get video properties from original file for writer
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    for i, frame_pil in enumerate(video_frames):
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        
        if i in outputs_per_frame:
            results = outputs_per_frame[i]
            # results['masks'] is a boolean tensor of shape (num_objects, height, width)
            # or (num_objects, 1, height, width) depending on postprocessing
            
            masks = results['masks']
            if len(masks) > 0:
                 # Ensure masks are in (num_objects, height, width) format
                if masks.ndim == 4:
                    masks = masks.squeeze(1)
                
                # Combine masks
                combined_mask = torch.any(torch.tensor(masks), dim=0).cpu().numpy()
                
                # Create overlay
                mask_overlay = np.zeros_like(frame)
                mask_overlay[combined_mask] = [0, 0, 255] # Red

                # Blend
                alpha = 0.5
                frame = cv2.addWeighted(frame, 1, mask_overlay, alpha, 0)
        
        out.write(frame)

    out.release()
    print(f"Finished processing. Output saved to {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    main()
