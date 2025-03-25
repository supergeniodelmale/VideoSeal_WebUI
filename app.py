import os
from PIL import Image
import torch
import torchvision
from torchvision.transforms.functional import to_tensor, to_pil_image
import gradio as gr
import cv2
import numpy as np
from torchvision.io import read_video, write_video
from torchvision.transforms.functional import to_pil_image


# Download the model and load it.
os.makedirs("ckpts", exist_ok=True)
if not os.path.exists("ckpts/y_256b_img.jit"):
    os.system("wget https://dl.fbaipublicfiles.com/videoseal/y_256b_img.jit -P ckpts/")


# Load the JIT model.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load("ckpts/y_256b_img.jit")
model.to(device)
model.eval()




def embed_watermark_image(image, number):
  image.save("tmp.jpg", "JPEG", quality=100)
  img = Image.open("tmp.jpg").convert("RGB")
  img_o = to_tensor(img).unsqueeze(0).float().to(device)

  # Create a message to embed (random binary vector of 256bits).
  msg = number_to_tensor(number)
  print(msg)

  # Option 2: Embedding only.
  with torch.no_grad():
      # Returns watermarked image directly.
      img_w = model.embed(img_o, msg)

  # Convert back to PIL Image for saving.
  img_w_pil = to_pil_image(img_w.squeeze().cpu())

  save_path = img_path.split(".")[0] + "_wm.jpg"
  img_w_pil.save(save_path, "JPEG", quality=100)

  return save_path


def retrieve_watermark_image(image):
  img_w = image.convert("RGB")
  img_w = to_tensor(img_w).unsqueeze(0).float().to(device)
  with torch.no_grad():
      # Returns predictions tensor directly.
      preds = model.detect(img_w)
      
      # Process predictions to get binary message.
      # Assuming first channel is detection mask and rest are bit predictions.
      bit_preds = preds[:, 1:]  # Exclude mask
      detected_message = (bit_preds > 0).float()  # Threshold

      return tensor_to_number(detected_message)




def embed_watermark_video(video_path, number):
    # Read video returns a tuple of (video_frames, audio_frames, metadata)
    frames, audio, metadata = read_video(video_path, pts_unit='sec')
    
    # Convert to float and normalize to [0, 1]
    video_tensor = frames.float() / 255.0
    
    # Move to device and ensure format [T, C, H, W]
    video_tensor = video_tensor.permute(0, 3, 1, 2).to(device)

    # Create a message to embed (must be of shape 1xK for video)
    message = number_to_tensor(number)

    # Embed watermark in video
    with torch.no_grad():
        # Returns watermarked video directly
        watermarked_video = model.embed(video_tensor, message, is_video=True)

    # Convert back to uint8 for saving
    watermarked_video = (watermarked_video.cpu() * 255.0).to(torch.uint8)

    # Convert back to format expected by write_video [T, H, W, C]
    watermarked_video = watermarked_video.permute(0, 2, 3, 1)

    # Save watermarked video with original audio
    output_path = "watermarked_video.mp4"
    
    # 🔹 Fix: Convert `fps` to an integer to avoid PyTorch issues
    fps = int(metadata["video_fps"])  # Ensure it's a standard int

    # Debug print
    #print(f"Using FPS: {fps}, Type: {type(fps)}")

    # Corrected: Explicitly specify 'audio_array' parameter
    write_video(output_path, watermarked_video, fps)
    
    return output_path


def retrieve_watermark_video(video_path):
  # Read video returns a tuple of (video_frames, audio_frames, metadata)
  frames, audio, metadata = read_video(video_path, pts_unit='sec')
  # Convert to float and normalize to [0, 1].
  video_tensor = frames.float() / 255.0
  # Move to device and ensure format [T, C, H, W].
  video_tensor = video_tensor.permute(0, 3, 1, 2).to(device)
  # Detect message from watermarked video.
  with torch.no_grad():

    # Aggregate predictions across frames.
    aggregated_msg = model.detect_video_and_aggregate(
        video_tensor,
        aggregation="avg"  # Options: "avg", "squared_avg", "l1norm_avg", "l2norm_avg"
    )
    return tensor_to_number(aggregated_msg)





def number_to_tensor(num: int, device='cpu'):
    """
    Converts a number into a PyTorch tensor of shape (1, 256) with 0s and 1s.
    
    - Converts the number to its binary representation.
    - Pads with 0s to ensure a length of 256 bits.
    
    Args:
        num (int): Input number.
        device (str): Device to place the tensor on ('cpu' or 'cuda').
    
    Returns:
        torch.Tensor: A (1, 256) tensor with 0s and 1s.
    """
    # Convert number to binary and fill up to 256 bits
    binary_str = bin(num)[2:].zfill(256)
    binary_values = [int(bit) for bit in binary_str]
    
    # Convert to tensor
    tensor = torch.tensor(binary_values, dtype=torch.float32, device=device).unsqueeze(0)  # Shape (1, 256)
    
    return tensor

def tensor_to_number(tensor: torch.Tensor):
    """
    Converts a (1, 256) PyTorch tensor of 0s and 1s back to a number.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (1, 256) with 0s and 1s.
    
    Returns:
        int: Reconstructed number.
    """
    # Ensure tensor is in the correct shape
    binary_values = tensor.squeeze(0).tolist()  # Convert to list
    
    # Convert binary list to string
    binary_str = ''.join(str(int(bit)) for bit in binary_values)
    
    # Convert binary string back to integer
    num = int(binary_str, 2)
    
    return num













# Create the Gradio interface
def create_demo():
    with gr.Blocks() as demo:
        with gr.Tab("Images"):
            with gr.Column():
                with gr.Row():
                  gr.Markdown("### Insert Watermark")
                with gr.Row():
                  with gr.Column():
                    watermark_number_input = gr.Number(label="Enter Watermark Number")
                    watermark_btn = gr.Button("Insert Watermark")
                  with gr.Column():
                    img_input = gr.Image(type="pil", label="Upload Image (JPG Only!)")
                  with gr.Column():
                    img_output = gr.Image(label="Watermarked Image")

 


                    watermark_btn.click(embed_watermark_image, inputs=[img_input, watermark_number_input], outputs=img_output)
                with gr.Row():
                  gr.Markdown("### Retrieve Watermark")
                with gr.Column():
                    watermarked_img_input = gr.Image(type="pil", label="Upload Watermarked Image")
                with gr.Row():
                    watermark_retrieve_btn = gr.Button("Retrieve Watermark")
                    watermark_output = gr.Textbox(label="Retrieved Watermark")

                    watermark_retrieve_btn.click(retrieve_watermark_image, inputs=watermarked_img_input, outputs=watermark_output)


        """with gr.Tab("Videos"):
            with gr.Column():
                with gr.Row():
                    gr.Markdown("### Insert Watermark")
                with gr.Row():
                    with gr.Column():
                        watermark_number_video_input = gr.Number(label="Enter Watermark Number")
                        watermark_video_btn = gr.Button("Insert Watermark")
                    with gr.Column():
                        video_input = gr.Video(label="Upload Video (MP4 Only!)")
                    with gr.Column():
                        video_output = gr.Video(label="Watermarked Video")

                    watermark_video_btn.click(embed_watermark_video, inputs=[video_input, watermark_number_video_input], outputs=video_output)

                with gr.Row():
                    gr.Markdown("### Retrieve Watermark")
                with gr.Column():
                    watermarked_video_input = gr.Video(label="Upload Watermarked Video")
                with gr.Row():
                    watermark_retrieve_video_btn = gr.Button("Retrieve Watermark")
                    watermark_video_output = gr.Textbox(label="Retrieved Watermark")

                    watermark_retrieve_video_btn.click(retrieve_watermark_video, inputs=watermarked_video_input, outputs=watermark_video_output)"""




    return demo

demo = create_demo()
demo.launch(show_error=True)
