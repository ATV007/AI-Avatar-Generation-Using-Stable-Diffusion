import torch
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model
from transformers import CLIPTokenizer
from PIL import Image
import cv2
import numpy as np
from facenet_pytorch import MTCNN
from basicsr.utils.download_util import load_file_from_url
from gfpgan import GFPGANer
from insightface.app import FaceAnalysis
from insightface.model_zoo.inswapper import INSwapper  # ✅ Correct import
import onnxruntime
from google.colab import files
import os

# ✅ Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ✅ Load Stable Diffusion 2.1 Model
model_name = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(
    model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
pipe.to(device)

# ✅ Enable VRAM optimizations
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_attention_slicing()

# ✅ Load Tokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# ✅ Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True, device=device)

# ✅ Initialize InsightFace for face analysis
face_app = FaceAnalysis(
    name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
face_app.prepare(ctx_id=0, det_size=(640, 640))

# ✅ Load the InSwapper model for face swapping
inswapper_model_path = "inswapper_128.onnx"
inswapper = INSwapper("inswapper_128.onnx")
print("✅ InSwapper model loaded successfully!")


def apply_color_correction(original_face, swapped_face):
    """Matches the skin tone of the swapped face with the original face."""
    original_lab = cv2.cvtColor(original_face, cv2.COLOR_RGB2LAB)
    swapped_lab = cv2.cvtColor(swapped_face, cv2.COLOR_RGB2LAB)

    # Resize swapped face channels to match original face size
    swapped_lab = cv2.resize(
        swapped_lab, (original_lab.shape[1], original_lab.shape[0])
    )

    l_orig, a_orig, b_orig = cv2.split(original_lab)
    l_swap, a_swap, b_swap = cv2.split(swapped_lab)

    l_swap = cv2.equalizeHist(l_swap)  # Adjust brightness
    a_swap = cv2.addWeighted(a_swap, 0.5, a_orig, 0.5, 0)  # Blend color
    b_swap = cv2.addWeighted(b_swap, 0.5, b_orig, 0.5, 0)

    corrected_lab = cv2.merge([l_swap, a_swap, b_swap])
    corrected_rgb = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2RGB)

    return corrected_rgb


def blend_faces(original, swapped):
    """Blends the swapped face into the original image using Gaussian blending."""
    mask = np.ones(swapped.shape, dtype=np.uint8) * 255
    blended = cv2.seamlessClone(
        swapped,
        original,
        mask,
        (original.shape[1] // 2, original.shape[0] // 2),
        cv2.NORMAL_CLONE,
    )
    return blended


def swap_face(
    original_face_path, generated_body_path, output_path="swapped_output.jpg"
):
    original_img = cv2.imread(original_face_path)
    generated_img = cv2.imread(generated_body_path)

    if original_img is None or generated_img is None:
        print("❌ Error: One or both images could not be loaded.")
        return None

    # Convert to RGB
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    generated_img = cv2.cvtColor(generated_img, cv2.COLOR_BGR2RGB)

    # Detect faces
    faces_original = face_app.get(original_img)
    faces_generated = face_app.get(generated_img)

    if faces_original and faces_generated:
        swapped_img = inswapper.get(
            generated_img, faces_generated[0], faces_original[0], paste_back=True
        )

        # ✅ Apply Color Correction
        swapped_img = apply_color_correction(original_img, swapped_img)

        # ✅ Blend the face naturally
        blended_img = blend_faces(original_img, swapped_img)

        # ✅ Convert back for saving
        blended_img = cv2.cvtColor(blended_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, blended_img)

        print(f"✅ Face swap successful! Saved as {output_path}")
        return output_path
    else:
        print("❌ No faces detected.")
        return None


# ✅ Generate Image with Stable Diffusion
def generate_image(prompt, negative_prompt="blurry, low quality"):
    with torch.no_grad():
        image = pipe(prompt, negative_prompt=negative_prompt).images[0]
    return image


# ✅ Upload and Process Image
input_image_path = "/content/tom-holland.jpg"  # Replace with actual uploaded file
if os.path.exists(input_image_path):
    print("✅ Face detected, proceeding with avatar generation...")
    prompt = "Create a detailed image of a military officer in a formal uniform with a stern yet composed expression. His face should have sharp contours, a strong jawline, and realistic textures, making it ideal for a seamless face swap."
    generated_image = generate_image(prompt)
    generated_image_path = "generated_avatar.png"
    generated_image.save(generated_image_path)

    # Swap the face
    final_image_path = swap_face(input_image_path, generated_image_path)

    if final_image_path:
        from IPython.display import display

        display(Image.open(final_image_path))
    else:
        print("❌ Face swap failed.")
else:
    print("❌ No file found. Please upload an image.")
