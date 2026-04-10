import os
import sys

import torch
from PIL import Image
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(BASE_DIR, "sam3")

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from sam3.model_builder import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

BPE_PATH = os.path.join(BASE_DIR, "sam3", "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")
IMG_PATH = os.path.join(BASE_DIR, "sam3", "assets", "images", "test_image.jpg")

print("BPE_PATH:", BPE_PATH)
print("IMG_PATH:", IMG_PATH)
print("BPE exists:", os.path.exists(BPE_PATH))
print("IMG exists:", os.path.exists(IMG_PATH))

print("Loading model...")
model = build_sam3_image_model(bpe_path=BPE_PATH)
model = model.to(device)

processor = Sam3Processor(model, confidence_threshold=0.5)

image = Image.open(IMG_PATH).convert("RGB")
width, height = image.size

state = processor.set_image(image)

processor.reset_all_prompts(state)
state = processor.set_text_prompt(state=state, prompt="shoe")

plot_results(image, state)
plt.show()

print("Done.")