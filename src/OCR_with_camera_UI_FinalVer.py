import os
import platform
import warnings
from pathlib import Path
import pygame
import pygame.camera
from PIL import Image, ImageFilter, ImageOps
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import openai

# Set environment variable so PyTorch falls back to CPU kernels when an MPS operation isn't available (macOS).
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
HARDCODED_OPENAI_KEY = "" #insert API key here
# Accuracy toggle (enable heavier preprocessing and models)
OCR_ACCURACY_MODE = True  # set to False if you want faster but less accurate

# ----------------- Paths & constants -----------------
BASE_DIR = Path("") #enter your folder path here
# Exportable paths for easy configuration (strings). Change these if you share the code.
answer_path = str(BASE_DIR / "answer.txt")
chatgpt_answer_path = str(BASE_DIR / "chatgpt_answer.txt")
# Internal Path objects built from the exportable strings
CAPTURE_PATH = BASE_DIR / "downloaded_image.png"   # save camera capture as PNG
ANSWER_PATH = Path(answer_path)
CHATGPT_PATH = Path(chatgpt_answer_path)
DESKTOP_DIR = Path.home() / "Desktop"

SCREEN_W, SCREEN_H = 900, 600
FPS = 30

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (40, 40, 40)
LIGHT_GRAY = (200, 200, 200)

BLUE = (70, 120, 255)

# ----------------- Font helpers (Vietnamese-capable) -----------------
def _find_font_path():
  pygame.font.init()
  candidates = [
    "DejaVu Sans", "DejaVuSans",
    "Arial Unicode MS", "Arial",
    "Helvetica Neue", "Helvetica",
    "Noto Sans", "NotoSans",
    "Times New Roman", "Times",
    "Tahoma", "Liberation Sans", "Segoe UI"
  ]
  for name in candidates:
    path = pygame.font.match_font(name, bold=False, italic=False)
    if path:
      return path
  return None

FONT_PATH = _find_font_path()

def make_font(size: int) -> pygame.font.Font:
  # Prefer a known Unicode-capable font; fall back to default if not found
  if FONT_PATH:
    return pygame.font.Font(FONT_PATH, size)
  return pygame.font.SysFont(None, size)

# ----------------- Pygame UI helpers -----------------
def draw_text(surface, text, x, y, font, color=WHITE):
  surf = font.render(text, True, color)
  surface.blit(surf, (x, y))
  return surf.get_rect(topleft=(x, y))

def draw_wrapped_text(surface, text, rect, font, color=WHITE, line_spacing=6):
  words = text.split()
  lines, cur = [], ""
  for w in words:
    test = f"{cur} {w}".strip()
    if font.size(test)[0] <= rect.width:
      cur = test
    else:
      lines.append(cur)
      cur = w
  if cur:
    lines.append(cur)
  prev_clip = surface.get_clip()
  surface.set_clip(rect)
  y = rect.top
  for line in lines:
    if y + font.get_height() > rect.bottom:
      break
    surface.blit(font.render(line, True, color), (rect.left, y))
    y += font.get_height() + line_spacing
  surface.set_clip(prev_clip)

# --- Wrapped text helpers for scrolling panes ---

# Helper: split an overlong token (e.g., a URL) into pieces that fit within max_width.
def _split_long_token(token: str, font: pygame.font.Font, max_width: int) -> list[str]:
  """Split an overlong token (e.g., a URL) into pieces that fit within max_width."""
  if font.size(token)[0] <= max_width:
    return [token]
  parts, cur = [], ""
  for ch in token:
    test = cur + ch
    # ensure at least 1 char per chunk even if a single glyph exceeds max_width
    if cur and font.size(test)[0] > max_width:
      parts.append(cur)
      cur = ch
    else:
      cur = test
  if cur:
    parts.append(cur)
  return parts

def wrap_text_lines(text: str, font: pygame.font.Font, max_width: int) -> list[str]:
  words_raw = (text or "").split()
  # Expand words: break long tokens (like URLs) into viewport-fitting chunks.
  words = []
  for w in words_raw:
    words.extend(_split_long_token(w, font, max_width))

  lines, cur = [], ""
  for w in words:
    test = f"{cur} {w}".strip()
    if not cur:
      cur = w
    elif font.size(test)[0] <= max_width:
      cur = test
    else:
      lines.append(cur)
      cur = w
  if cur:
    lines.append(cur)
  return lines

def draw_wrapped_text_scrolled(surface, text, rect, font, scroll_px, color=WHITE, line_spacing=6):
  lines = wrap_text_lines(text, font, rect.width)
  line_h = font.get_height() + line_spacing
  total_h = max(0, len(lines) * line_h - line_spacing)
  start_line = 0
  y_offset = 0
  if line_h > 0:
    start_line = max(0, scroll_px // line_h)
    y_offset = -(scroll_px % line_h)
  prev_clip = surface.get_clip()
  surface.set_clip(rect)
  y = rect.top + y_offset
  for i in range(start_line, len(lines)):
    if y + font.get_height() > rect.bottom:
      break
    surface.blit(font.render(lines[i], True, color), (rect.left, y))
    y += line_h
  surface.set_clip(prev_clip)
  return total_h

def button(surface, rect, label, font, bg=BLUE, fg=WHITE):
  pygame.draw.rect(surface, bg, rect, border_radius=8)
  txt = font.render(label, True, fg)
  surface.blit(txt, (rect.centerx - txt.get_width()//2, rect.centery - txt.get_height()//2))

# ----------------- Camera via pygame.camera -----------------
def capture_with_pygame_camera(save_path: Path) -> bool:
  pygame.camera.init()
  cams = pygame.camera.list_cameras()
  if not cams:
    return False
  cam = pygame.camera.Camera(cams[0], (640, 480))
  cam.start()
  clock = pygame.time.Clock()

  screen = pygame.display.get_surface()
  font = make_font(24)
  capturing = True
  result = False

  while capturing:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        capturing = False
      elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_ESCAPE:
          capturing = False
        elif event.key in (pygame.K_SPACE, pygame.K_RETURN):
          img = cam.get_image()
          pygame.image.save(img, str(save_path))
          result = True
          capturing = False

    img = cam.get_image()
    screen.fill(BLACK)
    if img:
      # center the camera image
      scale = min(SCREEN_W / img.get_width(), SCREEN_H / img.get_height())
      new_size = (int(img.get_width()*scale), int(img.get_height()*scale))
      frame = pygame.transform.smoothscale(img, new_size)
      screen.blit(frame, (SCREEN_W//2 - frame.get_width()//2, SCREEN_H//2 - frame.get_height()//2))
    instr = "SPACE/ENTER: Capture   |   ESC: Cancel"
    screen.blit(font.render(instr, True, WHITE), (20, SCREEN_H - 40))
    pygame.display.flip()
    clock.tick(FPS)

  cam.stop()
  return result

# ImageNet mean and standard deviation values used for normalization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

 # Default output filename placeholder (will be set by UI)
output_filename = str(CAPTURE_PATH)  # will be set by UI; placeholder default

# --- OCR preprocessing helper ---
def preprocess_for_ocr(image_path: str, out_path: Path) -> str:
  """
  Heavier preprocessing for better OCR accuracy (CPU friendly, no OpenCV GUI calls):
  - Convert to grayscale
  - Auto-contrast
  - Median filter for salt-and-pepper noise (optional)
  - Unsharp mask
  - Upscale (x2) with LANCZOS
  - Otsu-like global threshold to binarize
  """
  try:
    img = Image.open(image_path).convert("L")  # grayscale
    # boost contrast
    img = ImageOps.autocontrast(img)
    # light denoise
    if OCR_ACCURACY_MODE:
      img = img.filter(ImageFilter.MedianFilter(size=3))
    # sharpen
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=3))
    # upscale for recognizers that benefit from bigger x-height
    if OCR_ACCURACY_MODE:
      new_size = (img.width * 2, img.height * 2)
      img = img.resize(new_size, Image.LANCZOS)
    # simple Otsu-like threshold (without numpy): compute histogram + best thresh
    hist = img.histogram()  # 256 bins
    total = sum(hist)
    sumB = wB = 0
    maximum = sum1 = 0
    for i, h in enumerate(hist):
      sum1 += i * h
    threshold = 127
    for t in range(256):
      wB += hist[t]
      if wB == 0:
        continue
      wF = total - wB
      if wF == 0:
        break
      sumB += t * hist[t]
      mB = sumB / wB
      mF = (sum1 - sumB) / wF
      between = wB * wF * (mB - mF) * (mB - mF)
      if between >= maximum:
        threshold = t
        maximum = between
    # apply threshold
    img = img.point(lambda p: 255 if p > threshold else 0, mode='1')
    img = img.convert("L")  # back to L for engines expecting 8-bit
    img.save(out_path)
    return str(out_path)
  except Exception:
    # If anything fails, just return the original
    return image_path

def select_device() -> torch.device:
    """
    Choose the best device per-OS:
    - macOS: force CPU and enable MPS fallback (more reliable for this model).
    - Windows/Linux: prefer CUDA if available; otherwise CPU.
    """
    system = platform.system().lower()
    if system == "darwin":
        # Ensure MPS ops fall back to CPU gracefully; we still run on CPU.
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = select_device()
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True  # optimize CUDA kernels for current input sizes

 # Use float16 on CUDA and Apple M‑series (MPS) for better performance; fallback to float32 on CPU
if DEVICE.type == "cuda":
    torch_dtype = torch.float16
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    torch_dtype = torch.float16
else:
    torch_dtype = torch.float32
print(f"[Info] Using device: {DEVICE.type} | dtype: {torch_dtype}")

def build_transform(input_size: int) -> T.Compose:
    """
    Build a torchvision transformation pipeline for resizing and normalizing
    images according to ImageNet statistics.
    """
    # Compose a series of transformations:
    # 1. Ensure the image is in RGB mode
    # 2. Resize to the desired size using bicubic interpolation
    # 3. Convert the image to a PyTorch tensor
    # 4. Normalize using predefined mean and std values
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio: float, target_ratios: list[tuple[int, int]],
                              width: int, height: int, image_size: int) -> tuple[int, int]:
    """
    Given an aspect ratio and a set of candidate ratios, find the ratio
    closest to the image's aspect ratio. Tie‑breakers favor ratios that
    produce larger cropped areas.
    """
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        # Compute difference between desired aspect ratio and candidate
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            # Prefer ratios that produce larger crops when there is a tie
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image: Image.Image, min_num: int = 1, max_num: int = 12,
                       image_size: int = 448, use_thumbnail: bool = False) -> list[Image.Image]:
    """
    Dynamically crops an image into blocks with sizes determined by the closest
    aspect ratios. Optionally appends a thumbnail to the returned list.

    Parameters:
    - image: the input PIL Image.
    - min_num, max_num: specify the minimum and maximum number of blocks allowed.
    - image_size: the target size of each crop.
    - use_thumbnail: if True and multiple blocks are generated, append a full-size thumbnail.
    Returns a list of cropped PIL Images.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Generate candidate ratios within the specified range. Each ratio (i, j)
    # corresponds to cropping the image into a grid of i columns and j rows.
    target_ratios = [(i, j)
                     for n in range(min_num, max_num + 1)
                     for i in range(1, n + 1)
                     for j in range(1, n + 1)
                     if min_num <= i * j <= max_num]
    # Sort candidate ratios by the total number of blocks (i * j)
    target_ratios = sorted(set(target_ratios), key=lambda x: x[0] * x[1])

    # Choose the closest aspect ratio based on the original image dimensions
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # Compute new dimensions for the resized image
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    # Total number of blocks equals rows * columns
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize the image to match the computed dimensions
    resized_img = image.resize((target_width, target_height))
    processed_images = []

    # Calculate how many blocks fit in each row of the resized image
    blocks_per_row = target_width // image_size
    for idx in range(blocks):
        # Determine the row and column index for this block
        row = idx // blocks_per_row
        col = idx % blocks_per_row
        # Compute the coordinates of the crop box
        left = col * image_size
        upper = row * image_size
        right = left + image_size
        lower = upper + image_size
        # Crop the image block and append it to the list
        split_img = resized_img.crop((left, upper, right, lower))
        processed_images.append(split_img)
    # Ensure the number of processed blocks matches expectation
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        # Append a thumbnail of the original image if requested
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file: str, input_size: int = 448, max_num: int = 12,
               device: torch.device = None, dtype: torch.dtype = None) -> torch.Tensor:
    """
    Load an image from disk, preprocess it into blocks, and stack the pixel
    tensors into a single tensor.

    Parameters:
    - image_file: path to the image file on disk.
    - input_size: dimension of the square crops.
    - max_num: maximum number of blocks allowed when cropping.
    - device: torch.device to move the tensor to.
    - dtype: desired torch dtype for the tensor.

    Returns a 4D tensor of shape (num_blocks, 3, input_size, input_size).
    """
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(im) for im in images]
    pixel_values = torch.stack(pixel_values)
    if dtype is not None:
        pixel_values = pixel_values.to(dtype)
    if device is not None:
        pixel_values = pixel_values.to(device)
    return pixel_values


# ----------------- OCR + ChatGPT Pipeline -----------------

USE_HF_VLM = os.getenv("USE_HF_VLM", "0") == "1"

# Choose the model name from Hugging Face
model_name = "5CD-AI/Vintern-1B-v3_5"

def run_ocr_pipeline(image_path: str) -> str:
  global output_filename
  output_filename = image_path
  response = ""
  # Reverted: Use the VLM-based text extraction as in the provided txt (no heavy preprocessing; no PaddleOCR/Tesseract)
  try:
    model = AutoModel.from_pretrained(
      model_name,
      dtype=torch_dtype,
      low_cpu_mem_usage=True,
      trust_remote_code=True,
      use_flash_attn=(DEVICE.type == "cuda"),
    ).eval().to(DEVICE)
    if hasattr(model, "set_attention_implementation"):
      model.set_attention_implementation("sdpa")
  except Exception:
    # Fallback without flash-attn flag or nonstandard kwargs
    model = AutoModel.from_pretrained(
      model_name,
      dtype=torch_dtype,
      low_cpu_mem_usage=True,
      trust_remote_code=True,
    ).eval().to(DEVICE)
    if hasattr(model, "set_attention_implementation"):
      model.set_attention_implementation("sdpa")

  tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

  # Use the image path provided by the UI directly (no preprocessing)
  test_image = image_path

  # Prepare inputs for the model (match txt settings)
  pixel_values = load_image(test_image, max_num=6, device=DEVICE, dtype=torch_dtype)

  # Configuration dictionary controlling text generation (match txt)
  generation_config = dict(
      max_new_tokens=512,   # maximum number of tokens to generate
      do_sample=False,      # disable sampling; use deterministic beam search
      num_beams=3,          # number of beams in beam search
      repetition_penalty=3.5
  )

  # Vietnamese prompt requesting only the words detected on the image (match txt)
  question = '<image>\\nTrích xuất và trả về các từ xuất hiện trên hình ảnh.'

  # Run inference via model.chat to generate the response
  response = model.chat(tokenizer, pixel_values, question, generation_config)
  # write OCR output
  ANSWER_PATH.write_text(response, encoding="utf-8")
  return response


openai.api_key = os.getenv("OPENAI_API_KEY", HARDCODED_OPENAI_KEY)

SYSTEM_PROMPT = """You are given a text. As an intelligent assistant, you must read the given text, identify all the keywords of the text and then give me 5 most reliable online materials that are related to the keyword you have identified. You must also provide me with links to each materials that you have found and summarize the content of each website. You should ensure that every links that you have provided is active and as an extra step of caution you should access the link you intend to provide and verify that the contents is there.
"""

def run_chatgpt_from_answer() -> str:
  if not openai.api_key or openai.api_key == "sk-PASTE_YOUR_KEY_HERE":
    warnings.warn("OPENAI_API_KEY not set (or placeholder used); skipping ChatCompletion block.")
    CHATGPT_PATH.write_text("", encoding="utf-8")
    return ""
  user_question = ANSWER_PATH.read_text(encoding="utf-8").strip() if ANSWER_PATH.exists() else ""
  if not user_question:
    CHATGPT_PATH.write_text("", encoding="utf-8")
    return ""
  messages = [{"role": "system", "content": SYSTEM_PROMPT},
              {"role": "user", "content": user_question}]
  try:
    chat = openai.ChatCompletion.create(model="gpt-4", messages=messages)
    reply = chat.choices[0].message["content"]
  except Exception:
    reply = ""
  CHATGPT_PATH.write_text("The answer: " + reply, encoding="utf-8")
  return reply

def perform_pipeline(image_path: str) -> tuple[str, str]:
  ocr_text = run_ocr_pipeline(image_path)
  gpt_text = run_chatgpt_from_answer()
  return ocr_text, gpt_text

# ----------------- Pygame App -----------------

def file_picker(start_dir: Path) -> str | None:
  cur = start_dir if start_dir.exists() else Path.home()
  entries = []
  idx = 0
  clock = pygame.time.Clock()
  screen = pygame.display.get_surface()
  font = make_font(28)
  small = make_font(22)

  def refresh():
    nonlocal entries, idx
    items = []
    # parent dir
    if cur.parent != cur:
      items.append(("..", True))
    for name in sorted(os.listdir(cur)):
      p = cur / name
      if p.is_dir():
        items.append((name + "/", True))
      else:
        items.append((name, False))
    entries = items
    idx = 0

  refresh()
  running = True
  while running:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        return None
      elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_ESCAPE:
          return None
        elif event.key in (pygame.K_DOWN, pygame.K_j):
          idx = min(idx + 1, len(entries) - 1)
        elif event.key in (pygame.K_UP, pygame.K_k):
          idx = max(idx - 1, 0)
        elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
          name, is_dir = entries[idx]
          if name == "..":
            cur = cur.parent
            refresh()
          else:
            path = cur / name
            if is_dir:
              cur = path
              refresh()
            else:
              if path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"]:
                return str(path)
      elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
        mx, my = event.pos
        # simple click selection
        start_y = 100
        row_h = 28
        rel = (my - start_y) // row_h
        if 0 <= rel < len(entries):
          idx = int(rel)

    screen.fill(GRAY)
    draw_text(screen, f"Choose a file (Start: {cur})", 20, 20, font)
    start_y = 100
    row_h = 28
    for i, (name, is_dir) in enumerate(entries[:min(16, len(entries))]):
      y = start_y + i * row_h
      color = BLUE if i == idx else WHITE
      icon = "[D]" if is_dir or name == ".." else "   "
      draw_text(screen, f"{icon} {name}", 24, y, small, color=color)
    draw_text(screen, "↑/k, ↓/j to move  •  ENTER/SPACE to open/select  •  ESC to cancel", 20, SCREEN_H - 40, small)
    pygame.display.flip()
    clock.tick(FPS)
  return None

def pick_file_native_or_pygame(start_dir: Path) -> str | None:
  """
  Prefer the native OS file picker (Finder on macOS, File Explorer on Windows)
  via a short subprocess that runs Tk, to avoid SDL/Tk conflicts on macOS.
  Falls back to the in‑Pygame file picker if anything fails.
  """
  try:
    import subprocess, sys, os
    script = r"""
import sys, os
try:
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.update_idletasks()
    initialdir = sys.argv[1] if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]) else os.path.expanduser("~")
    filetypes = [("Image files","*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff"), ("All files","*.*")]
    path = filedialog.askopenfilename(title="Select an image", initialdir=initialdir, filetypes=filetypes)
    print(path or "")
except Exception:
    print("")
"""
    result = subprocess.run([sys.executable, "-c", script, str(start_dir)],
                            capture_output=True, text=True)
    sel = (result.stdout or "").strip()
    if sel:
      return sel
  except Exception:
    # Any issue -> fall back to the Pygame-based picker
    pass
  return file_picker(start_dir)

def run_pygame_ui():
  pygame.init()
  pygame.display.set_caption("OCR + ChatGPT – Pygame UI")
  screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
  clock = pygame.time.Clock()
  title_font = make_font(40)  # slightly smaller to avoid overlap with pane headers
  small = make_font(28)

  state = "menu"  # menu | capture | choose | processing | results
  ocr_text, gpt_text = "", ""
  img_path = None

  # scrolling state for results panes
  scroll_left = 0
  scroll_right = 0
  active_pane = "left"  # which pane responds to arrow keys
  SCROLL_STEP = 40

  def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

  while True:
    events = pygame.event.get()
    for e in events:
      if e.type == pygame.QUIT:
        pygame.quit()
        return

    screen.fill(GRAY)

    if state == "menu":
      draw_text(screen, "Choose an option", 30, 40, title_font)
      take_rect = pygame.Rect(SCREEN_W//2 - 160, 180, 320, 70)
      pick_rect = pygame.Rect(SCREEN_W//2 - 160, 280, 320, 70)
      button(screen, take_rect, "1) Take a photo", small)
      button(screen, pick_rect, "2) Choose a file", small)
      # handle events captured this frame
      for e in events:
        if e.type == pygame.KEYDOWN:
          if e.key == pygame.K_1:
            state = "capture"
          elif e.key == pygame.K_2:
            state = "choose"
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
          if take_rect.collidepoint(e.pos):
            state = "capture"
          elif pick_rect.collidepoint(e.pos):
            state = "choose"
      pygame.display.flip()
      clock.tick(FPS)
      continue

    if state == "capture":
      ok = capture_with_pygame_camera(CAPTURE_PATH)
      if ok:
        img_path = str(CAPTURE_PATH)
        state = "processing"
      else:
        state = "menu"
      continue

    if state == "choose":
      picked = pick_file_native_or_pygame(DESKTOP_DIR)
      if picked:
        img_path = picked
        state = "processing"
      else:
        state = "menu"
      continue

    if state == "processing":
      draw_text(screen, "Processing... please wait", 30, 40, title_font)
      pygame.display.flip()
      ocr_text, gpt_text = perform_pipeline(img_path)
      state = "results"
      continue

    if state == "results":
      draw_text(screen, "Results", 30, 20, title_font)
      sub = make_font(26)
      left_rect = pygame.Rect(30, 80, SCREEN_W//2 - 50, SCREEN_H - 180)
      right_rect = pygame.Rect(SCREEN_W//2 + 20, 80, SCREEN_W//2 - 50, SCREEN_H - 180)
      pygame.draw.rect(screen, (60,60,60), left_rect)
      pygame.draw.rect(screen, (60,60,60), right_rect)

      draw_text(screen, "answer.txt (OCR)", left_rect.left, left_rect.top - 30, sub)
      draw_text(screen, "chatgpt_answer.txt", right_rect.left, right_rect.top - 30, sub)
      left_inner = left_rect.inflate(-16, -16)
      right_inner = right_rect.inflate(-16, -16)
      total_left = draw_wrapped_text_scrolled(screen, ocr_text or "(empty)", left_inner, sub, scroll_left)
      right_text = Path(CHATGPT_PATH).read_text(encoding="utf-8") if CHATGPT_PATH.exists() else "(empty)"
      total_right = draw_wrapped_text_scrolled(screen, right_text, right_inner, sub, scroll_right)

      back_rect = pygame.Rect(SCREEN_W//2 - 220, SCREEN_H - 80, 200, 50)
      exit_rect = pygame.Rect(SCREEN_W//2 + 20, SCREEN_H - 80, 200, 50)
      button(screen, back_rect, "Back to menu", small)
      button(screen, exit_rect, "Exit", small)
      # handle events captured this frame
      for e in events:
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
          if back_rect.collidepoint(e.pos):
            state = "menu"
          elif exit_rect.collidepoint(e.pos):
            pygame.quit(); return
          # set active pane focus on click
          elif left_inner.collidepoint(e.pos):
            active_pane = "left"
          elif right_inner.collidepoint(e.pos):
            active_pane = "right"
        elif e.type == pygame.MOUSEWHEEL:
          mx, my = pygame.mouse.get_pos()
          if left_inner.collidepoint(mx, my):
            max_left = max(0, total_left - left_inner.height)
            scroll_left = clamp(scroll_left - e.y * SCROLL_STEP, 0, max_left)
          elif right_inner.collidepoint(mx, my):
            max_right = max(0, total_right - right_inner.height)
            scroll_right = clamp(scroll_right - e.y * SCROLL_STEP, 0, max_right)
          else:
            # scroll active pane if mouse is outside both panes
            if active_pane == "left":
              max_left = max(0, total_left - left_inner.height)
              scroll_left = clamp(scroll_left - e.y * SCROLL_STEP, 0, max_left)
            else:
              max_right = max(0, total_right - right_inner.height)
              scroll_right = clamp(scroll_right - e.y * SCROLL_STEP, 0, max_right)
        elif e.type == pygame.KEYDOWN:
          if e.key == pygame.K_ESCAPE:
            state = "menu"
          elif e.key in (pygame.K_q, pygame.K_x):
            pygame.quit(); return
          elif e.key in (pygame.K_UP, pygame.K_DOWN, pygame.K_PAGEUP, pygame.K_PAGEDOWN, pygame.K_HOME, pygame.K_END):
            # arrow/page/home/end keys control the active pane
            if active_pane == "left":
              max_left = max(0, total_left - left_inner.height)
              if e.key == pygame.K_UP:
                scroll_left = clamp(scroll_left - SCROLL_STEP, 0, max_left)
              elif e.key == pygame.K_DOWN:
                scroll_left = clamp(scroll_left + SCROLL_STEP, 0, max_left)
              elif e.key == pygame.K_PAGEUP:
                scroll_left = clamp(scroll_left - left_inner.height, 0, max_left)
              elif e.key == pygame.K_PAGEDOWN:
                scroll_left = clamp(scroll_left + left_inner.height, 0, max_left)
              elif e.key == pygame.K_HOME:
                scroll_left = 0
              elif e.key == pygame.K_END:
                scroll_left = max_left
            else:
              max_right = max(0, total_right - right_inner.height)
              if e.key == pygame.K_UP:
                scroll_right = clamp(scroll_right - SCROLL_STEP, 0, max_right)
              elif e.key == pygame.K_DOWN:
                scroll_right = clamp(scroll_right + SCROLL_STEP, 0, max_right)
              elif e.key == pygame.K_PAGEUP:
                scroll_right = clamp(scroll_right - right_inner.height, 0, max_right)
              elif e.key == pygame.K_PAGEDOWN:
                scroll_right = clamp(scroll_right + right_inner.height, 0, max_right)
              elif e.key == pygame.K_HOME:
                scroll_right = 0
              elif e.key == pygame.K_END:
                scroll_right = max_right
      pygame.display.flip()
      clock.tick(FPS)
      continue

if __name__ == "__main__":
  run_pygame_ui()