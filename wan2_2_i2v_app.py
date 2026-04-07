import os, random, time, shutil, uuid, re

import torch
import numpy as np
from PIL import Image
import torchvision.io as tvio
from nodes import NODE_CLASS_MAPPINGS

from comfy_extras.nodes_wan import WanImageToVideo
from comfy_extras.nodes_video import CreateVideo
from comfy_extras.nodes_model_advanced import ModelSamplingSD3

# ── Model Loading ──────────────────────────────────────────────
print("\n" + "=" * 50)
print("  Wan2.2 I2V Starting Up")
print("=" * 50)

UNETLoader            = NODE_CLASS_MAPPINGS["UNETLoader"]()
CLIPLoader            = NODE_CLASS_MAPPINGS["CLIPLoader"]()
VAELoader             = NODE_CLASS_MAPPINGS["VAELoader"]()
CLIPTextEncode        = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
KSamplerAdvanced      = NODE_CLASS_MAPPINGS["KSamplerAdvanced"]()
VAEDecode             = NODE_CLASS_MAPPINGS["VAEDecode"]()
WanImageToVideo       = WanImageToVideo
LoraLoaderModelOnly   = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
ModelSamplingSD3      = ModelSamplingSD3
CreateVideo           = CreateVideo

# ── Default model filenames (edit if your filenames differ) ────
UNET_HIGH_NOISE  = "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
UNET_LOW_NOISE   = "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
LORA_HIGH_NOISE  = "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"
LORA_LOW_NOISE   = "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"
CLIP_NAME        = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
VAE_NAME         = "wan_2.1_vae.safetensors"

startup_start = time.time()

with torch.inference_mode():
    print("\n[1/6] Loading CLIP (umt5_xxl)... ", end="", flush=True)
    t0 = time.time()
    clip = CLIPLoader.load_clip(CLIP_NAME, type="wan")[0]
    print(f"done ({time.time()-t0:.1f}s)")

    print("[2/6] Loading VAE... ", end="", flush=True)
    t0 = time.time()
    vae = VAELoader.load_vae(VAE_NAME)[0]
    print(f"done ({time.time()-t0:.1f}s)")

    print("[3/6] Loading UNet high-noise... ", end="", flush=True)
    t0 = time.time()
    unet_high = UNETLoader.load_unet(UNET_HIGH_NOISE, "default")[0]
    print(f"done ({time.time()-t0:.1f}s)")

    print("[4/6] Loading UNet low-noise... ", end="", flush=True)
    t0 = time.time()
    unet_low = UNETLoader.load_unet(UNET_LOW_NOISE, "default")[0]
    print(f"done ({time.time()-t0:.1f}s)")

    print("[5/6] Loading LoRA high-noise... ", end="", flush=True)
    t0 = time.time()
    unet_high_lora = LoraLoaderModelOnly.load_lora_model_only(unet_high, LORA_HIGH_NOISE, 1.0)[0]
    print(f"done ({time.time()-t0:.1f}s)")

    print("[6/6] Loading LoRA low-noise... ", end="", flush=True)
    t0 = time.time()
    unet_low_lora = LoraLoaderModelOnly.load_lora_model_only(unet_low, LORA_LOW_NOISE, 1.0)[0]
    print(f"done ({time.time()-t0:.1f}s)")

print(f"\n✅ All models loaded in {time.time()-startup_start:.1f}s")
print("=" * 50 + "\n")

# ── Helpers ────────────────────────────────────────────────────
save_dir = "./results"
os.makedirs(save_dir, exist_ok=True)

def get_save_path(prompt, ext="mp4"):
    safe = re.sub(r'[^a-zA-Z0-9_-]', '_', prompt)[:25]
    uid  = uuid.uuid4().hex[:6]
    return os.path.join(save_dir, f"{safe}_{uid}.{ext}")

def load_image_tensor(pil_img: Image.Image):
    """Convert PIL image to ComfyUI IMAGE tensor [1, H, W, 3] float32 0-1."""
    arr = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)   # [1, H, W, 3]

# ── Generation ─────────────────────────────────────────────────
@torch.inference_mode()
def generate(input):
    v = input["input"]
    start_image_pil  = v["start_image"]          # PIL Image
    positive_prompt  = v["positive_prompt"]
    negative_prompt  = v["negative_prompt"]
    width            = int(v["width"])
    height           = int(v["height"])
    length           = int(v["length"])           # number of frames
    seed             = int(v["seed"])
    sampler_name     = v["sampler_name"]
    # Normal-mode params
    steps_normal     = int(v["steps_normal"])
    split_step_normal= int(v["split_step_normal"])
    cfg_normal       = float(v["cfg_normal"])
    # Turbo-mode params (4-step LoRA)
    enable_turbo     = bool(v["enable_turbo"])
    steps_turbo      = int(v["steps_turbo"])
    split_step_turbo = int(v["split_step_turbo"])
    cfg_turbo        = float(v["cfg_turbo"])
    fps              = int(v["fps"])

    if seed == 0:
        seed = random.randint(1, 2**31 - 1)

    print("\n" + "=" * 50)
    print("  New Wan2.2 I2V Generation Request")
    print("=" * 50)
    total_start = time.time()

    # ── Pick params based on turbo mode switch ─────────────────
    if enable_turbo:
        unet_high_use   = unet_high_lora
        unet_low_use    = unet_low_lora
        steps           = steps_turbo
        split_step      = split_step_turbo
        cfg             = cfg_turbo
    else:
        unet_high_use   = unet_high
        unet_low_use    = unet_low
        steps           = steps_normal
        split_step      = split_step_normal
        cfg             = cfg_normal

    # ── [1] Encode prompts ─────────────────────────────────────
    print("\n[1/6] Encoding prompts... ", end="", flush=True)
    t0 = time.time()
    positive_cond = CLIPTextEncode.encode(clip, positive_prompt)[0]
    negative_cond = CLIPTextEncode.encode(clip, negative_prompt)[0]
    print(f"done ({time.time()-t0:.1f}s)")

    # ── [2] WanImageToVideo (build latent from start image) ────
    print("[2/6] Preparing image-to-video latent... ", end="", flush=True)
    t0 = time.time()
    start_image_tensor = load_image_tensor(start_image_pil)
    wan_out = WanImageToVideo.execute(
        positive=positive_cond,
        negative=negative_cond,
        vae=vae,
        width=width,
        height=height,
        length=length,
        batch_size=1,
        start_image=start_image_tensor,
    )
    positive_cond_i2v = wan_out[0]
    negative_cond_i2v = wan_out[1]
    latent             = wan_out[2]
    print(f"done ({time.time()-t0:.1f}s)")

    # ── [3] ModelSamplingSD3 on high-noise model ───────────────
    print("[3/6] Applying ModelSamplingSD3 on high-noise unet... ", end="", flush=True)
    t0 = time.time()
    shift = 5.0
    sampler = ModelSamplingSD3()
    unet_high_sd3 = sampler.patch(unet_high_use, shift)[0]
    print(f"done ({time.time()-t0:.1f}s)")

    # ── [4] KSamplerAdvanced — high-noise pass ─────────────────
    print(f"[4/6] Sampling high-noise ({steps} steps, split at {split_step})...")
    t0 = time.time()
    latent_mid = KSamplerAdvanced.sample(
        model=unet_high_sd3,
        add_noise="enable",
        noise_seed=seed,
        steps=steps,
        cfg=cfg,
        sampler_name=sampler_name,
        scheduler="simple",
        positive=positive_cond_i2v,
        negative=negative_cond_i2v,
        latent_image=latent,
        start_at_step=0,
        end_at_step=split_step,
        return_with_leftover_noise="enable",
    )[0]
    print(f"      High-noise pass done ({time.time()-t0:.1f}s)")

    # ── [5] KSamplerAdvanced — low-noise pass ─────────────────
    print(f"[5/6] Sampling low-noise (step {split_step} → {steps})...")
    t0 = time.time()
    latent_final = KSamplerAdvanced.sample(
        model=unet_low_use,
        add_noise="disable",
        noise_seed=0,
        steps=steps,
        cfg=cfg,
        sampler_name=sampler_name,
        scheduler="simple",
        positive=positive_cond_i2v,
        negative=negative_cond_i2v,
        latent_image=latent_mid,
        start_at_step=split_step,
        end_at_step=steps,
        return_with_leftover_noise="disable",
    )[0]
    print(f"      Low-noise pass done ({time.time()-t0:.1f}s)")

    # ── [6] VAEDecode → CreateVideo → save ────────────────────
    print("[6/6] VAE decode + create video... ", end="", flush=True)
    t0 = time.time()
    decoded_images = VAEDecode.decode(vae, latent_final)[0]   # [F, H, W, 3]

    # Save as mp4 via torchvision
    frames_uint8 = (decoded_images.detach().cpu().clamp(0, 1) * 255).to(torch.uint8)
    # torchvision expects [T, H, W, C]
    save_path = get_save_path(positive_prompt, "mp4")
    tvio.write_video(save_path, frames_uint8, fps=fps, video_codec="libx264")
    print(f"done ({time.time()-t0:.1f}s)")

    print(f"\n💾 Saved to : {save_path}")

    drive_path = "/content/gdrive/MyDrive/wan2_2_i2v"
    if os.path.exists(drive_path):
        shutil.copy(save_path, drive_path)
        print(f"☁️  Copied to Google Drive: {drive_path}")

    print(f"✅ Total    : {time.time()-total_start:.1f}s")
    print(f"✅ Total    : {time.time()-total_start:.1f}s")
    print("=" * 50 + "\n")

    return save_path, seed


# ── Gradio UI ──────────────────────────────────────────────────
import gradio as gr

DEFAULT_POSITIVE = (
    "The white dragon warrior stands still, eyes full of determination and strength. "
    "The camera slowly moves closer or circles around the warrior, highlighting the "
    "powerful presence and heroic spirit of the character."
)

DEFAULT_NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)

def generate_ui(
    start_image,
    positive_prompt,
    negative_prompt,
    width,
    height,
    length,
    seed,
    sampler_name,
    enable_turbo,
    # Normal mode
    steps_normal,
    split_step_normal,
    cfg_normal,
    # Turbo mode
    steps_turbo,
    split_step_turbo,
    cfg_turbo,
    fps,
):
    if start_image is None:
        raise gr.Error("Please upload a start image.")

    input_data = {
        "input": {
            "start_image":       start_image,         # PIL Image from gr.Image
            "positive_prompt":   positive_prompt,
            "negative_prompt":   negative_prompt,
            "width":             int(width),
            "height":            int(height),
            "length":            int(length),
            "seed":              int(seed),
            "sampler_name":      sampler_name,
            "enable_turbo":      enable_turbo,
            "steps_normal":      int(steps_normal),
            "split_step_normal": int(split_step_normal),
            "cfg_normal":        float(cfg_normal),
            "steps_turbo":       int(steps_turbo),
            "split_step_turbo":  int(split_step_turbo),
            "cfg_turbo":         float(cfg_turbo),
            "fps":               int(fps),
        }
    }

    video_path, used_seed = generate(input_data)
    return video_path, video_path, str(used_seed)


with gr.Blocks() as demo:
    gr.HTML("""
<div style="width:100%; display:flex; flex-direction:column; align-items:center; justify-content:center; margin:20px 0;">
    <h1 style="font-size:2.5em; margin-bottom:10px;">Wan2.2 Image-to-Video</h1>
</div>
""")

    with gr.Row():
        # ── Left column: inputs ───────────────────────────────
        with gr.Column():
            start_image = gr.Image(
                label="Start Image",
                type="pil",
                height=300,
            )
            positive = gr.Textbox(DEFAULT_POSITIVE, label="Positive Prompt", lines=4)

            with gr.Row():
                width  = gr.Number(value=640,  label="Width",          precision=0)
                height = gr.Number(value=640,  label="Height",         precision=0)
                length = gr.Slider(9, 201, value=81, step=4, label="Frames (length)")
                fps    = gr.Slider(8, 30,  value=16, step=1, label="FPS")

            with gr.Row():
                seed         = gr.Number(value=0, label="Seed (0 = random)", precision=0)
                sampler_name = gr.Dropdown(
                    choices=["euler", "euler_ancestral", "dpm_2", "dpm_2_ancestral",
                             "heun", "lms", "dpm_fast", "dpm_adaptive"],
                    value="euler",
                    label="Sampler",
                )

            with gr.Row():
                run = gr.Button("🚀 Generate", variant="primary")

            with gr.Accordion("⚡ Turbo Mode (4-step LoRA)", open=True):
                enable_turbo = gr.Checkbox(
                    value=True,
                    label="Enable Turbo Mode (faster, ~4 steps)",
                )
                with gr.Row():
                    steps_turbo       = gr.Slider(1, 10,  value=4,  step=1,   label="Steps (turbo)")
                    split_step_turbo  = gr.Slider(1, 9,   value=2,  step=1,   label="Split Step (turbo)")
                    cfg_turbo         = gr.Slider(0.5, 5, value=1.0, step=0.1, label="CFG (turbo)")

            with gr.Accordion("🎛️ Normal Mode Settings", open=False):
                with gr.Row():
                    steps_normal      = gr.Slider(5, 50,  value=20, step=1,   label="Steps (normal)")
                    split_step_normal = gr.Slider(1, 49,  value=10, step=1,   label="Split Step (normal)")
                    cfg_normal        = gr.Slider(0.5, 9, value=3.5, step=0.1, label="CFG (normal)")

            with gr.Accordion("Negative Prompt", open=False):
                negative = gr.Textbox(DEFAULT_NEGATIVE, label="Negative Prompt", lines=4)

        # ── Right column: outputs ─────────────────────────────
        with gr.Column():
            download_video = gr.File(label="Download Video")
            output_video   = gr.Video(label="Generated Video", height=480)
            used_seed      = gr.Textbox(label="Seed Used", interactive=False)

    run.click(
        fn=generate_ui,
        inputs=[
            start_image, positive, negative,
            width, height, length, seed, sampler_name,
            enable_turbo,
            steps_normal, split_step_normal, cfg_normal,
            steps_turbo,  split_step_turbo,  cfg_turbo,
            fps,
        ],
        outputs=[download_video, output_video, used_seed],
    )

demo.launch(theme=gr.themes.Monochrome(), share=True, debug=True)
