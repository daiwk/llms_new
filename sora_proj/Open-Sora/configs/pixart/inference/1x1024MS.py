num_frames = 1
fps = 1
image_size = (1920, 512)
multi_resolution = True

# Define model
model = dict(
    type="PixArtMS-XL/2",
    space_scale=2.0,
    time_scale=1.0,
    no_temporal_pos_emb=True,
    from_pretrained="PixArt-XL-2-1024-MS.pth",
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="stabilityai/sd-vae-ft-ema",
)
text_encoder = dict(
    type="t5",
    from_pretrained="./pretrained_models/t5_ckpts",
    model_max_length=120,
)
scheduler = dict(
    type="dpm-solver",
    num_sampling_steps=20,
    cfg_scale=7.0,
)
dtype = "fp16"

# Others
batch_size = 2
seed = 42
prompt_path = "./assets/texts/t2i_samples.txt"
save_dir = "./outputs/samples/"
