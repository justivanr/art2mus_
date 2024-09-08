import scipy
import torch
import sys

sys.path.append("src")
import conf
PROJ_DIR = conf.PROJ_DIR

from art2mus.art2mus_pipeline import AudioLDM2Pipeline
import art2mus.utils.train_test_utils as tu


def generate_audio(prompt=None, img_path=None, img_emb=None, neg_prompt="Low quality.",
                   inf_steps=200, aud_len=10.0, waveforms=3,
                   output_dir=PROJ_DIR, file_name="techno.wav"):
    
    print(f"Generating music based on the provided artwork....\n======================")
    
    audio = pipe(
        prompt=prompt,
        image_path=img_path,
        image_embeds=img_emb,
        negative_prompt=neg_prompt,
        num_inference_steps=inf_steps,
        audio_length_in_s=aud_len,
        num_waveforms_per_prompt=waveforms,
        generator=generator,
    ).audios
        
    scipy.io.wavfile.write(output_dir + file_name, rate=16000, data=audio[0])
    
    print(f"Music file stored at: {output_dir + file_name}")


# Output folder
OUT_DIR = tu.PROJ_DIR
# Model repo id on HuggingFace
MODEL_REPO_ID = tu.AUDIOLDM2_REPO_ID
# Seed
SEED = 0
# Path of an example artwork to use to generate music
EXAMPLE_ARTWORK_PATH = 'test_images/erin-hanson_thistles-on-orange-2016.jpg'
# Example negative prompt
NEG_PROMPT = "Low quality."


if torch.cuda.is_available():
    tot_gpu_mem = round(torch.cuda.mem_get_info()[1] / 1024 ** 3, 2)
    free_gpu_mem = round(torch.cuda.mem_get_info()[0] / 1024 ** 3, 2)
    print(f"Free memory/Total memory: {free_gpu_mem}/{tot_gpu_mem} GIB")
    
    # At least 2.8GiB are needed to run Art2Mus on GPU
    if free_gpu_mem >= 3.0:
        print("Using CUDA!")
        device = "cuda"
    else:
        print("Not enough free GPU memory! Using CPU.")
        device = 'cpu'
else:
    print("CUDA not found! Using CPU.")
    device = 'cpu'

if device == 'cuda':
    print("Loading model with torch.float16, needed to work with GPU.")
    pipe = AudioLDM2Pipeline.from_pretrained(pretrained_model_name_or_path=MODEL_REPO_ID, torch_dtype=torch.float16, 
                                             custom_pipeline=PROJ_DIR+"/src/art2mus/art2mus_pipeline.py")
else:
    print("Loading model without torch.float16, needed to work with CPU.")
    pipe = AudioLDM2Pipeline.from_pretrained(pretrained_model_name_or_path=MODEL_REPO_ID, 
                                             custom_pipeline=PROJ_DIR+"/src/art2mus/art2mus_pipeline.py")
    
pipe = pipe.to(device)
print(f"Pipeline moved to: {device}!")

generator = torch.Generator(device).manual_seed(SEED)

generate_audio(prompt=None, img_path=EXAMPLE_ARTWORK_PATH, img_emb=None, neg_prompt=NEG_PROMPT,
               inf_steps=200, aud_len=10.0, waveforms=3, output_dir=OUT_DIR, file_name=f"/test_music/art2mus_example.wav")