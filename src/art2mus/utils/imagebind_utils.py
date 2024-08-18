"""Script containing methods concerning the ImageBind model.

This script provides methods for:
- Calculating memory size required to store model parameters.
- Checking whether the model should be loaded on CPU or GPU based on memory availability.
- Loading a pre-trained ImageBind model and setting it to evaluation mode.
- Loading the model onto GPU if available, otherwise on CPU.
- Generating embeddings using the provided model for either vision or audio data.
"""

import sys
sys.path.append("src")
import conf

# Directory in which the project is stored
PROJ_DIR = conf.PROJ_DIR
sys.path.append(PROJ_DIR + "/src/ImageBind")

import torch
import numpy as np
from ImageBind.imagebind import data
from ImageBind.imagebind.models import imagebind_model
from ImageBind.imagebind.models.imagebind_model import ModalityType
from ImageBind.imagebind.models.multimodal_preprocessors import SimpleTokenizer
from sklearn.metrics.pairwise import cosine_similarity

BPE_PATH = PROJ_DIR + "/src/ImageBind/bpe/bpe_simple_vocab_16e6.txt.gz"


def cal_size(num_params, dtype):
    """
    Calculates the memory size required to store the parameters of a model based on the data type.

    Params:
        - num_params (int): Number of parameters in the model.
        - dtype (str): Data type of the parameters. Supported types: "float32", "float16", "int8".

    Returns:
        float: Memory size required in GB.
    """
    if dtype == "float32":
        size_mb = (num_params / 1024**2) * 4
    elif dtype == "float16":
        size_mb = (num_params / 1024**2) * 2
    elif dtype == "int8":
        size_mb = (num_params / 1024**2) * 1
    else:
        return -1

    # Convert MB to GB
    return size_mb / 1024


def check_cpu_or_gpu(model, full_log=True):
    """
    Checks whether the model should be loaded on CPU or GPU based on memory availability.

    Params:
        - model: Model object.
        - full_log (bool): Whether to print full log or not, default is True.

    Returns:
        str: "cuda" if GPU is available and has sufficient memory, otherwise "cpu".
    """    
    if torch.cuda.is_available():
        tot_gpu_mem = torch.cuda.mem_get_info()[1] / 1024 ** 3
        free_gpu_mem = torch.cuda.mem_get_info()[0] / 1024 ** 3

        total_params = sum(p.numel() for p in model.parameters())
        model_size = round(cal_size(total_params, "float32"), 2)
        if full_log:
            print(f"Free memory/Total memory: {free_gpu_mem}/{tot_gpu_mem}\n"
                f"Model size: {model_size}\n=================")
            
        if model_size > free_gpu_mem:
            if full_log:
                print("Not enough VRAM available to load the model!\n=================")
            return 'cpu'
        else:
            return "cuda"
    else:
        if full_log:
            print("Not GPU found to load the model!\n=================")
        return 'cpu'


def load_model(full_log=True, use_cpu=True):
    """
    Loads a pre-trained ImageBind model and sets it to evaluation mode.
    
    Params:
        - full_log (bool): Whether to print full log or not, default is True.

    Returns:
        torch.nn.Module: Loaded pre-trained model.
    """
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    if full_log:
        print("ImageBind has been loaded and is ready to be used!")
    if use_cpu:
        model.to('cpu')
    else:
       gpu_if_available(model, full_log) 
    return model
  

def gpu_if_available(model, full_log=True):
    """
    Loads the model onto GPU if available, otherwise on CPU.

    Params:
        - model: Model object to be loaded onto GPU/CPU.
        - full_log (bool): Whether to print full log or not, default is True.

    Returns:
        None
    """
    device = check_cpu_or_gpu(model, full_log)
    if full_log:
        print(f"Loading model on: {device}...\n=================")
    model.to(device)
    if full_log:
        print(f"ImageBind moved to {device}!")  


# We redefine this to adapt the BPE_PATH based on its position in our repo
def my_load_and_transform_text(text, device):
    if text is None:
        return None
    tokenizer = SimpleTokenizer(bpe_path=BPE_PATH)
    tokens = [tokenizer(t).unsqueeze(0).to(device) for t in text]
    tokens = torch.cat(tokens, dim=0)
    return tokens


def generate_embeds(model, curr_dev='cpu', 
                    image_paths=None, audio_paths=None, text=None,
                    extract_emb=True, emb_type=None):
    """
    Generates embeddings using the provided model for either vision or audio data.

    Parameters:
        - model: Model object to generate embeddings.
        - curr_dev (str): Device to use for inference, default is "cpu".
        - image_paths (list): List of paths to image files.
        - audio_paths (list): List of paths to audio files.
        - extract_emb (bool): Whether to extract embeddings or not, default is True.
        - emb_type (str): Type of embeddings to generate, either 'vision' or 'audio'.

    Returns:
        list or torch.Tensor: Generated embeddings.
    """    
    if emb_type == 'vision':
        inputs = {
            ModalityType.VISION: data.load_and_transform_vision_data(image_paths, curr_dev),
        }
    elif emb_type == 'audio':
        inputs = {
            ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, curr_dev),
        }
    elif emb_type == 'text':
        inputs = {
            ModalityType.TEXT: my_load_and_transform_text(text, curr_dev),
        }
    
    with torch.no_grad():
        embeddings = model(inputs)

    if extract_emb:
        embeddings = [entry for entry in embeddings[emb_type]]

    return embeddings

def compute_similarity(embeds_1, embeds_2):
    """
    Computes the cosine similarity between two embeddings.

    Params:
    - embeds_1 (numpy.ndarray): First embeddings.
    - embeds_2 (numpy.ndarray): Second embeddings.

    Returns:
    - float: Cosine similarity between the image and audio embeddings.
    """    
    # Reshape embeddings before computing similarity
    if embeds_1.ndim == 1:
        embeds_1 = embeds_1.reshape(1, -1)
        
    if embeds_2.ndim == 1:
        embeds_2 = embeds_2.reshape(1, -1)
        
    # Move embeddings to cpu if needed
    if embeds_1.is_cuda:
             embeds_1 = embeds_1.cpu()
             
    if not isinstance(embeds_2, np.ndarray) and embeds_2.is_cuda:
             embeds_2 = embeds_2.cpu()
             
    embeds_1 = embeds_1.detach().numpy()
             
    # Compute similiarty among embeddings
    similarity = cosine_similarity(embeds_1, embeds_2)[0][0]
    
    return similarity