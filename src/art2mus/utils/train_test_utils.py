"""
Script that contains variables and methods that are used during the model's training.
"""

import os 
import sys
sys.path.append("src")
import conf

# Directory in which the project is stored
PROJ_DIR = conf.PROJ_DIR
sys.path.append(PROJ_DIR + "/src/audioldm")

import scipy
import torch
import shutil
import librosa
import scipy as sp
import numpy as np
from dataclasses import dataclass, asdict
from audioldm.stft import TacotronSTFT
from audioldm.utils import default_audioldm_config
from frechet_audio_distance import FrechetAudioDistance
from art2mus.utils.frechet_audio_distance.frechet_audio_distance import FrechetAudioDistance as MyFrechetAudioDistance
from art2mus.utils.imagebind_utils import load_model, generate_embeds, compute_similarity


# Repository containg the model
REPO_ID = "cvssp/audioldm2"
REPO_MUSIC_ID = "cvssp/audioldm2-music"

CUSTOM_PIPE = PROJ_DIR + "/src/art2mus/my_pipeline.py"
CUSTOM_PIPE_2 = PROJ_DIR + "/src/art2mus/my_pipeline_2.py"

# Data and Src folders
DATA_FOLDER = PROJ_DIR + "/data/"
SRC_FOLDER = PROJ_DIR + "/src/"

# Audio folders
AUDIO_FOLDER = DATA_FOLDER + "audio/"
FMA_FOLDER = AUDIO_FOLDER + "fma_large/"

# Image folders
IMAGE_FOLDER = DATA_FOLDER + "images/"
ARTGRAPH_FOLDER = IMAGE_FOLDER + "imagesf2/"

# Extra data folder
EXTRA_FOLDER = DATA_FOLDER + "extra/"
OTHER_DATASETS_VERSIONS_FOLDER = EXTRA_FOLDER + "other_datasets_version/"

""" --- Full datasets --- """
# IMAGE_AUDIO_JSON = EXTRA_FOLDER + "image_audio_df.json"
# HIST_GEO_DATASET = EXTRA_FOLDER + "loc_img_aud_df.json"
# EMOTIONAL_DATASET = EXTRA_FOLDER + "art_ag_emot_audio_df.json"
""" --- Datasets' subsets --- """
IMAGE_AUDIO_JSON = EXTRA_FOLDER + "image_audio_subset_df.json"
HIST_GEO_DATASET = EXTRA_FOLDER + "loc_img_aud_subset_df.json"
EMOTIONAL_DATASET = EXTRA_FOLDER + "art_ag_emot_audio_subset_df.json"

# ArtworkDesc-Music dataset
ARTW_DESC_MUSIC_DATASET = EXTRA_FOLDER + "artwdesc_music_df.json"
# Artwork-Dataset Dictionary
ART_DATASETS_DICT_PATH = EXTRA_FOLDER + "art_datasets_dict.json"
ART_DATASETS_DICT_SUBSET_PATH = EXTRA_FOLDER + "art_datasets_subset_dict.json"

# Artwork descriptions dataset
ARTW_DESC_FILE = EXTRA_FOLDER + "data_creation/artw_artwdesc_df.json"
ARTW_DESC_MUSIC_FILE = EXTRA_FOLDER +  "artwdesc_music_df.json"

# Embeddings' safetensors
IMAGE_ST = EXTRA_FOLDER + "images.safetensors"
AUDIO_ST = EXTRA_FOLDER + "audios.safetensors"

# Output folder
OUT_DIR = SRC_FOLDER + "test_audios/"
VAL_AUDIO_DIR = SRC_FOLDER + "art2mus/val_audios/"
TEST_AUDIO_DIR = SRC_FOLDER + "art2mus/test_audios/"

# Model Output folder
MODEL_OUT_DIR = PROJ_DIR + "/model/"

# Logs folder
LOG_DIR = PROJ_DIR + "/logs/"

# Tmp directories
TMP_GT_DIR = SRC_FOLDER + "art2mus/tmp_ground_truth/"
TMP_GEN_DIR = SRC_FOLDER + "art2mus/tmp_generated/"
TEST_GEN_AUDIOS_DIR = SRC_FOLDER + "test_audios/"

# Dtype needed to work with GPU
EMBEDS_DTYPE = torch.float16

# Negative Prompts
DEFAULT_NEGATIVE_PROMPT = "Low quality."
PERSONALIZED_NEGATIVE_PROMPT = "Low quality. Robotic noises."

# Image Projection Layer Weights
IMG_PROJ_LAYER_WEIGHTS = PROJ_DIR + "/model/img_proj_layer.pt"
TMP_LAYER_WEIGHTS_DIR = PROJ_DIR + "/model/tmp_weights/"

# VGGISH directory
TORCH_HUB_DICT = torch.hub.get_dir()
VGGISH_DIR = TORCH_HUB_DICT + "/harritaylor_torchvggish_master"

@dataclass
class TrainingConfig:
    """
    Configuration parameters for training our version of the Art2Mus model
    """
    use_snr_gamma: bool = False
    use_large_batch_size: bool = True
    use_training_subset: bool = False
    use_val_subset: bool = False
    use_cpu: bool = False
    res_from_checkpoint: bool = False
    unfreeze_unet: bool = False
    skip_train: bool = False
        
    seed: int = 0
    guidance_scale: float = 3.5
    audio_duration: int = 10 
    
    set_wandb_offline: bool = False
    
    # ------- Training and Validation Stuff -------
    num_epochs: int = 5
    small_batch_size: int = 1
    # large_batch_size: int = 16
    large_batch_size: int = 4
    max_batch_size: int = 16
    max_train_steps: int = None
    # This has to be equal to: max_batch_size/curr_batch_size (either small_batch_size or large_batch_size)
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    snr_gamma: float = 5.0
    eval_audios: int = 0
    max_eval_audios: int = 150
    
    # ------- Output Directory Stuff -------
    checkpoint_output_dir: str = PROJ_DIR + "/src/art2mus/train_checkpoints/"
    overwrite_output_dir: bool = True
    
    # ------- Checkpoint Stuff -------
    """ --- Either 'latest' or a specific path --- """
    resume_from_checkpoint: str = "latest"
    """ --- Save training state's checkpoint every X updates. --- """
    checkpointing_steps: int = 500
    checkpoints_total_limit: int = None
    
    # ------- Adam Configs ------- 
    use_8bit_adam: bool = False
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-08
    
    # -------  Learning Rate Scheduler ------- 
    """
    Choose between: 
    ["linear", "cosine", "cosine_with_restarts", "polynomial",'' "constant", "constant_with_warmup"]
    """
    learning_rate: float = 2e-5
    lr_scheduler: str = "constant" 
    lr_warmup_steps: int = 0
    
    # -------  Dataloader stuff ------- 
    """
    Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
    """
    dataloader_num_workers: int = 4
    
    # -------  AudioLDM2 stuff ------- 
    """
    Tweak these based on your preferences.
    """
    audio_duration_in_seconds: float = 10.0
    num_inference_steps: int = 200
    no_waveforms_per_prompt: int = 1


class TestConfig:
    """
    Configuration parameters for testing either the Art2Mus or AudioLDM2 model
    """
    tested_model: str = 'art2mus'
    
    set_wandb_offline: bool = True
    
    use_large_batch_size: bool = False
    small_batch_size: int = 1
    large_batch_size: int = 4
    use_test_subset: bool = True
    use_cpu: bool = False

    seed: int = 0
    guidance_scale: float = 3.5
    
    
    test_audios: int = 0
    max_test_audios: int = 10
    
    # -------  Dataloader stuff ------- 
    """
    Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
    """
    dataloader_num_workers: int = 2
    
    # -------  AudioLDM2 stuff ------- 
    """
    Tweak these based on your preferences.
    """
    audio_duration_in_seconds: float = 10.0
    num_inference_steps: int = 200
    no_waveforms_per_prompt: int = 1
    
    

def get_config_dict(curr_dict):
    """
    Returns the dictionary representation of the TrainingConfig dataclass.
    
    Returns:
        - dict: Dictionary containing the configuration parameters.
    """
    return asdict(curr_dict)


"""
###############################################
---- Update TrainingConfig with new values ----
###############################################
"""
def update_current_config(curr_config, new_config):
    """
    Update the attributes of either a TrainingConfig or TestConfig instance based on a Namespace object.

    Params:
        - config (TrainingConfig or TestConfig): An instance of the TrainingConfig or TestConfig dataclass to be updated.
        - namespace (argparse.Namespace): A Namespace object containing the new values for the config attributes.

    Returns:
        - None
    """
    for key, value in vars(new_config).items():
        if hasattr(curr_config, key):
            setattr(curr_config, key, value)


"""
#####################
Early Stopping class.
#####################
"""
class EarlyStopper:
    
    def __init__(self, patience=5, min_delta=0, goal_val='min'):

        self.patience = patience
        self.min_delta = min_delta
        self.goal_val = goal_val
        self.best_score = None
        self.counter = 0
        self.early_stop = False

        if goal_val == 'min':
            self.compare = lambda current, best: current < best - self.min_delta
        elif goal_val == 'max':
            self.compare = lambda current, best: current > best + self.min_delta
        else:
            raise ValueError(f"goal_val {goal_val} is not supported. Use 'min' or 'max'.")

    def __call__(self, current_score):

        # No score was stored
        if self.best_score is None:
            self.best_score = current_score

        else:
            # New best score
            if self.compare(current_score, self.best_score):
                self.best_score = current_score
                self.counter = 0
            # Worse score
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                    
        return self.early_stop


"""
############################################################################
How to load stft: https://github.com/declare-lab/tango/blob/master/models.py
############################################################################
"""
def load_stft():
    """
    Load and return a Short-Time Fourier Transform (STFT) module.

    Returns:
        - TacotronSTFT: Instance of TacotronSTFT initialized with default configuration parameters.
    """
    audioldm_config = default_audioldm_config()
    
    fn_STFT = TacotronSTFT(
        audioldm_config["preprocessing"]["stft"]["filter_length"],
        audioldm_config["preprocessing"]["stft"]["hop_length"],
        audioldm_config["preprocessing"]["stft"]["win_length"],
        audioldm_config["preprocessing"]["mel"]["n_mel_channels"],
        audioldm_config["preprocessing"]["audio"]["sampling_rate"],
        audioldm_config["preprocessing"]["mel"]["mel_fmin"],
        audioldm_config["preprocessing"]["mel"]["mel_fmax"],
    )
    
    fn_STFT.eval()
    return fn_STFT


""" 
##############################################################
---- Frechet Audio Distance (FAD) score between two audio ---- 
##############################################################
"""
def calculate_fad(ground_truth_dir_path, generated_audio_dir_path, model_to_use='vggish', load_from_local=False):
    """
    Calculate the Frechet Audio Distance (FAD) score between two sets of audio files.

    Params:
        - ground_truth_dir_path (str): Path to the directory containing the ground truth audio files.
        - generated_audio_dir_path (str): Path to the directory containing the generated audio files.
        - model_to_use (str): The model to use for FAD calculation, either 'vggish' or 'pann'. Default is 'vggish'.

    Returns:
        - float: The FAD score between the audio files in the two directories.
    """
    
    if load_from_local:
        # Instantiate the Frechet Audio Distance model from local cache
        frechet = MyFrechetAudioDistance(
            model_name=model_to_use,
            sample_rate=16000,
            use_pca=False, 
            use_activation=False,
            verbose=False,
            load_from_local=load_from_local,
            local_model_dir=VGGISH_DIR,
        )
    else:
        # Instantiate the Frechet Audio Distance model from Github
        frechet = FrechetAudioDistance(
            model_name=model_to_use,
            sample_rate=16000,
            use_pca=False, 
            use_activation=False,
            verbose=False,
        )
    
    # Compute FAD score among the files stored in the two directories
    return frechet.score(
        ground_truth_dir_path, 
        generated_audio_dir_path, 
        dtype="float32"
    )


""" 
####################################################################
---- Kullback-Leibler (KL) divergence between two audio signals ---- 
####################################################################
"""
def load_audio(file_path):
    """
    Load an audio file.

    Params:
        - file_path (str): Path to the audio file.

    Returns:
        - tuple: A tuple containing the audio time series and the sample rate.
    """
    y, sr = librosa.load(file_path, sr=None)
    return y, sr


def compute_stft(y, sr, n_fft=2048, hop_length=512):
    """
    Compute the Short-Time Fourier Transform (STFT) of an audio signal.

    Params:
        - y (ndarray): Audio time series.
        - sr (int): Sample rate.
        - n_fft (int): Length of the FFT window. Default is 2048.
        - hop_length (int): Number of samples between successive frames. Default is 512.

    Returns:
        - ndarray: The STFT of the audio signal.
    """
    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))**2
    return stft


def compute_kl_div(audio_path1, audio_path2):
    """
    Compute the Kullback-Leibler (KL) divergence between the spectral representations of two audio files.

    Params:
        - audio_path1 (str): Path to the first audio file.
        - audio_path2 (str): Path to the second audio file.

    Returns:
        - float: Kullback-Leibler divergence between the two audio files.
    """    
    y1, sr1 = load_audio(audio_path1)
    y2, sr2 = load_audio(audio_path2)

    # Compute STFT
    stft1 = compute_stft(y1, sr1)
    stft2 = compute_stft(y2, sr2)
    
    # Convert to decibels
    DB1 = librosa.amplitude_to_db(stft1, ref=np.max)
    DB2 = librosa.amplitude_to_db(stft2, ref=np.max)
    
    # Compute probability distribution
    # Sum across frequency bins to get distribution over time
    prob_dist1 = np.sum(DB1, axis=1)
    prob_dist2 = np.sum(DB2, axis=1)

    # Normalize (sum of all elements in the distribution has to be equal to 1)
    # Divide prob_dist1's elements by its sum
    prob_dist1 /= np.sum(prob_dist1)
    prob_dist2 /= np.sum(prob_dist2)

    """ 
    As stated in the documentation (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html)
    Providing two distributions in input to the entropy method, the latter computes the Kullback-Leibler divergence

    If two distributions don’t sum to 1, those will be normalized
    """
    return sp.stats.entropy(prob_dist1, prob_dist2)


""" 
#########################################################################
---- ImageBind Score between the generated audio and the input image ---- 
#########################################################################
"""
def compute_imagebind_score(image_embedding, gt_audio_emb, generated_audio, artw_desc=None, data_type='artw',
                            imagebind_model=None, tmp_gen_audio_dir=None, only_imgbind_am=False):
    """
    Computes the similarity score between an image embedding and generated audio or text.

    Params:
        - image_embedding (numpy.ndarray): Embeddings of the image.
        - gt_audio_emb (numpy.ndarray): Ground truth audio embeddings for comparison.
        - generated_audio (str or numpy.ndarray or torch.Tensor): Path to the generated audio file, audio data, 
          or precomputed audio embeddings.
        - artw_desc (str, optional): Description text to generate text embeddings if data_type is 'text'.
        - data_type (str): Type of the generated data ('audio' or 'text'). Default is 'audio'.
        - imagebind_model (keras.Model, optional): Pre-trained ImageBind model. If None, loads a default model.
        - tmp_gen_audio_dir (str, optional): Directory to temporarily store generated audio files. Required if 
          generated_audio is audio data.
        - only_imgbind_am (bool): If True, only the image-to-audio similarity score is returned. Default is False.

    Returns:
        - float or tuple: Similarity score(s) between the image embedding and generated audio/text embedding. 
          If only_imgbind_am is True, returns a single float (image-to-audio similarity). Otherwise, returns a tuple 
          (image-to-audio similarity, audio-to-audio similarity).
    """
    if imagebind_model is None:
        imagebind_model = load_model(False)
    
    if data_type == 'text':

        compared_emb = generate_embeds(imagebind_model, text=artw_desc, 
                                       extract_emb=True, emb_type='text')
        compared_emb = compared_emb[0]
        
    else: 
        
        compared_emb = image_embedding
        
    skip_generation = False
    
    if isinstance(generated_audio, torch.Tensor):
        audio_emb = generated_audio
        skip_generation = True
    elif isinstance(generated_audio, str):
        # If an audio path is provided in input, use it
        generated_audio_path = generated_audio
    else:
        # Otherwise temporarly store it to generate its embedding
        generated_audio_path = tmp_gen_audio_dir + f"val_audio.wav"
        scipy.io.wavfile.write(generated_audio_path, rate=16000, data=generated_audio[0])

    if not skip_generation:
        
        # Generate audio embeddings using ImageBind
        audio_emb = generate_embeds(imagebind_model, audio_paths=[generated_audio_path], 
                                    extract_emb=True, emb_type='audio')
        audio_emb = [t.numpy() for t in audio_emb]
        
        if os.path.exists(generated_audio_path):
            os.remove(generated_audio_path)
    
    if len(audio_emb) == 1:
        audio_emb = audio_emb[0]
        
    
    imgbind_score_am = compute_similarity(compared_emb, audio_emb)
    
    if not only_imgbind_am:
        imgbind_score_mm = compute_similarity(gt_audio_emb, audio_emb)
    
    if only_imgbind_am:
        return imgbind_score_am
    else:
        return imgbind_score_am , imgbind_score_mm        

""" 
###############################################
---- Methods concerning model's components ---- 
###############################################
"""    
def store_component(component, file_path=PROJ_DIR+"/src/art2mus/train_output/component.pt"):
    """
    Saves the state dictionary of a PyTorch model component to a specified file path.

    Params:
        - component: The PyTorch model or layer whose state dictionary is to be saved.
        - file_path (optional): The file path where the state dictionary will be saved. Defaults to "component.pt".
    
    Returns:
        - None
    """
    
    directory = os.path.dirname(MODEL_OUT_DIR)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    torch.save(component.state_dict(), file_path)
    

def parameters_have_changed(model, initial_params):
    """
    Checks if any parameters of a given model have changed compared to their initial values.

    Params:
        - model (torch.nn.Module): The model whose parameters are to be checked.
        - initial_params (list of torch.Tensor): A list of the initial parameters to compare against.

    Returns:
        - None
    """
    for param, initial_param in zip(model.parameters(), initial_params):
        if not torch.equal(param.data, initial_param.data):
            print("Parameters changed!")
            return
    print("Parameters didn't change!")
    return


""" 
#########################
---- Utility Methods ---- 
#########################
"""    
def empty_folder(folder_path):
    """
    Remove all files and subdirectories in a given folder, except for .gitkeep

    Params:
        - folder_path (str): Path to the folder to be emptied.

    Returns:
        - None
    """
    for filename in os.listdir(folder_path):
        if filename == '.gitkeep':
            continue
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

def is_valid_audio(audio):
    """
    Checks if all elements in the audio array are finite numbers.

    Params:
        - audio (numpy.ndarray): Array representing the audio data.

    Returns:
        - bool: True if all elements in the audio array are finite, False otherwise.
    """
    return np.isfinite(audio).all()