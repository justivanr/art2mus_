import os
import sys
import math
import scipy
import shutil
import logging
from PIL import Image
from tqdm.auto import tqdm
from datetime import datetime
# Diffusers
from diffusers.utils import is_wandb_available
from diffusers.training_utils import compute_snr
from diffusers.optimization import get_scheduler
from diffusers.utils.torch_utils import is_compiled_module
# Accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
# Torch
import torch
from torch.utils.data import Subset
import torch.nn.functional as torch_func
# Wandb
import wandb 

sys.path.append("src")
import conf

# Directory in which the project is stored
PROJ_DIR = conf.PROJ_DIR
sys.path.append(PROJ_DIR + "/src")
sys.path.append(PROJ_DIR + "/src/audioldm")
sys.path.append(PROJ_DIR + "/src/art2mus")

# My dataset and pipeline
from my_dataset import ImageAudioDataset
from art2mus_4_pipeline import AudioLDM2Pipeline
# Tango's torch_tools
import art2mus.utils.torch_tools as tt
# AudioLDM stuff
import art2mus.utils.train_test_utils as tu
# ImageBind stuff
from art2mus.utils.imagebind_utils import load_model
# Argparse stuff
from art2mus.utils.train_test_argparse import parse_train_args


def create_artifact(path_to_image, path_to_audio, val_instance_no):
    """
    Creates W&B artifacts for the provided image and audio files, including the validation step number.

    Params:
        - path_to_image (str): Path to the image file.
        - path_to_audio (str): Path to the audio file.
        - val_instance_no (int): The validation step number.

    Returns:
        - dict: A dictionary containing the W&B Image, Audio, and validation step number artifacts.
    """
    img = Image.open(path_to_image)
    audio_caption = f"Audio generated based on {path_to_image.split('imagesf2/')[1]}"
    return {"val/step": val_instance_no,
            "artwork": wandb.Image(img), 
            "generated_audio": wandb.Audio(path_to_audio, caption=audio_caption)}


""" Logger """
logger = get_logger(__name__, log_level="INFO")

""" Folders """
# FMA folder
FMA_FOLDER = tu.FMA_FOLDER
# ArtGraph folder
ARTGRAPH_FOLDER = tu.ARTGRAPH_FOLDER
# Extra data folder
IMAGE_AUDIO_JSON = tu.IMAGE_AUDIO_JSON
IMAGE_ST = tu.IMAGE_ST
AUDIO_ST = tu.AUDIO_ST

# Output folder
VAL_AUDIO_DIR = tu.VAL_AUDIO_DIR
MODEL_OUT_DIR = tu.MODEL_OUT_DIR
LOG_DIR = tu.LOG_DIR

# AudioLDM2 HuggingFace Repo ID
AUDIOLDM_REPO = tu.AUDIOLDM2_REPO_ID

# Tmp directories (needed for FAD scores)
TMP_DIR_GT = tu.TMP_GT_DIR
TMP_DIR_GEN = tu.TMP_GEN_DIR

# Negative prompt needed during generation
NEGATIVE_PROMPT = tu.DEFAULT_NEGATIVE_PROMPT

# Default training Config
TRAIN_CONFIG = tu.TrainingConfig()

# Image Projection Layer Weights Path
LAYER_WEIGHTS = tu.IMG_PROJ_LAYER_WEIGHTS

CUSTOM_PIPE = tu.CUSTOM_PIPE_2

def main():
    
    if is_wandb_available():
        import wandb
        
    # When run via Command Line Interface (CLI), update train configs
    new_config = parse_train_args()
    tu.update_current_config(TRAIN_CONFIG, new_config)
    print("Train configs have been updated! 🤗✅")
    
    wandb_mode = os.getenv("WANDB_MODE", "online")

    if TRAIN_CONFIG.set_wandb_offline:
        os.environ["WANDB_MODE"] = "offline"
        wandb_mode = os.getenv("WANDB_MODE", "online")
        
    print(f"WANDB_MODE now set to {wandb_mode}")
    
    # Dtype needed to work with GPU
    EMBEDS_DTYPE = torch.float16
    
    # Compute SNR Loss or regular MSE
    use_snr_gamma = TRAIN_CONFIG.use_snr_gamma
    
    # Use training batches of size 1 or more
    use_large_batch_size = TRAIN_CONFIG.use_large_batch_size
    
    # Use training and validation set subsets 
    use_training_subset = TRAIN_CONFIG.use_training_subset
    use_val_subset = TRAIN_CONFIG.use_val_subset
    
    # Use only CPU
    use_cpu = TRAIN_CONFIG.use_cpu
    
    # Train the model from a checkpoint or not
    res_from_checkpoint = TRAIN_CONFIG.res_from_checkpoint
    
    # Skip first train epoch (Set to True to run validation w/o waiting for the first training epoch to end)
    skip_train = TRAIN_CONFIG.skip_train
    
    print(f"==============================================\n"
          f"Use large training set batch size? {use_large_batch_size}\n"
          f"Use training set subset? {use_training_subset}\n"
          f"Use validation set subset? {use_val_subset}\n"
          f"Compute SNR loss? {use_snr_gamma}\n"
          f"==============================================\n"
          f"Resume from latest checkpoint? {res_from_checkpoint}\n"
          f"Skip first train epoch? {skip_train}\n"
          f"=============================================="
          ) 
    
    # Load ImageBind to compute ImageBind Score during validation
    imagebind = load_model(False)
    
    if use_snr_gamma:
        TRAIN_CONFIG.snr_gamma = 5.0

    if TRAIN_CONFIG.seed is not None:
        set_seed(TRAIN_CONFIG.seed)
        
    if use_cpu:
        EMBEDS_DTYPE = torch.float32
        device = 'cpu'

    # If res_from_checkpoint is False, move the image projection layer weights (if any) to another folder
    if not res_from_checkpoint:
        if os.path.exists(LAYER_WEIGHTS):
            shutil.copy(LAYER_WEIGHTS, tu.TMP_LAYER_WEIGHTS_DIR)
            if os.path.isfile(LAYER_WEIGHTS) or os.path.islink(LAYER_WEIGHTS):
                os.remove(LAYER_WEIGHTS)
        
    # Check if there are already validation audios in the val_audios folder
    print("Checking if there are validation audios have been already stored...")
    stored_val_audios = [file for file in os.listdir(VAL_AUDIO_DIR) if ".wav" in file]
    if len(stored_val_audios) == 0:
        print("No validation audios found! ⛳")
    else:
        TRAIN_CONFIG.eval_audios = len(stored_val_audios)
        print(f"Found {len(stored_val_audios)} validation audios! Updating training config "
          f"number of validation audios to: {TRAIN_CONFIG.eval_audios} ⛳")

    """
    #########################
    ###### ACCELERATOR ######
    #########################
    """
    accelerator_project_conf = ProjectConfiguration(MODEL_OUT_DIR, LOG_DIR)

    # Assess whether to use a large batch size for the training data
    BATCH_SIZE = TRAIN_CONFIG.large_batch_size if use_large_batch_size else TRAIN_CONFIG.small_batch_size
    BATCH_SIZE = 2
    print(f"Using training batch size: {BATCH_SIZE}")

    # Update gradient_accumulation_steps based on current BATCH_SIZE
    TRAIN_CONFIG.gradient_accumulation_steps = TRAIN_CONFIG.max_batch_size // BATCH_SIZE
    
    TRAIN_CONFIG.max_eval_audios = 101 + (TRAIN_CONFIG.num_epochs * 100) 

    accelerator = Accelerator(gradient_accumulation_steps=TRAIN_CONFIG.gradient_accumulation_steps,
                              project_config=accelerator_project_conf,
                              log_with="wandb",
                              cpu=use_cpu,
                              )

    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
    logger.info(accelerator.state, main_process_only=False)

    # Generate dynamic name for Wandb's run
    now = datetime.now()
    curr_date_and_hour = now.strftime("%d_%m_%Y_%H:%M:%S")
    wandb_run_name = f"art2mus_4_training_" + curr_date_and_hour
    print(f"Run name displayed on Wandb: {wandb_run_name}")

    # Wandb trackers
    accelerator.init_trackers(
        project_name="Art2Mus", 
        config={"num_epochs": TRAIN_CONFIG.num_epochs, "gradient_accumulation_steps": TRAIN_CONFIG.gradient_accumulation_steps, 
                "batch_size": BATCH_SIZE, "learning_rate": TRAIN_CONFIG.learning_rate, "guidance_scale": TRAIN_CONFIG.guidance_scale,  
                "use_8bit_adam": TRAIN_CONFIG.use_8bit_adam, "audio_duration": TRAIN_CONFIG.audio_duration_in_seconds, 
                "val_inference_steps": TRAIN_CONFIG.num_inference_steps, "no_waveforms": TRAIN_CONFIG.no_waveforms_per_prompt, 
                "use_snr_loss": TRAIN_CONFIG.use_snr_gamma, "snr_gamma_loss": TRAIN_CONFIG.snr_gamma,
                },
        init_kwargs={"wandb":{"name":wandb_run_name}}
    )

    # Determine what device to load and move the model to (either CPU or GPU)
    if not use_cpu and torch.cuda.is_available():
        tot_gpu_mem = round(torch.cuda.mem_get_info()[1] / 1024 ** 3, 2)
        free_gpu_mem = round(torch.cuda.mem_get_info()[0] / 1024 ** 3, 2)
        
        if free_gpu_mem > 0.4 * tot_gpu_mem:
            print(f"Using GPU!\nCurrent VRAM usage: {free_gpu_mem}\{tot_gpu_mem}")
            device = "cuda"
            using_cuda = True
        else:
            EMBEDS_DTYPE = torch.float32
            print(f"Using CPU!\nCurrent VRAM usage: {free_gpu_mem}\{tot_gpu_mem}")
            device = 'cpu'
            using_cuda = False
    else:
        print(f"Using: {device}")
        using_cuda = False

    """
    ######################
    ###### PIPELINE ######
    ######################
    """
    # Load the model weights from HuggingFace, and instantiate our pipeline
    if wandb_mode == 'online':
        try:
            if device == 'cuda':
                print("Loading model with torch.float16, needed to work with GPU.")
                pipe = AudioLDM2Pipeline.from_pretrained(pretrained_model_name_or_path=AUDIOLDM_REPO, torch_dtype=torch.float16, 
                                                         custom_pipeline=CUSTOM_PIPE)
            else:
                print("Loading model without torch.float16, needed to work with CPU.")
                pipe = AudioLDM2Pipeline.from_pretrained(pretrained_model_name_or_path=AUDIOLDM_REPO, 
                                                         custom_pipeline=CUSTOM_PIPE)
        # If there are connection issues, load the model from the local cache
        except Exception as _:
            print("Loading model from local cache due to connection issues...")
            if device == 'cuda':
                pipe = AudioLDM2Pipeline.from_pretrained(pretrained_model_name_or_path=AUDIOLDM_REPO, torch_dtype=torch.float16, 
                                                         custom_pipeline=CUSTOM_PIPE,
                                                         local_files_only = True)
            else:
                pipe = AudioLDM2Pipeline.from_pretrained(pretrained_model_name_or_path=AUDIOLDM_REPO, 
                                                         custom_pipeline=CUSTOM_PIPE,
                                                         local_files_only = True)
    else:
        print("Loading model from local cache due to connection issues...")
        if device == 'cuda':
            pipe = AudioLDM2Pipeline.from_pretrained(pretrained_model_name_or_path=AUDIOLDM_REPO, torch_dtype=torch.float16, 
                                                     custom_pipeline=CUSTOM_PIPE,
                                                     local_files_only = True)
        else:
            pipe = AudioLDM2Pipeline.from_pretrained(pretrained_model_name_or_path=AUDIOLDM_REPO, 
                                                     custom_pipeline=CUSTOM_PIPE,
                                                     local_files_only = True)
        
    pipe = pipe.to(device)
    generator = torch.Generator(device).manual_seed(TRAIN_CONFIG.seed)
    
    # Freeze all the components of the architecture apart from the image projection layer
    pipe.img_project_model.requires_grad_(True) 
    pipe.projection_model.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    pipe.language_model.requires_grad_(False) 
    pipe.text_encoder.requires_grad_(False)    
    pipe.vocoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)         

    # Just to be sure, we set the Image Projection Layer to training mode and the unet to validation
    pipe.img_project_model.train()
    pipe.unet.eval()
    
    # Set the pipeline's scheduler as the noise scheduler
    noise_scheduler = pipe.scheduler
    
    print(f"Model loaded and ready to be used! ⛳\n"
          f"=====================================")

    """
    #######################
    ###### LOAD DATA ######
    #######################
    """
    print("Loading dataset...")
    dataset = ImageAudioDataset(json_file=IMAGE_AUDIO_JSON,images_dir=ARTGRAPH_FOLDER,
                                img_emb_file=IMAGE_ST, audios_dir=FMA_FOLDER, audio_emb_file=AUDIO_ST)

    train_data, val_data = dataset.train_val_test_split(val_size=0.2, random_state=0)

    """
    No processing needed. The data can be used as it is.
    """
    
    # If data subsets have to be used, retrieve them
    if use_training_subset:
        train_data_subs_amount = len(train_data) // 32
        
        print(f"=====================================\nWill work with {train_data_subs_amount} training set's instances. ⛳")
        
        subset_ids_train = list(range(train_data_subs_amount))
        train_data = Subset(train_data, subset_ids_train)
        
    if use_val_subset:
        val_data_subs_amount = len(val_data) // 10
        
        print(f"=====================================\nWill work with {val_data_subs_amount} validation set's instances. ⛳")
        
        subset_ids_val = list(range(val_data_subs_amount))
        val_data = Subset(val_data, subset_ids_val)

    # Create the Training and Validation DataLoaders
    train_dataloader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=BATCH_SIZE, 
                                                   shuffle=True,
                                                   num_workers=TRAIN_CONFIG.dataloader_num_workers,)

    val_dataloader = torch.utils.data.DataLoader(val_data, 
                                                 batch_size=TRAIN_CONFIG.small_batch_size, 
                                                 shuffle=True,
                                                 num_workers=TRAIN_CONFIG.dataloader_num_workers,)
    
    print(f"=====================================\n"
          f"- {len(train_dataloader)} training batches 🤗\n"
          f"- {len(val_dataloader)} validation batches 🤗\n"
          f"=====================================")
    
    print(f"Dataset and Dataloaders are ready to be used! ⛳\n"
          f"=====================================\n")

    # Re-Compute total training steps based on the size of the train dataloader
    update_steps_per_epoch = math.ceil(len(train_dataloader) / TRAIN_CONFIG.gradient_accumulation_steps)
    if TRAIN_CONFIG.max_train_steps is None:
            TRAIN_CONFIG.max_train_steps = TRAIN_CONFIG.num_epochs * update_steps_per_epoch

    # Update the number of traning epochs based on the no. update steps per epoch
    TRAIN_CONFIG.num_epochs = math.ceil(TRAIN_CONFIG.max_train_steps / update_steps_per_epoch)
    
    """
    ###################################################
    ###### OPTIMIZER AND LEARNING RATE SCHEDULER ######
    ###################################################
    """
    if TRAIN_CONFIG.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    
    optimizer = optimizer_cls(
        pipe.img_project_model.parameters(),
        lr=TRAIN_CONFIG.learning_rate,
        betas=(TRAIN_CONFIG.adam_beta1, TRAIN_CONFIG.adam_beta2),
        weight_decay=TRAIN_CONFIG.adam_weight_decay,
        eps=TRAIN_CONFIG.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
            TRAIN_CONFIG.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=TRAIN_CONFIG.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=TRAIN_CONFIG.max_train_steps * accelerator.num_processes,
        )

    
    pipe.img_project_model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        pipe.img_project_model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    # Load Short-Time Fourier Transform (STFT) module 
    stft = tu.load_stft()
    target_length = int(TRAIN_CONFIG.audio_duration * 102.4)

    # Number of completed train and validation steps 
    global_step = 0
    completed_val_steps = TRAIN_CONFIG.eval_audios
    first_epoch = 0

    total_batch_size = BATCH_SIZE * accelerator.num_processes * TRAIN_CONFIG.gradient_accumulation_steps
    logger.info("***** Training *****")
    logger.info(f"No. examples = {dataset.__len__()}")
    logger.info(f"No. training examples = {train_data.__len__()}")
    logger.info(f"No. validation examples = {val_data.__len__()}")
    logger.info(f"No. training epochs = {TRAIN_CONFIG.num_epochs}")
    logger.info(f"Instantaneous batch size per device = {BATCH_SIZE}")
    logger.info(f"Tot. train batch size = {total_batch_size}")
    logger.info(f"Gradient accumulation steps = {TRAIN_CONFIG.gradient_accumulation_steps}")
    logger.info(f"Tot. optimization steps = {TRAIN_CONFIG.max_train_steps}")
    logger.info(f"****************************************************")
    
    # Potentially load in the weights and states from a previous save
    if res_from_checkpoint and TRAIN_CONFIG.resume_from_checkpoint:
        if TRAIN_CONFIG.resume_from_checkpoint != "latest":
            path = os.path.basename(TRAIN_CONFIG.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(TRAIN_CONFIG.checkpoint_output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{TRAIN_CONFIG.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            TRAIN_CONFIG.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(TRAIN_CONFIG.checkpoint_output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // update_steps_per_epoch

    else:
        initial_global_step = 0

    no_steps_per_epoch = len(train_dataloader) // TRAIN_CONFIG.gradient_accumulation_steps
    print(f"Will check if epochs has to end after {no_steps_per_epoch} steps.\n\n")

    print(f"{first_epoch}/{TRAIN_CONFIG.num_epochs} epochs have already been completed. Starting from epoch no. {first_epoch+1}...\n=============\n")

    progress_bar = tqdm(
                    range(0, TRAIN_CONFIG.max_train_steps),
                    initial=initial_global_step,
                    desc="Current step (w.r.t. max train steps)",
                    # Only show the progress bar once on each machine.
                    disable=not accelerator.is_local_main_process,
                )
    
    # If i skip the first train epoch, i add one to restore it
    if skip_train:
        first_epoch -= 1
    
    # Loss per specific timesteps
    timesteps_list = [10, 50, 100, 250, 500, 800]
    ts_loss_dict = {f'{ts_val}': [] for ts_val in timesteps_list}
    
    for epoch in tqdm(range(first_epoch, TRAIN_CONFIG.num_epochs), desc="Training epochs"):
            
            # Training step loss
            train_step_loss = 0.0
            # Total epoch loss
            epoch_loss = 0.0
            
            validation_done = False
            
            noise_scheduler.set_timesteps(pipe.scheduler.config.num_train_timesteps)
            
            for _, batch in enumerate(train_dataloader):
                            
                # Skip first train epoch if needed
                if skip_train:
                    break
                
                with accelerator.accumulate(pipe.img_project_model):
                                    
                    image_emb, audio_path = batch
                    
                    # Repeat the Negative Prompt based on the batch size (image_emb.shape[0])
                    if image_emb.shape[0] > 1:
                        negative_prompt = [NEGATIVE_PROMPT] * image_emb.shape[0]
                    else:
                        negative_prompt = NEGATIVE_PROMPT
                    
                    """ --- Convert audio to latent space --- """
                    # Compute mel-spectrogram of the audio (this is our ground truth)
                    try:
                        mel, _, _ = tt.wav_to_fbank(audio_path, target_length, stft)
                    except Exception as e:
                        print(f"Issues with {audio_path}: {e}")
                        progress_bar.update(1)
                        global_step += 1
                        continue
                        
                    mel = mel.unsqueeze(1).to(device)
                    mel = mel.to(dtype=EMBEDS_DTYPE)
                    
                    """ --- Latents Computation --- """
                    # Compute latents starting from the mel-spectrogram
                    with torch.no_grad():
                        latents = pipe.vae.encode(mel).latent_dist.sample()
                    latents = latents * pipe.vae.config.scaling_factor

                    """ --- Noise Generation --- """
                    # Sample random noise to add to the latents
                    noise = torch.randn_like(latents)
                    target = noise
                    
                    """ --- Sample Timestep --- """
                    timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, 
                                            (latents.shape[0],), device=latents.device)
                    timesteps = timesteps.long()
                    
                    """ --- Noisy Latents Computation --- """
                    # Add noise to previously computed latents (fed in input to the UNet)
                    noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
                    noisy_latents = noisy_latents.to(device=device)
                    
                    """ --- Noise Generation Procedure --- """
                    generated_noise, _ = pipe.__train__(
                        image_embeds=image_emb,
                        negative_prompt=negative_prompt,
                        num_waveforms_per_prompt=TRAIN_CONFIG.no_waveforms_per_prompt,
                        latents=noisy_latents,
                        guidance_scale=TRAIN_CONFIG.guidance_scale,
                        timesteps=timesteps,
                    )

                    """ --- Loss Computation --- """
                    if not use_snr_gamma:
                        """
                        Standard Mean Squared Error (MSE).
                        'reduction' parameter values:
                            - 'none': no reduction applied to the loss;
                            - 'mean': the mean of the output will be taken;
                            - 'sum': the output will be summed.
                        """ 
                        loss = torch_func.mse_loss(generated_noise.float(), target.float(), reduction="none")
                        
                        # Check if the loss has been computed at a specific timestamp (timesteps_list)
                        for idts, ts in enumerate(timesteps):
                            ts = str(ts.item())
                            if ts in ts_loss_dict:
                                ts_loss_dict[ts].append(loss[idts].mean().item())
                                                
                        # Compute the mean loss for each timesteps' loss
                        for ts_key, ts_losses in ts_loss_dict.items():
                            if len(ts_losses) != 0:
                                loss_sum = sum(ts_losses)
                                mean_loss = loss_sum / len(ts_losses)
                                accelerator.log({f"train/step": global_step,
                                                f"train/timestep_{ts_key}_loss": mean_loss})
                                # Reset the key of the specific timestep
                                ts_loss_dict[ts_key] = []
                        
                        loss = loss.mean()
                        
                    else:
                        """
                        Signal to Noise Ratio Loss.
                        """
                        snr = compute_snr(noise_scheduler, timesteps)
                        mse_loss_weights = torch.stack([snr, TRAIN_CONFIG.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                            dim=1
                        )[0]
                        mse_loss_weights = mse_loss_weights / snr

                        loss = torch_func.mse_loss(generated_noise.float(), target.float(), reduction="none")
                        
                        # Check if the loss has been computed at a specific timestamp (timesteps_list)
                        for idts, ts in enumerate(timesteps):
                            ts = str(ts.item())
                            if ts in ts_loss_dict:
                                ts_loss_dict[ts].append(loss[idts].mean().item())
                                                
                        # Compute the mean loss for each timesteps' loss
                        for ts_key, ts_losses in ts_loss_dict.items():
                            if len(ts_losses) != 0:
                                loss_sum = sum(ts_losses)
                                mean_loss = loss_sum / len(ts_losses)
                                accelerator.log({f"train/step": global_step,
                                                f"train/timestep_{ts_key}_loss": mean_loss})
                                # Reset the key of the specific timestep
                                ts_loss_dict[ts_key] = []
                        
                        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                        loss = loss.mean()
                        
                    # Gather losses across all processes for logging (if distributed training is used)
                    avg_loss = accelerator.gather(loss.repeat(BATCH_SIZE)).mean()
                    # Update step and epoch loss
                    train_step_loss += avg_loss.item() / TRAIN_CONFIG.gradient_accumulation_steps
                    epoch_loss += train_step_loss
                    
                    # Backpropagate the computed loss
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(pipe.img_project_model.parameters(), TRAIN_CONFIG.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                # Checks if the accelerator has performed an optimization step
                if accelerator.sync_gradients:
                    
                    progress_bar.update(1)
                    global_step += 1
                    # Log current step's loss on Wandb
                    accelerator.log({f"train/step": global_step, f"train/step_loss": train_step_loss})
                                    
                    # Reset step loss
                    train_step_loss = 0.0
                    
                    if global_step % TRAIN_CONFIG.checkpointing_steps == 0:
                        if accelerator.is_main_process:
                            print(f"Storing checkpoint!\n=========================")
                            # Check if this save would set us over the `checkpoints_total_limit`
                            if TRAIN_CONFIG.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(TRAIN_CONFIG.checkpoint_output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                # Before saving the new checkpoint, we need remove some of the stored ones
                                if len(checkpoints) >= TRAIN_CONFIG.checkpoints_total_limit:
                                    num_to_remove = len(checkpoints) - TRAIN_CONFIG.checkpoints_total_limit + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]
                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(TRAIN_CONFIG.checkpoint_output_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)

                            # Store checkpoint
                            save_path = os.path.join(TRAIN_CONFIG.checkpoint_output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")
                    
                # Update train progress bar
                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                
                # Check if the training loop needs to be stopped
                if global_step >= TRAIN_CONFIG.max_train_steps:
                    break
                elif accelerator.sync_gradients and global_step > 0 and (global_step % no_steps_per_epoch) == 0:
                    break
           
            # Compute epoch's loss and log it on Wandb
            if not skip_train:
                epoch_loss = epoch_loss / len(train_dataloader)
                accelerator.log({"train/epoch": epoch, "train/epoch_loss": epoch_loss, 
                                 "train/step": global_step})
            
                # Store epoch checkpoint
                save_path = os.path.join(TRAIN_CONFIG.checkpoint_output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
                
                print(f"***************************************************\n"
                    f"Training epoch {epoch} completed! Starting validation....\n"
                    f"***************************************************\n"
                    )      
            else:
                print(f"***************************************************\n"
                      f"Skipped first epoch! Starting validation....\n"
                      f"***************************************************\n"
                      )
                skip_train = False        
                

            """
            Run validation after each training epoch.
            """
            # Total epoch loss
            val_fad_score = 0.0
            val_imgbind_score_am = 0.0
            val_imgbind_score_mm = 0.0
            val_kl_div = 0.0
            
            if accelerator.is_main_process:
                if val_dataloader is not None and validation_done == False:
                    
                    total_instances = val_dataloader.__len__()
                    instance_completed = 0
                    noise_scheduler.set_timesteps(TRAIN_CONFIG.num_inference_steps)

                    for step, batch in enumerate(val_dataloader):
                        print(f"Currently: {instance_completed}/{total_instances} instances")
                        with torch.no_grad():
                            
                            image_emb, audio_path = batch
                            audio_path = audio_path[0]
                            
                            # Retrieve image path based on image embedding (needed to log Wandb artifact)
                            img_path = dataset.__get_image_name_from_emb__(image_emb.cpu().detach())
                            gt_aud_emb = dataset.__get_aud_emb_from_path__(audio_path)
                            
                            """ --- Inference - Audio Generation --- """
                            gen_music = pipe(
                                image_embeds=image_emb,
                                negative_prompt=NEGATIVE_PROMPT,
                                num_inference_steps=TRAIN_CONFIG.num_inference_steps,
                                audio_length_in_s=TRAIN_CONFIG.audio_duration_in_seconds,
                                num_waveforms_per_prompt=TRAIN_CONFIG.no_waveforms_per_prompt,
                                generator=generator,
                                guidance_scale=TRAIN_CONFIG.guidance_scale,
                            ).audios

                            # Empty folder after 500 music files have been stored [avoid storing too many files at once]
                            if TRAIN_CONFIG.eval_audios == 500:
                                tu.empty_folder(VAL_AUDIO_DIR)
                                TRAIN_CONFIG.eval_audios = 0

                            if TRAIN_CONFIG.eval_audios < TRAIN_CONFIG.max_eval_audios:
                                generated_audio_path = VAL_AUDIO_DIR + f"val_audio_{TRAIN_CONFIG.eval_audios}.wav"
                                scipy.io.wavfile.write(generated_audio_path, rate=16000, data=gen_music[0])
                                TRAIN_CONFIG.eval_audios += 1
                            
                                # Create Wandb artifact with artwork and generated audio, and log it
                                wandb_artifact = create_artifact(img_path, generated_audio_path, completed_val_steps)
                                accelerator.log(wandb_artifact)
                                # accelerator.log(wandb_artifact, step=completed_val_steps)
                            
                            """ --- Metrics Computation ---"""
                            # KL Divergence
                            kl_div = tu.compute_kl_div(audio_path, generated_audio_path)
                            
                            # ImageBind Score
                            imgbind_score_am, imgbind_score_mm = tu.compute_imagebind_score(image_embedding=image_emb,
                                                                                            gt_audio_emb=gt_aud_emb, 
                                                                                            generated_audio=gen_music,
                                                                                            imagebind_model=imagebind, 
                                                                                            tmp_gen_audio_dir=TMP_DIR_GT)
                            
                            # FAD Score
                            if using_cuda:
                                
                                # Copy files before computing FAD score
                                shutil.copy(audio_path, TMP_DIR_GT)
                                shutil.copy(generated_audio_path, TMP_DIR_GEN)
                                
                                try:
                                    fad_score = tu.calculate_fad(ground_truth_dir_path=TMP_DIR_GT,
                                                                 generated_audio_dir_path=TMP_DIR_GEN,
                                                                 load_from_local=True)
                                    val_fad_score += fad_score
                                except Exception as _:
                                    fad_score=None
                                
                                # Remove files after computing FAD score
                                tu.empty_folder(TMP_DIR_GT)
                                tu.empty_folder(TMP_DIR_GEN)
                            
                            else:
                                fad_score = None
                            
                            val_imgbind_score_am += imgbind_score_am
                            val_imgbind_score_mm += imgbind_score_mm
                            val_kl_div += kl_div
                            
                            logger.info("***** Validation Metrics *****")
                            logger.info(f"KL-Divergence = {kl_div}")
                            logger.info(f"ImageBind Score Artwork-Music = {imgbind_score_am}")
                            logger.info(f"ImageBind Score Music-Music = {imgbind_score_mm}")
                            if using_cuda:
                                logger.info(f"FAD Score = {fad_score}")
                            
                            # Log validation metrics on Wandb
                            accelerator.log({"val/step": completed_val_steps, 
                                             "val/kl_div": kl_div, 
                                             "val/fad_score": fad_score, 
                                             "val/imagebind_score_am": imgbind_score_am,
                                             "val/imagebind_score_mm": imgbind_score_mm})
                            
                            instance_completed += 1
                            completed_val_steps += 1
                            
                    # Validation completed
                    validation_done = True
                    print(f"***************************************************\n"
                          f"Validation for epoch {epoch} completed!\n"
                          f"***************************************************\n"
                          )
                    
                    # Log average validation metrics after each validation
                    # If no fad score was computed, we have val_fad_score == 0.0
                    val_fad_score = val_fad_score / len(val_dataloader)
                    val_imgbind_score_am = val_imgbind_score_am / len(val_dataloader)
                    val_imgbind_score_mm = val_imgbind_score_mm / len(val_dataloader)
                    val_kl_div = val_kl_div / len(val_dataloader)
                    
                    accelerator.log({"val/step": completed_val_steps, 
                                     "val/avg_fad_score": val_fad_score,
                                     "val/avg_imgbind_score_am": val_imgbind_score_am,
                                     "val/avg_imgbind_score_mm": val_imgbind_score_mm,
                                     "val/avg_kl_div": val_kl_div})
                    
                    # Reset these metrics values after logging them
                    val_fad_score = 0.0
                    val_imgbind_score_am = 0.0
                    val_imgbind_score_mm = 0.0
                    val_kl_div = 0.0
                    
    # Store Image Projection Model weights after training
    accelerator.wait_for_everyone()
    # Rename the .pt file name if you want
    tu.store_component(pipe.img_project_model, MODEL_OUT_DIR + "img_proj_layer.pt")
    accelerator.end_training()
    
    # Remove the remaining generated music files
    tu.empty_folder(VAL_AUDIO_DIR)

if __name__ == "__main__":
    print(f"=============================\nStarting training... 🤗")
    main()
    print(f"=============================\nTraining completed! ⛳✅")