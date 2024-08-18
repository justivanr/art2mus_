"""
Script defining the argparse method required to run the training of the model via CLI.
"""
import sys
import argparse
sys.path.append("src")
import conf
from art2mus.utils import train_test_utils as tu


import argparse

def parse_train_args():
    parser = argparse.ArgumentParser(description="Example of a training script.")

    # Seed Stuff
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    
    # Guidance Scale
    parser.add_argument(
        "--guidance_scale", type=float, default=3.5,
        help="Guidance scale to use. Set this to 1.0 if you do not want to use it.",
    )
    
    # Training Stuff
    parser.add_argument("--skip_train", action='store_true', 
                        help="Whether to skip the first training epoch or not.")
    parser.add_argument("--unfreeze_unet", action='store_true', 
                        help="Whether to unfreeze the UNet during training.")
    
    parser.add_argument("--audio_duration", type=int, default=10, 
                        help="Desired duration of the generated audio (int).")
    
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument(
        "--max_train_steps", type=int, default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1,
        help="Number of update steps to accumulate before performing a backward/update pass.",
    )

    # SNR Gamma Loss Stuff
    parser.add_argument("--use_snr_gamma", action='store_true',
                        help="Whether to compute SNR Loss during training. If False, MSE will be computed.",)
    
    parser.add_argument(
        "--snr_gamma", type=float, default=5.0,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    
    # Batch Size Stuff
    parser.add_argument("--use_large_batch_size", action='store_true', 
                        help="Whether to use a large batch size during training.")
    parser.add_argument("--small_batch_size", default=1, 
                        type=int, help="Small batch size value. Used if --use_large_batch_size is not set.")
    parser.add_argument("--large_batch_size", default=16, 
                        type=int, help="Large batch size value. Used if --use_large_batch_size is set.")
    
    # Dataset Stuff
    parser.add_argument("--use_training_subset", action='store_true', 
                    help="Whether to use a subset of the training set.")
    parser.add_argument("--use_val_subset", action='store_true', 
                        help="Whether to use a subset of the validation set.")

    # GPU Usage Stuff
    parser.add_argument("--use_cpu", action='store_true', 
                        help="Whether to run the training on CPU instead of GPU.")
    
    # Validation Audios Stuff
    parser.add_argument("--eval_audios", default=0, 
                        type=int, help="Number of stored audios during validation.")
    parser.add_argument("--max_eval_audios", default=150, 
                        type=int, help="Max number of audios to store during validation.")
    
    # Checkpoints Stuff
    parser.add_argument("--res_from_checkpoint", action='store_true', 
                        help="Whether to train the model from a checkpoint or not.")
    
    parser.add_argument(
        "--checkpoint_output_dir", type=str, 
        default= conf.PROJ_DIR + "/src/art2mus/train_checkpoints/",
        help="The output directory where the model checkpoints will be written.",
    )
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default="latest",
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--checkpointing_steps", type=int, default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit", type=int, default=None,
        help="Max number of checkpoints to store.",
    )
    
    # Optimizer Stuff
    parser.add_argument("--use_8bit_adam", action="store_true", 
                        help="Whether to use 8-bit Adam from bitsandbytes.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, 
                        help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, 
                        help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, 
                        help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, 
                        help="Epsilon value for the Adam optimizer")
    
    # Learning Rate Scheduler Stuff
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler", type=str, default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=0, 
        help=(
            'Number of steps for the warmup in the lr scheduler.'
            ' Approximately 25% of the train batches (e.g., if with batch size 8 you have 11648 batches, this should be around 2910).'
        ),
    )
    
    # Dataloader Stuff
    parser.add_argument(
        "--dataloader_num_workers", type=int, default=2,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    
    # AudioLDM2 Stuff
    parser.add_argument(
        "--audio_duration_in_seconds", type=int, default=10, 
        help="Desired duration of the generated audio (int)."
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=200, 
        help="Number of inference steps needed to generate audios."
    )
    parser.add_argument(
        "--no_waveforms_per_prompt", type=int, default=1, 
        help="Number of audios to generate."
    )
    
    parser.add_argument("--set_wandb_offline", action='store_true', 
                        help="Whether to use Wandb offline or online.")
    
    args = parser.parse_args()

    print("Args have been parsed out! 🤗\n========================================")
    return args


def parse_test_args():
    parser = argparse.ArgumentParser(description="Example of a testing script.")

    # Seed Stuff
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
   
    # Batch Size Stuff
    parser.add_argument("--use_large_batch_size", action='store_true', 
                        help="Whether to use small or large batch size during training.")
    parser.add_argument("--small_batch_size", default=1, 
                        type=int, help="Small batch size value. Used if --use_large_batch_size is set to False.")
    parser.add_argument("--large_batch_size", default=4, 
                        type=int, help="Large batch size value. Used if --use_large_batch_size is set to True.")
    
    # Dataset Stuff
    parser.add_argument("--use_test_subset", action='store_true', 
                    help="Whether to use a subset of the test set.")

    # Dataloader Stuff
    parser.add_argument(
        "--dataloader_num_workers", type=int, default=2,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    # GPU Usage Stuff
    parser.add_argument("--use_cpu", action='store_true', 
                        help="Whether to run the training on CPU or GPU.")
    
    # Testing Audios Stuff
    parser.add_argument("--test_audios", default=0, 
                        type=int, help="Number of stored audios during testing.")
    parser.add_argument("--max_test_audios", default=150, 
                        type=int, help="Max number of audios to store during testing.")
    
    # Music Generation (Inference) Stuff
    parser.add_argument(
        "--audio_duration_in_seconds", type=float, default=10.0, 
        help="Desired duration of the generated audio (float)."
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=200, 
        help="Number of inference steps needed to generate audios."
    )
    parser.add_argument(
        "--no_waveforms_per_prompt", type=int, default=1, 
        help="Number of audios to generate."
    )
    
    parser.add_argument("--set_wandb_offline", action='store_true', 
                        help="Whether to use Wandb offline or online.")
    
    args = parser.parse_args()

    print("Args have been parsed out! 🤗\n========================================")
    return args


def main():
    # Assess if everything works properly
    tmp_configs = tu.TrainingConfig()
    print(f"Initial train config: {tmp_configs}\n========================================")
    
    args = parse_train_args()
    print(f"Parsed args: {args}")
    
    print("Updating train config.... ⛳\n========================================")
    tu.update_current_config(tmp_configs, args)
    print(f"Train config updated! ✅\n========================================{tmp_configs}")
    
    
if __name__ == "__main__":
    main()