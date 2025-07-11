# Import necessary modules and classes
from importlib.resources import files
from model import CFM, DiT, Trainer
from prefigure.prefigure import get_all_args
import json
import os

# Limit OpenMP and MKL to single thread to avoid resource contention
os.environ['OMP_NUM_THREADS']="1"
os.environ['MKL_NUM_THREADS']="1"

def main():
    # Load configuration parameters from default config file
    args = get_all_args("config/default.ini")

    # Load model-specific configuration from JSON file
    with open(args.model_config) as f:
        model_config = json.load(f)

    # Set model class and wandb parameters based on model type
    if model_config["model_type"] == "soundsation":
        wandb_resume_id = None
        model_cls = DiT

    # Initialize the Conditioned Flow Matching model with transformer architecture
    model = CFM(
        transformer=model_cls(**model_config["model"], max_frames=args.max_frames),
        num_channels=model_config["model"]['mel_dim'],
        audio_drop_prob=args.audio_drop_prob,
        cond_drop_prob=args.cond_drop_prob,
        style_drop_prob=args.style_drop_prob,
        lrc_drop_prob=args.lrc_drop_prob,
        max_frames=args.max_frames
    )

    # Calculate and display total parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    # Initialize trainer with model and training configuration
    trainer = Trainer(
        model,
        args,
        args.epochs,
        args.learning_rate,
        num_warmup_updates=args.num_warmup_updates,
        save_per_updates=args.save_per_updates,
        checkpoint_path=f"ckpts/{args.exp_name}",
        grad_accumulation_steps=args.grad_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        wandb_project="soundsation-run",
        wandb_run_name=args.exp_name,
        wandb_resume_id=wandb_resume_id,
        last_per_steps=args.last_per_steps,
        bnb_optimizer=False,
        reset_lr=args.reset_lr,
        batch_size=args.batch_size,
        grad_ckpt=args.grad_ckpt
    )

    # Start the training process with optional resumption capability
    trainer.train(
        resumable_with_seed=args.resumable_with_seed,  # Seed for consistent dataset shuffling when resuming
    )


if __name__ == "__main__":
    main()
