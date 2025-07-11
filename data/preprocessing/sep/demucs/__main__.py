import json
import os
import sys
import time
from dataclasses import dataclass, field
from fractions import Fraction

import torch as th
from torch import distributed, nn
from torch.nn.parallel.distributed import DistributedDataParallel

from .augment import FlipChannels, FlipSign, Remix, Shift
from .compressed import StemsSet, build_musdb_metadata, get_musdb_tracks
from .model import Demucs
from .parser import get_name, get_parser
from .raw import Rawset
from .tasnet import ConvTasNet
from .test import evaluate
from .train import train_model, validate_model
from .utils import human_seconds, load_model, save_model, sizeof_fmt


@dataclass
class SavedState:
    """Data structure to store training state for checkpointing and resumption.
    
    Attributes:
        metrics: List of training metrics for each epoch
        last_state: Model state dict from the most recent epoch
        best_state: Model state dict that achieved the best validation performance
        optimizer: Optimizer state for training continuation
    """
    metrics: list = field(default_factory=list)
    last_state: dict = None
    best_state: dict = None
    optimizer: dict = None


def main():
    """Main function for training and evaluating audio source separation models.
    
    Handles argument parsing, model initialization, training loop execution,
    checkpointing, and final evaluation.
    """
    # Parse command line arguments
    parser = get_parser()
    args = parser.parse_args()
    name = get_name(parser, args)
    print(f"Experiment {name}")

    # Ensure MusDB dataset path is provided (if not in testing mode)
    if args.musdb is None and args.rank == 0:
        print(
            "You must provide the path to the MusDB dataset with the --musdb flag. "
            "To download the MusDB dataset, see https://sigsep.github.io/datasets/musdb.html.",
            file=sys.stderr)
        sys.exit(1)

    # Set up directory structure for outputs
    eval_folder = args.evals / name
    eval_folder.mkdir(exist_ok=True, parents=True)
    args.logs.mkdir(exist_ok=True)
    metrics_path = args.logs / f"{name}.json"
    eval_folder.mkdir(exist_ok=True, parents=True)
    args.checkpoints.mkdir(exist_ok=True, parents=True)
    args.models.mkdir(exist_ok=True, parents=True)

    # Determine which device to use (CPU/CUDA)
    if args.device is None:
        device = "cpu"
        if th.cuda.is_available():
            device = "cuda"
    else:
        device = args.device

    # Set random seed for reproducibility
    th.manual_seed(args.seed)
    # Limits thread count for museval to prevent inefficiency on NUMA systems
    os.environ["OMP_NUM_THREADS"] = "1"

    # Initialize distributed training if needed
    if args.world_size > 1:
        if device != "cuda" and args.rank == 0:
            print("Error: distributed training is only available with cuda device", file=sys.stderr)
            sys.exit(1)
        th.cuda.set_device(args.rank % th.cuda.device_count())
        distributed.init_process_group(backend="nccl",
                                       init_method="tcp://" + args.master,
                                       rank=args.rank,
                                       world_size=args.world_size)

    # Set up checkpoint paths
    checkpoint = args.checkpoints / f"{name}.th"
    checkpoint_tmp = args.checkpoints / f"{name}.th.tmp"
    if args.restart and checkpoint.exists():
        checkpoint.unlink()

    # Model initialization: either load existing model or create new
    if args.test:
        # Testing mode - load model from file
        args.epochs = 1
        args.repeat = 0
        model = load_model(args.models / args.test)
    elif args.tasnet:
        # Initialize ConvTasNet model
        model = ConvTasNet(audio_channels=args.audio_channels, samplerate=args.samplerate, X=args.X)
    else:
        # Initialize Demucs model with specified parameters
        model = Demucs(
            audio_channels=args.audio_channels,
            channels=args.channels,
            context=args.context,
            depth=args.depth,
            glu=args.glu,
            growth=args.growth,
            kernel_size=args.kernel_size,
            lstm_layers=args.lstm_layers,
            rescale=args.rescale,
            rewrite=args.rewrite,
            sources=4,  # Fixed to 4 sources: drums, bass, vocals, other
            stride=args.conv_stride,
            upsample=args.upsample,
            samplerate=args.samplerate
        )
    
    # Move model to appropriate device
    model.to(device)
    
    # If show mode, just print model structure and exit
    if args.show:
        print(model)
        size = sizeof_fmt(4 * sum(p.numel() for p in model.parameters()))
        print(f"Model size {size}")
        return

    # Initialize optimizer
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr)

    # Load saved state if exists (for resuming training)
    try:
        saved = th.load(checkpoint, map_location='cpu')
    except IOError:
        # Create new saved state if no checkpoint found
        saved = SavedState()
    else:
        # Restore model and optimizer state
        model.load_state_dict(saved.last_state)
        optimizer.load_state_dict(saved.optimizer)

    # Handle save_model mode - just save best model and exit
    if args.save_model:
        if args.rank == 0:
            model.to("cpu")
            model.load_state_dict(saved.best_state)
            save_model(model, args.models / f"{name}.th")
        return

    # Clean up completion marker if it exists
    if args.rank == 0:
        done = args.logs / f"{name}.done"
        if done.exists():
            done.unlink()

    # Set up data augmentation pipeline
    if args.augment:
        augment = nn.Sequential(
            FlipSign(),           # Randomly flip signal sign
            FlipChannels(),       # Randomly flip left-right channels
            Shift(args.data_stride),  # Apply random time shift
            Remix(group_size=args.remix_group_size)  # Remix sources
        ).to(device)
    else:
        # Minimal augmentation with just time shift
        augment = Shift(args.data_stride)

    # Set up training criterion (loss function)
    if args.mse:
        criterion = nn.MSELoss()
    else:
        criterion = nn.L1Loss()

    # Adjust number of samples to ensure valid convolution windows
    samples = model.valid_length(args.samples)
    print(f"Number of training samples adjusted to {samples}")

    # Set up training and validation datasets
    if args.raw:
        # Raw audio dataset setup
        train_set = Rawset(args.raw / "train",
                           samples=samples + args.data_stride,
                           channels=args.audio_channels,
                           streams=[0, 1, 2, 3, 4],  # mixture + 4 sources
                           stride=args.data_stride)

        valid_set = Rawset(args.raw / "valid", channels=args.audio_channels)
    else:
        # MusDB dataset setup
        if not args.metadata.is_file() and args.rank == 0:
            # Build metadata if it doesn't exist
            build_musdb_metadata(args.metadata, args.musdb, args.workers)
        if args.world_size > 1:
            # Synchronize processes in distributed training
            distributed.barrier()
        
        # Load metadata
        metadata = json.load(open(args.metadata))
        
        # Calculate duration and stride in time units
        duration = Fraction(samples + args.data_stride, args.samplerate)
        stride = Fraction(args.data_stride, args.samplerate)
        
        # Create training and validation datasets
        train_set = StemsSet(
            get_musdb_tracks(args.musdb, subsets=["train"], split="train"),
            metadata,
            duration=duration,
            stride=stride,
            samplerate=args.samplerate,
            channels=args.audio_channels
        )
        
        valid_set = StemsSet(
            get_musdb_tracks(args.musdb, subsets=["train"], split="valid"),
            metadata,
            samplerate=args.samplerate,
            channels=args.audio_channels
        )

    # Display previous training metrics if resuming
    best_loss = float("inf")
    for epoch, metrics in enumerate(saved.metrics):
        print(f"Epoch {epoch:03d}: "
              f"train={metrics['train']:.8f} "
              f"valid={metrics['valid']:.8f} "
              f"best={metrics['best']:.4f} "
              f"duration={human_seconds(metrics['duration'])}")
        best_loss = metrics['best']

    # Wrap model for distributed training if needed
    if args.world_size > 1:
        dmodel = DistributedDataParallel(
            model,
            device_ids=[th.cuda.current_device()],
            output_device=th.cuda.current_device()
        )
    else:
        dmodel = model

    # Main training loop
    for epoch in range(len(saved.metrics), args.epochs):
        begin = time.time()
        
        # Training phase
        model.train()
        train_loss = train_model(
            epoch,
            train_set,
            dmodel,
            criterion,
            optimizer,
            augment,
            batch_size=args.batch_size,
            device=device,
            repeat=args.repeat,
            seed=args.seed,
            workers=args.workers,
            world_size=args.world_size
        )
        
        # Validation phase
        model.eval()
        valid_loss = validate_model(
            epoch,
            valid_set,
            model,
            criterion,
            device=device,
            rank=args.rank,
            split=args.split_valid,
            world_size=args.world_size
        )

        # Track metrics and save checkpoints
        duration = time.time() - begin
        
        # Update best model if validation improves
        if valid_loss < best_loss:
            best_loss = valid_loss
            saved.best_state = {
                key: value.to("cpu").clone()
                for key, value in model.state_dict().items()
            }
            
        # Store metrics for this epoch
        saved.metrics.append({
            "train": train_loss,
            "valid": valid_loss,
            "best": best_loss,
            "duration": duration
        })
        
        # Save metrics to disk (master process only)
        if args.rank == 0:
            json.dump(saved.metrics, open(metrics_path, "w"))

        # Update checkpoint
        saved.last_state = model.state_dict()
        saved.optimizer = optimizer.state_dict()
        if args.rank == 0 and not args.test:
            # Save atomically by writing to temp file then renaming
            th.save(saved, checkpoint_tmp)
            checkpoint_tmp.rename(checkpoint)

        # Print epoch summary
        print(f"Epoch {epoch:03d}: "
              f"train={train_loss:.8f} valid={valid_loss:.8f} best={best_loss:.4f} "
              f"duration={human_seconds(duration)}")

    # Cleanup distributed model wrapper
    del dmodel
    
    # Final evaluation with best model weights
    model.load_state_dict(saved.best_state)
    
    # Use CPU for evaluation if specified
    if args.eval_cpu:
        device = "cpu"
        model.to(device)
        
    # Run evaluation on test set
    model.eval()
    evaluate(
        model,
        args.musdb,
        eval_folder,
        rank=args.rank,
        world_size=args.world_size,
        device=device,
        save=args.save,
        split=args.split_valid,
        shifts=args.shifts,
        workers=args.eval_workers
    )
    
    # Save final model
    model.to("cpu")
    save_model(model, args.models / f"{name}.th")
    
    # Mark experiment as complete
    if args.rank == 0:
        print("done")
        done.write_text("done")


if __name__ == "__main__":
    main()
