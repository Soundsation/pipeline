import webrtcvad  # Import WebRTC Voice Activity Detection library
import torch.multiprocessing as mp  # For parallel processing
import os  # For file and directory operations
import threading  # For thread management
from tqdm import tqdm  # For progress bars
import traceback  # For exception handling
import argparse  # For command line argument parsing
import glob  # For file pattern matching

# Locks to prevent concurrent file access conflicts
vocal_file_lock = threading.Lock()
bgm_file_lock = threading.Lock()

# Import custom functions from vad_tool module
from vad_tool import read_wave_to_frames, vad_generator, cut_points_generator, cut_points_storage_generator, wavs_generator

# Logging configuration
LOGGING_INTERVAL = 3  # Update progress every 3 seconds

# Audio processing parameters
SAMPLE_RATE = 16000  # Audio sample rate in Hz
FRAME_DURATION = 10  # Frame duration in milliseconds

# VAD configuration parameters
MIN_ACTIVE_TIME_MS = 100  # Minimum duration for an active speech segment (ms)
SIL_HEAD_TAIL_MS = 50  # Silence to preserve at beginning and end of segments (ms)
SIL_MID_MS = 200  # Maximum allowed silence duration within a segment (ms)
CUT_MIN_MS = 800  # Minimum total duration for a valid segment (ms)
CUT_MAX_MS = 20000  # Maximum total duration for a valid segment (ms)

# Convert time-based parameters to frame counts
MIN_ACTIVE_FRAME = MIN_ACTIVE_TIME_MS // FRAME_DURATION
SIL_FRAME = SIL_HEAD_TAIL_MS // FRAME_DURATION
SIL_MID_FRAME = SIL_MID_MS // FRAME_DURATION
CUT_MIN_FRAME = CUT_MIN_MS // FRAME_DURATION
CUT_MAX_FRAME = CUT_MAX_MS // FRAME_DURATION
RANDOM_MIN_FRAME = True  # Whether to randomize minimum frame threshold

def inference(rank, out_dir, filelist_name, queue: mp.Queue):
    """
    Worker function that processes audio files from the queue.
    
    Args:
        rank: Worker ID for this process
        out_dir: Directory to save output files
        filelist_name: Name of the file list being processed
        queue: Multiprocessing queue containing files to process
    """
    # Create directory for VAD information files
    info_dir = os.path.join(out_dir, "vad_info")
    os.makedirs(info_dir, exist_ok=True)
    
    while True:
        # Get next file from queue
        input_path = queue.get()
        
        # Exit condition: None is sent to the queue
        if input_path is None:
            break
            
        try:
            # Create a new VAD instance with aggressiveness level 3 (0-3, higher is more aggressive)
            vad_tools = webrtcvad.Vad(3)  # New instance for each file to avoid bugs
            
            # Get the vocal path from the input (assuming it's a list with the path as first element)
            vocal_path = input_path[0]
            
            # Extract filename without extension
            filename = os.path.basename(vocal_path).split(".")[0]
            
            # Read and prepare audio frames
            frames, wav = read_wave_to_frames(vocal_path, SAMPLE_RATE, FRAME_DURATION)
            
            # Generate VAD information (which frames contain speech)
            vad_info = vad_generator(frames, SAMPLE_RATE, vad_tools)

            # Generate cut points based on VAD information and parameters
            cut_points = cut_points_generator(
                vad_info, 
                MIN_ACTIVE_FRAME, 
                SIL_FRAME, 
                SIL_MID_FRAME, 
                CUT_MIN_FRAME, 
                CUT_MAX_FRAME, 
                RANDOM_MIN_FRAME
            )
            
            # Generate storage content for cut points information
            raw_vad_content, file_content = cut_points_storage_generator(vad_info, cut_points, CUT_MIN_MS)

            # Write VAD information to output file
            with open(os.path.join(info_dir, filename+".txt"), "w") as f:
                f.write(file_content)

        except Exception as e:
            # Print full exception traceback and message if an error occurs
            traceback.print_exc()
            print(e)

def setInterval(interval):
    """
    Decorator factory that creates a decorator to run a function periodically.
    
    Args:
        interval: Time in seconds between function calls
        
    Returns:
        Decorator function
    """
    def decorator(function):
        def wrapper(*args, **kwargs):
            # Event to signal stopping the periodic execution
            stopped = threading.Event()

            def loop():  # Executed in another thread
                while not stopped.wait(interval):  # Continue until stopped
                    function(*args, **kwargs)

            # Create and start daemon thread
            t = threading.Thread(target=loop)
            t.daemon = True  # Thread will exit when main program exits
            t.start()
            return stopped  # Return event object to allow stopping

        return wrapper
    return decorator


# Global variable to store previous queue size
last_batches = None


@setInterval(LOGGING_INTERVAL)
def QueueWatcher(queue, bar):
    """
    Updates the progress bar based on queue size.
    Runs periodically based on LOGGING_INTERVAL.
    
    Args:
        queue: The multiprocessing queue to watch
        bar: The tqdm progress bar to update
    """
    global last_batches
    curr_batches = queue.qsize()  # Get current queue size
    bar.update(last_batches-curr_batches)  # Update progress bar
    last_batches = curr_batches  # Store current size for next update


if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--filelist_or_dir", type=str, required=True, 
                        help="Path to file list or directory containing audio files")
    parser.add_argument("--out_dir", type=str, required=True, 
                        help="Directory to save output files")
    parser.add_argument("--jobs", type=int, required=False, default=2, 
                        help="Number of parallel processing jobs")
    parser.add_argument("--log_dir", type=str, required=False, default="large-v3", 
                        help="Directory for logs")
    parser.add_argument("--model_dir", type=str, required=False, default="large-v3", 
                        help="Directory containing model files")
    args = parser.parse_args()

    # Extract arguments
    filelist_or_dir = args.filelist_or_dir
    out_dir = args.out_dir
    NUM_THREADS = args.jobs

    # Handle input: either a file list or a directory
    if os.path.isfile(filelist_or_dir):
        # If input is a file, read lines from it
        filelist_name = filelist_or_dir.split('/')[-1].split('.')[0]
        generator = [x for x in open(filelist_or_dir).read().splitlines()]
    else:
        # If input is a directory, find .mp3 files with "Vocals" in the name
        filelist_name = "vocals"
        generator = [x for x in glob.glob(f"{filelist_or_dir}/*.mp3") if x.endswith(".Vocals.mp3")]
    
    # Initialize multiprocessing with spawn method (required for cross-platform compatibility)
    mp.set_start_method('spawn', force=True)

    print(f"Running with {NUM_THREADS} threads and batchsize 1")
    
    # Create and start worker processes
    processes = []
    queue = mp.Queue()  # Create shared queue for tasks
    for rank in range(NUM_THREADS):
        p = mp.Process(target=inference, args=(rank, out_dir, filelist_name, queue), daemon=True)
        p.start()
        processes.append(p)

    # Add tasks to the queue
    accum = []  # Temporary accumulator for batching (currently batch size is 1)
    for filename in tqdm(generator):
        accum.append(filename)
        if len(accum) == 1:  # When batch is complete (size 1)
            queue.put(accum.copy())  # Add to queue
            accum.clear()  # Clear accumulator

    # Add termination signals to queue (one per worker)
    for _ in range(NUM_THREADS):
        queue.put(None)

    # Set up progress tracking
    last_batches = queue.qsize()
    bar = tqdm(total=last_batches)
    queue_watcher = QueueWatcher(queue, bar)  # Start periodic progress updates
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
        
    # Stop the queue watcher
    queue_watcher.set()
    