import torch
import random

class DiffusionDataset(torch.utils.data.Dataset):
    """
    Dataset for training diffusion models on audio-lyrics pairs.
    This dataset loads audio latents, lyrics with timestamps, and style prompts.
    
    Args:
        file_path (str): Path to the file containing data samples info
        max_frames (int): Maximum number of frames to use from each sample
        min_frames (int): Minimum number of frames required for a valid sample
        sampling_rate (int): Audio sampling rate in Hz
        downsample_rate (int): Rate to downsample audio features
        precision (str): Floating point precision ('fp16', 'bf16', or 'fp32')
    """
    def __init__(self, file_path, max_frames=2048, min_frames=512, sampling_rate=44100, downsample_rate=2048, precision='fp16'):
        # Store dataset parameters
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.sampling_rate = sampling_rate
        self.downsample_rate = downsample_rate
        # Calculate max duration in seconds based on frames and rates
        self.max_secs = max_frames / (sampling_rate / downsample_rate)
        
        # Load the file list from provided path
        self.file_path = file_path
        with open(file_path, 'r') as f:
            self.file_lst = [line.strip() for line in f.readlines()]

        # Define special token IDs for lyrics processing
        self.pad_token_id = 0       # Token for padding
        self.comma_token_id = 1     # Token for comma
        self.period_token_id = 2    # Token for period/end
        self.start_token_id = 355   # Token indicating sequence start

        # Set data type based on precision parameter
        if precision == 'fp16':
            self.feature_dtype = torch.float16
        elif precision == 'bf16':
            self.feature_dtype = torch.bfloat16
        elif precision == 'fp32':
            self.feature_dtype = torch.float32

        # Shuffle the dataset with fixed seed for reproducibility
        random.seed(42)
        random.shuffle(self.file_lst)

    def load_item(self, item, field):
        """
        Helper method to load an item from a Lance database.
        
        Args:
            item: The item to load
            field: The field to extract
            
        Returns:
            The loaded item or None if loading fails
        """
        try:
            item, reader_idx = item[field]
            item = self.lance_connections[reader_idx].get_datas_by_rowids([item._rowid])[0]
        except Exception as e:
            return None
        return item
    
    def get_triple(self, item):
        """
        Process a single data sample to extract prompt, lyrics, and audio latent.
        
        Args:
            item (str): A pipe-separated string containing paths to utterance, lyrics, 
                        latent features, and style prompt
                        
        Returns:
            tuple: (prompt, lyrics, latent, normalized_start_time)
                - prompt: Style embedding tensor
                - lyrics: Tokenized lyrics tensor aligned with audio frames
                - latent: Audio latent features
                - normalized_start_time: Start position normalized to [0,1]
        """
        # Split the input string to get file paths
        utt, lrc_path, latent_path, style_path = item.split("|")

        # Load lyrics with timestamps
        time_lrc = torch.load(lrc_path, map_location='cpu')
        input_times = time_lrc['time']
        input_lrcs = time_lrc['lrc']
        lrc_with_time = list(zip(input_times, input_lrcs))
        
        # Load audio latent features
        latent = torch.load(latent_path, map_location='cpu') # [batch, dim, time]
        latent = latent.squeeze(0)  # Remove batch dimension
        
        # Load style prompt embedding
        prompt = torch.load(style_path, map_location='cpu') # [batch, dim]
        prompt = prompt.squeeze(0)  # Remove batch dimension
        
        # Randomly select a starting frame within the sample
        max_start_frame = max(0, latent.shape[-1] - self.max_frames)
        start_frame = random.randint(0, max_start_frame)
        start_time = start_frame * self.downsample_rate / self.sampling_rate
        normalized_start_time = start_frame / latent.shape[-1]  # Normalize to [0,1]
        latent = latent[:, start_frame:]  # Trim latent to start at selected frame
    
        # Filter lyrics to only include those after start_time and before max_secs
        lrc_with_time = [(time_start - start_time, line) for (time_start, line) in lrc_with_time if (time_start - start_time) >= 0]
        lrc_with_time = [(time_start, line) for (time_start, line) in lrc_with_time if time_start < self.max_secs]

        # Determine end time from last lyric or raise exception if no lyrics
        if len(lrc_with_time) >= 1:
            latent_end_time = lrc_with_time[-1][0]
        else:
            raise Exception("No valid lyrics found within time window")
        
        # Drop last lyric for 2048-frame mode to avoid cutoff
        if self.max_frames == 2048:
            lrc_with_time = lrc_with_time[:-1] if len(lrc_with_time) >= 1 else lrc_with_time

        # Initialize lyrics tensor with zeros (padding)
        lrc = torch.zeros((self.max_frames,), dtype=torch.long)
        
        # Place lyrics tokens at appropriate frame positions
        tokens_count = 0
        last_end_pos = 0
        for time_start, line in lrc_with_time:
            # Convert periods to commas except for end of line
            tokens = [token if token != self.period_token_id else self.comma_token_id for token in line] + [self.period_token_id]
            tokens = torch.tensor(tokens, dtype=torch.long)
            num_tokens = tokens.shape[0]

            # Calculate frame position from timestamp
            gt_frame_start = int(time_start * self.sampling_rate / self.downsample_rate)
            
            # No frame shift applied (could be used for augmentation)
            frame_shift = 0

            # Ensure lyrics don't overlap by using max of calculated position and last end position
            frame_start = max(gt_frame_start - frame_shift, last_end_pos)
            # Ensure lyrics fit within max frames
            frame_len = min(num_tokens, self.max_frames - frame_start)

            # Place tokens in the lyrics tensor
            lrc[frame_start:frame_start + frame_len] = tokens[:frame_len]

            tokens_count += num_tokens
            last_end_pos = frame_start + frame_len        

        # Trim latent to end at the last lyric time
        latent = latent[:, :int(latent_end_time * self.sampling_rate / self.downsample_rate)]

        # Convert tensors to specified precision
        latent = latent.to(self.feature_dtype)
        prompt = prompt.to(self.feature_dtype)

        return prompt, lrc, latent, normalized_start_time

    def __getitem__(self, index):
        """
        Get a single item from the dataset.
        Automatically retries with a random index if the selected sample is invalid.
        
        Args:
            index (int): Index of the item to get
            
        Returns:
            dict: Contains 'prompt', 'lrc', 'latent', and 'start_time'
        """
        idx = index
        while True:
            try:
                # Try to get and process the sample
                prompt, lrc, latent, start_time = self.get_triple(self.file_lst[idx])
                # Skip samples that are too short
                if latent.shape[-1] < self.min_frames:
                    raise Exception("Sample too short")
                
                # Return valid sample as a dictionary
                item = {'prompt': prompt, "lrc": lrc, "latent": latent, "start_time": start_time}
                return item
            except Exception as e:
                # On failure, try a different random index
                idx = random.randint(0, self.__len__() - 1)
                continue

    def __len__(self):
        """
        Return the number of samples in the dataset.
        
        Returns:
            int: Dataset size
        """
        return len(self.file_lst)

    def custom_collate_fn(self, batch):
        """
        Custom collate function for creating batches.
        Handles padding of variable-length sequences.
        
        Args:
            batch (list): List of sample dictionaries
            
        Returns:
            dict: Batched samples with padded tensors and length information
        """
        # Extract individual tensors from each sample
        latent_list = [item['latent'] for item in batch]
        prompt_list = [item['prompt'] for item in batch]
        lrc_list = [item['lrc'] for item in batch]
        start_time_list = [item['start_time'] for item in batch]

        # Store original lengths before padding
        latent_lengths = torch.LongTensor([latent.shape[-1] for latent in latent_list])
        prompt_lengths = torch.LongTensor([prompt.shape[-1] for prompt in prompt_list])
        lrc_lengths = torch.LongTensor([lrc.shape[-1] for lrc in lrc_list])

        # Find maximum prompt length for padding
        max_prompt_length = prompt_lengths.amax()

        # Pad prompts to equal length
        padded_prompt_list = []
        for prompt in prompt_list:
            padded_prompt = torch.nn.functional.pad(prompt, (0, max_prompt_length - prompt.shape[-1]))
            padded_prompt_list.append(padded_prompt)

        # Pad latents to max_frames
        padded_latent_list = []
        for latent in latent_list:
            padded_latent = torch.nn.functional.pad(latent, (0, self.max_frames - latent.shape[-1]))
            padded_latent_list.append(padded_latent)

        # No need to pad start times (they're scalar values)
        padded_start_time_list = start_time_list

        # Stack individual tensors into batched tensors
        prompt_tensor = torch.stack(padded_prompt_list)
        lrc_tensor = torch.stack(lrc_list)
        latent_tensor = torch.stack(padded_latent_list)
        start_time_tensor = torch.tensor(padded_start_time_list)

        # Return all tensors and metadata in a single dictionary
        return {
            'prompt': prompt_tensor,               # Style embeddings
            'lrc': lrc_tensor,                     # Lyrics tokens
            'latent': latent_tensor,               # Audio latent features
            "prompt_lengths": prompt_lengths,      # Original prompt lengths
            "lrc_lengths": lrc_lengths,            # Original lyrics lengths
            "latent_lengths": latent_lengths,      # Original latent lengths
            "start_time": start_time_tensor        # Normalized start positions
        }


if __name__ == "__main__":
    # Test the dataset by creating an instance and accessing the first item
    dd = DiffusionDataset("train.scp", 2048, 512)
    x = dd[0]
    import pdb; pdb.set_trace()  # Breakpoint for interactive debugging
    print(x)
    