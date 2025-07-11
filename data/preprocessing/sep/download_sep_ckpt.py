from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="m-a-p/soundsation_sep_ckpts",
    local_dir="./soundsation_sep_ckpts",
    local_dir_use_symlinks=False,
    repo_type="model",
)
