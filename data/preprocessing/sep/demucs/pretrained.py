import logging
from pathlib import Path
import typing as tp

# For fatal error handling in production code
# from dora.log import fatal

import logging

from diffq import DiffQuantizer
import torch.hub

from .model import Demucs
from .tasnet_v2 import ConvTasNet
from .utils import set_state

from .hdemucs import HDemucs
from .repo import RemoteRepo, LocalRepo, ModelOnlyRepo, BagOnlyRepo, AnyModelRepo, ModelLoadingError  # noqa

logger = logging.getLogger(__name__)
# Base URL for downloading MDX models
ROOT_URL = "https://dl.fbaipublicfiles.com/demucs/mdx_final/"
REMOTE_ROOT = Path(__file__).parent / 'remote'

# Audio source types that can be separated
SOURCES = ["drums", "bass", "other", "vocals"]


def demucs_unittest():
    """Create a minimal HDemucs model for testing purposes."""
    model = HDemucs(channels=4, sources=SOURCES)
    return model


def add_model_flags(parser):
    """Add model selection arguments to the command line parser."""
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("-s", "--sig", help="Locally trained XP signature.")
    group.add_argument("-n", "--name", default="mdx_extra_q",
                       help="Pretrained model name or signature. Default is mdx_extra_q.")
    parser.add_argument("--repo", type=Path,
                        help="Folder containing all pre-trained models for use with -n.")


def _parse_remote_files(remote_file_list) -> tp.Dict[str, str]:
    """Parse remote file listing to extract model URLs.
    
    Builds a dictionary mapping model signatures to their download URLs.
    """
    root: str = ''
    models: tp.Dict[str, str] = {}
    for line in remote_file_list.read_text().split('\n'):
        line = line.strip()
        if line.startswith('#'):
            continue
        elif line.startswith('root:'):
            root = line.split(':', 1)[1].strip()
        else:
            sig = line.split('-', 1)[0]
            assert sig not in models
            models[sig] = ROOT_URL + root + line
    return models


def get_model(name: str,
              repo: tp.Optional[Path] = None):
    """Retrieve a model by name from remote or local repository.
    
    Args:
        name: Model name or signature identifier
        repo: Optional path to local model repository
        
    Returns:
        The loaded model in evaluation mode
    """
    if name == 'demucs_unittest':
        return demucs_unittest()
    model_repo: ModelOnlyRepo
    if repo is None:
        # Use remote repository when no local path provided
        models = _parse_remote_files(REMOTE_ROOT / 'files.txt')
        model_repo = RemoteRepo(models)
        bag_repo = BagOnlyRepo(REMOTE_ROOT, model_repo)
    else:
        # Use local repository at specified path
        if not repo.is_dir():
            fatal(f"{repo} must exist and be a directory.")
        model_repo = LocalRepo(repo)
        bag_repo = BagOnlyRepo(repo, model_repo)
    any_repo = AnyModelRepo(model_repo, bag_repo)
    model = any_repo.get_model(name)
    model.eval()
    return model


def get_model_from_args(args):
    """Load model based on command line arguments."""
    return get_model(name=args.name, repo=args.repo)


logger = logging.getLogger(__name__)
# Root URL for v3.0 pre-trained models
ROOT = "https://dl.fbaipublicfiles.com/demucs/v3.0/"

# Mapping of model names to their signatures for v3.0 models
PRETRAINED
