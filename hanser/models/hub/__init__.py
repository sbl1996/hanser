import os
import sys
import re

from urllib.parse import urlparse
from pathlib import Path

from hhutil.io import fmt_path, download_file, unzip, eglob
from hhutil.hash import sha256

_model_path = {
    "resnetvd50": "https://github.com/sbl1996/hanser/releases/download/0.1.1/resnetvd50-8df2a5e4.zip",
    "resnetvd50_nlb": "https://github.com/sbl1996/hanser/releases/download/0.1.1/resnetvd50_nlb-f273a9d1.zip",
    "res2netvd50": "https://github.com/sbl1996/hanser/releases/download/0.1.2/res2netvd50-f9294020.zip",
    "res2netvd50_nlb": "https://github.com/sbl1996/hanser/releases/download/0.1.2/res2netvd50_nlb-a4c28fd9.zip",
    "ppresnet50": "https://github.com/sbl1996/hanser/releases/download/0.1.2/ppresnet-14c86ae2.zip",
    "ppresnet50_nlb": "https://github.com/sbl1996/hanser/releases/download/0.1.2/ppresnet50_nlb-167cdde6.zip",
    "legacy_res2netvd50": "https://github.com/sbl1996/hanser/releases/download/0.1.1/res2netvd50-f7ab3d99.zip",
    "legacy_res2netvd50_nlb": "https://github.com/sbl1996/hanser/releases/download/0.1.1/res2netvd50_nlb-8958f30f.zip",
    "legacy_ppresnet50": "https://github.com/sbl1996/hanser/releases/download/0.1.1/ppresnet50-7c0adb0b.zip",
    "legacy_ppresnet50_nlb": "https://github.com/sbl1996/hanser/releases/download/0.1.1/ppresnetvd50_nlb-c77c5f1c.zip",
}

HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')
ENV_HANSER_HOME = 'HANSER_HOME'
DEFAULT_CACHE_DIR = '~/.cache'

def _get_hanser_home():
    default_hanser_home = fmt_path(DEFAULT_CACHE_DIR) / "hanser"
    return fmt_path(os.getenv(ENV_HANSER_HOME, default_hanser_home))


def get_dir():
    r"""
    Get the Torch Hub cache directory used for storing downloaded models & weights.

    If :func:`~torch.hub.set_dir` is not called, default path is ``$TORCH_HOME/hub`` where
    environment variable ``$TORCH_HOME`` defaults to ``$XDG_CACHE_HOME/torch``.
    ``$XDG_CACHE_HOME`` follows the X Design Group specification of the Linux
    filesystem layout, with a default value ``~/.cache`` if the environment
    variable is not set.
    """
    # Issue warning to move data if old env is set

    return os.path.join(_get_hanser_home(), 'hub')


def checkpoint_exists(ckpt_dir: Path):
    ckpt_dir = fmt_path(ckpt_dir)
    if not ckpt_dir.exists():
        return False
    indices = list(eglob(ckpt_dir, "*.index"))
    if not indices:
        return False
    ckpt_name = indices[0].stem
    data_files = list(eglob(ckpt_dir, f"{ckpt_name}.data-*"))
    if not data_files:
        return False
    return True


def load_model_from_url(url, model_dir=None, check_hash=False):
    r"""Loads the Torch serialized object at the given URL.

    If downloaded file is a zip file, it will be automatically
    decompressed.

    If the object is already present in `model_dir`, it's deserialized and
    returned.
    The default value of `model_dir` is ``<hub_dir>/checkpoints`` where
    `hub_dir` is the directory returned by :func:`~torch.hub.get_dir`.

    Args:
        url (string): URL of the object to download
        model_dir (string, optional): directory in which to save the object
        check_hash(bool, optional): If True, the filename part of the URL should follow the naming convention
            ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
            digits of the SHA256 hash of the contents of the file. The hash is used to
            ensure unique names and to verify the contents of the file.
            Default: False

    Example:
        >>> state_dict = load_model_from_url('https://github.com/sbl1996/hanser/releases/download/0.1.1/resnetvd50_nlb-f273a9d1.zip')

    """
    # Issue warning to move data if old env is set

    if model_dir is None:
        hub_dir = _get_hanser_home() / 'hub'
        model_dir = hub_dir / 'checkpoints'

    model_dir.mkdir(parents=True, exist_ok=True)
    path = Path(urlparse(url).path)
    file_name = path.name
    stem = path.stem
    ckpt_dir = model_dir / stem
    if checkpoint_exists(ckpt_dir):
        return str(ckpt_dir / "model")

    cached_file = model_dir / file_name
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        if check_hash:
            r = HASH_REGEX.search(file_name)  # r is Optional[Match[str]]
            hash_prefix = r.group(1) if r else None
        download_file(url, cached_file)
        if hash_prefix is not None:
            digest = sha256(cached_file)
            if digest[:len(hash_prefix)] != hash_prefix:
                raise RuntimeError('invalid hash value (expected "{}", got "{}")'
                                   .format(hash_prefix, digest))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    unzip(cached_file, ckpt_dir)
    return str(ckpt_dir / "model")


def is_url(s):
    if not (s.startswith("http://") or s.startswith("https://")):
        return False
    result = urlparse(s)
    return all([result.scheme, result.netloc, result.path])


def load_model_from_hub(name_or_url, model_dir=None, check_hash=False):
    if not is_url(name_or_url):
        if not name_or_url in _model_path:
            raise ValueError("No model named %s" % name_or_url)
        url = _model_path[name_or_url]
    else:
        url = name_or_url
    return load_model_from_url(url, model_dir, check_hash)