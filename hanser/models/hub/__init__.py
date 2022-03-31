import os
import sys
import re

from urllib.parse import urlparse
from pathlib import Path

from hhutil.io import fmt_path, download_file, unzip, eglob, download_github_private_assert
from hhutil.hash import sha256

_model_path = {
    "resnetvd50": "https://github.com/sbl1996/hanser/releases/download/0.1.1/resnetvd50-8df2a5e4.zip",
    "resnetvd50_nlb": "https://github.com/sbl1996/hanser/releases/download/0.1.1/resnetvd50_nlb-f273a9d1.zip",
    "res2netvd50": "https://github.com/sbl1996/hanser/releases/download/0.1.2/res2netvd50-f9294020.zip",
    "res2netvd50_nlb": "https://github.com/sbl1996/hanser/releases/download/0.1.2/res2netvd50_nlb-a4c28fd9.zip",
    "ppresnet50": "https://github.com/sbl1996/hanser/releases/download/0.1.2/ppresnet-14c86ae2.zip",
    "ppresnet50_nlb": "https://github.com/sbl1996/hanser/releases/download/0.1.2/ppresnet50_nlb-167cdde6.zip",
    "ppresnet50_g2": "https://github.com/sbl1996/hanser/releases/download/0.1.2/ppresnet50_g2-d62fce29.zip",
    "ppresnet50_g2_aug": "https://github.com/sbl1996/hanser/releases/download/0.1.2/ppresnet50_g2_aug-577b470c.zip",
    "legacy_res2netvd50": "https://github.com/sbl1996/hanser/releases/download/0.1.1/res2netvd50-f7ab3d99.zip",
    "legacy_res2netvd50_nlb": "https://github.com/sbl1996/hanser/releases/download/0.1.1/res2netvd50_nlb-8958f30f.zip",
    "legacy_ppresnet50": "https://github.com/sbl1996/hanser/releases/download/0.1.1/ppresnet50-7c0adb0b.zip",
    "legacy_ppresnet50_nlb": "https://github.com/sbl1996/hanser/releases/download/0.1.1/ppresnetvd50_nlb-c77c5f1c.zip",
    "re_resnet_s": "https://github.com/sbl1996/hanser/releases/download/0.1.3/re_resnet_s-5001074b.zip",
    "re_resnet_s_aug": "https://github.com/sbl1996/hanser/releases/download/0.1.3/re_resnet_s_aug-ed44e04e.zip",
    "re_resnet_s_aug1": "https://github.com/sbl1996/hanser/releases/download/0.1.3/re_resnet_s_aug1-f9259322.zip",
    "resnetvd50_nls": "https://github.com/sbl1996/hanser/releases/download/0.1.4/resnetvd50_nls-1d7559da.zip",
    "resnet50_nls": "https://github.com/sbl1996/hanser/releases/download/0.1.4/resnet50_nls-028184dc.zip",
    "resnet50_dra": "https://github.com/sbl1996/hanser/releases/download/0.1.4/resnet50_dra-ff7b0f8b.zip",
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


def _get_default_model_dir():
    hub_dir = _get_hanser_home() / 'hub'
    model_dir = hub_dir / 'checkpoints'
    return model_dir


def load_model_from_url_or_path(url_or_path, model_dir=None, check_hash=False):
    r"""Loads the Torch serialized object at the given URL.

    If downloaded file is a zip file, it will be automatically
    decompressed.

    If the object is already present in `model_dir`, it's deserialized and
    returned.
    The default value of `model_dir` is ``<hub_dir>/checkpoints`` where
    `hub_dir` is the directory returned by :func:`~torch.hub.get_dir`.

    Args:
        url_or_path (string or Path): URL of the object to download, or path to the model zip file
        model_dir (string, optional): directory in which to save the object
        check_hash(bool, optional): If True, the filename part of the URL should follow the naming convention
            ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
            digits of the SHA256 hash of the contents of the file. The hash is used to
            ensure unique names and to verify the contents of the file.
            Default: False

    Example:
        >>> state_dict = load_model_from_url_or_path('https://github.com/sbl1996/hanser/releases/download/0.1.1/resnetvd50_nlb-f273a9d1.zip')

    """

    if model_dir is None:
        model_dir = _get_default_model_dir()

    model_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(url_or_path, Path) or not is_url(url_or_path):
        url = None
        path = fmt_path(url_or_path)
        if not path.exists():
            raise FileNotFoundError("Not found model zip file in %s" % path)
        if not path.suffix == '.zip':
            raise AttributeError("Only support model zip file")
    else:
        url = url_or_path
        path = Path(urlparse(url).path)

    file_name = path.name
    stem = path.stem
    ckpt_dir = model_dir / stem
    if checkpoint_exists(ckpt_dir):
        return str(ckpt_dir / "model")

    if isinstance(url_or_path, Path) or not is_url(url_or_path):
        cached_file = path
    else:
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


def load_model_from_hub(name_or_url_or_path: str, model_dir=None, check_hash=False, github_access_token=None):

    if isinstance(name_or_url_or_path, Path):
        path = name_or_url_or_path
        return load_model_from_url_or_path(path, model_dir, check_hash)

    if is_url(name_or_url_or_path):
        if github_access_token is not None:
            url = name_or_url_or_path
            if model_dir is None:
                model_dir = _get_default_model_dir()
            else:
                model_dir = fmt_path(model_dir)
            model_dir.mkdir(parents=True, exist_ok=True)

            stem = Path(urlparse(url).path).stem
            ckpt_dir = model_dir / stem
            if checkpoint_exists(ckpt_dir):
                return str(ckpt_dir / "model")

            path = download_github_private_assert(url, model_dir, github_access_token)
            sys.stderr.write('Downloading: "{}" to {}\n'.format(url, path))
            url_or_path = path
        else:
            url_or_path = name_or_url_or_path
    else:
        name_or_path = name_or_url_or_path
        if name_or_path.endswith(".zip"):
            url_or_path = name_or_path
        else:
            name = name_or_path
            if not name in _model_path:
                raise ValueError("No model named %s" % name)
            url_or_path = _model_path[name]
    return load_model_from_url_or_path(url_or_path, model_dir, check_hash)


def model_registered(name):
    return name in _model_path