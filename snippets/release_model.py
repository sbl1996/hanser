import os
import tempfile
from pathlib import Path

import typer

from hhutil.io import fmt_path, zip_files, rename, rm
from hhutil.hash import sha256


def list_checkpoint_files(ckpt_path):
    ckpt_path = fmt_path(ckpt_path)

    files = [
        ckpt_path.with_suffix(".index"),
        ckpt_path.with_suffix(".data-00000-of-00001")
    ]
    return files


def _release_model(ckpt_path, model_name, target_dir=None, delete=False):
    ckpt_path = fmt_path(ckpt_path)

    files = list_checkpoint_files(ckpt_path)

    for f in files:
        assert f.exists()

    if target_dir is None:
        target_dir = ckpt_path.parent
    else:
        target_dir = fmt_path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
    dst = target_dir / f"{model_name}.zip"
    zip_files(files, dst, deterministic=True)
    hash_suffix = sha256(dst)[:8]
    rename(dst, f"{model_name}-{hash_suffix}")

    if delete:
        for f in files: rm(f)

app = typer.Typer(add_completion=False)


@app.command()
def release_model(
    ckpt_path: Path = typer.Argument(..., help="Path to checkpoint file, i.e. ./resnet50/ckpt"),
    model_name: str = typer.Argument(..., help="Model name, i.e. resnetvd50"),
    target_dir: Path = typer.Argument(..., help="Target directory to save released model file"),
    convert: bool = typer.Option(True, "-c/-nc", help="Convert to model checkpoint first"),
):
    if convert:
        from hanser.models.utils import convert_checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            model_ckpt_path = os.path.join(tmpdir, "model")
            convert_checkpoint(ckpt_path, model_ckpt_path)
            _release_model(model_ckpt_path, model_name, target_dir)
    else:
        _release_model(ckpt_path, model_name, target_dir)


if __name__ == '__main__':
    app()

# python release_model.py "~/Downloads/290/ckpt" re_resnet_sp1 "~/Downloads"
