# Model file has three formats:
# 1. Train checkpoint, produced by training, including model, optimizer, epoch
# 2. Model checkpoint, produced by convert_checkpoint from format 1, including only model
# 3. Model zip file, produced by this script from format 2, for release on GitHub

from pathlib import Path
from typing import Optional

import typer

from hhutil.io import fmt_path, zip_files, rename
from hhutil.hash import sha256


def _release_model(ckpt_path, model_name, target_dir=None):
    ckpt_path = fmt_path(ckpt_path)

    files = [
        ckpt_path.with_suffix(".index"),
        ckpt_path.with_suffix(".data-00000-of-00001")
    ]

    for f in files:
        assert f.exists()

    if target_dir is None:
        target_dir = ckpt_path.parent
    else:
        target_dir = fmt_path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
    dst = target_dir / f"{model_name}.zip"
    zip_files(files, dst)
    hash_suffix = sha256(dst)[:8]
    rename(dst, f"{model_name}-{hash_suffix}")

app = typer.Typer(add_completion=False)


@app.command()
def release_model(
    ckpt_path: Path = typer.Argument(..., help="Path to checkpoint file"),
    model_name: str = typer.Argument(..., help="Model name, i.e. resnetvd50"),
    target_dir: Path = typer.Argument(..., help="Target directory to save released model file"),
):
    _release_model(ckpt_path, model_name, target_dir)


if __name__ == '__main__':
    app()


# python release_model.py "/Users/hrvvi/Downloads/88/model" ppresnetvd50_nlb "/Users/hrvvi/Downloads"
# ckpt_path = "/Users/hrvvi/Downloads/88/model"
# model_name = "ppresnetvd50_nlb"
# target_dir = "/Users/hrvvi/Downloads"
# release_model(ckpt_path, model_name, target_dir)
