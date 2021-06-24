from hhutil.io import fmt_path, zip_files, rename
from hhutil.hash import sha256


def release_model(ckpt_path, model_prefix, target_dir=None):
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
    dst = target_dir / f"{model_prefix}.zip"
    zip_files(files, dst)
    hash_suffix = sha256(dst)[:8]
    rename(dst, f"{model_prefix}-{hash_suffix}")

ckpt_path = "/Users/hrvvi/Downloads/88/model"
model_prefix = "ppresnetvd50_nlb"
target_dir = "/Users/hrvvi/Downloads"
release_model(ckpt_path, model_prefix, target_dir)
