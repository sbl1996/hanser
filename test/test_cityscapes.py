from tqdm import tqdm

from PIL import Image
import numpy as np
from hhutil.io import fmt_path, eglob

c = np.bincount([], minlength=256)

d = fmt_path("/Users/hrvvi/Downloads/datasets/Cityscapes/gtFine_trainvaltest/gtFine/val")
images = list(eglob(d, "*/*_gtFine_labelIds.png"))

for img in tqdm(images):
    x = np.array(Image.open(img))
    c += np.bincount(x.flat, minlength=256)