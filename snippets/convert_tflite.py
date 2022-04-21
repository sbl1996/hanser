import importlib
import tempfile
from pathlib import Path

import numpy as np

import typer
from typing import Optional
from hhutil.io import fmt_path, mv


def import_model(import_name):
    module_name, model_name = import_name.rsplit('.', 1)
    model_fn = getattr(importlib.import_module(module_name), model_name)
    return model_fn()


def infer_tflite(tflite_model_path, x):
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(str(tflite_model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']

    interpreter.set_tensor(input_index, x)
    interpreter.invoke()
    y = interpreter.get_tensor(output_index)
    return y


def check_model(model, tflite_path, input_shape=(224, 224, 3)):
    import tensorflow as tf

    x = np.random.uniform(size=input_shape).astype(np.float32)

    xt = tf.convert_to_tensor(x)
    y1 = model(xt[None]).numpy()[0]

    y2 = infer_tflite(tflite_path, x[None])[0]
    np.testing.assert_allclose(y1, y2, rtol=1e-5, atol=1e-5)


def convert_checkpoint(ckpt_path, import_name, tflite_path, input_shape=(224, 224, 3), check=True, from_hub=False):
    import tensorflow as tf
    from hanser.models.utils import load_checkpoint, load_fc_from_checkpoint

    model = import_model(import_name)
    model.build((None, *input_shape))
    load_checkpoint(ckpt_path, model=model)
    if from_hub:
        load_fc_from_checkpoint(model, ckpt_path)

    x = np.random.uniform(size=(1, *input_shape)).astype(np.float32)
    x = tf.convert_to_tensor(x)
    model(x)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = []
    tflite_model = converter.convert()

    if check:
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            tmpfile.write(tflite_model)
            check_model(model, tmpfile.name, input_shape)
            mv(tmpfile.name, tflite_path)
    else:
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)


def convert_saved_model(saved_model_path, tflite_path, input_shape=(224, 224, 3), check=True):
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))
    converter.optimizations = []
    tflite_model = converter.convert()

    if check:
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            tmpfile.write(tflite_model)
            model = tf.saved_model.load(str(saved_model_path))
            check_model(model, tmpfile.name, input_shape=input_shape)
            mv(tmpfile.name, tflite_path)
    else:
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)


app = typer.Typer(add_completion=False)


@app.command("checkpoint")
def cmd_checkpoint(
    ckpt_path: Path = typer.Argument(..., help="Path to checkpoint, i.e. ./mobilenetv3_large/ckpt"),
    import_name: str = typer.Argument(..., help="Model name, i.e. hanser.models.imagenet.mobilenet_v3.mobilenet_v3_large"),
    tflite_path: Optional[Path] = typer.Option(None, help="Target directory to save tflite model file"),
    check: bool = typer.Option(True, "-c", help="Whether to do sanity check"),
    input_shape: str = typer.Option("224,224,3", "-i", help="Input shape, i.e. 224,224,3"),
    from_hub: bool = typer.Option(False, "-h", help="Whether from hub"),
):
    ckpt_path = fmt_path(ckpt_path)
    if tflite_path is None:
        tflite_path = ckpt_path.parent.parent / (ckpt_path.parent.stem + ".tflite")
    tflite_path = fmt_path(tflite_path)

    input_shape = tuple(map(int, input_shape.split(",")))
    convert_checkpoint(ckpt_path, import_name, tflite_path, input_shape=input_shape, check=check, from_hub=from_hub)


@app.command("saved_model")
def cmd_saved_model(
    saved_model_path: Path = typer.Argument(..., help="Path to saved model, i.e. ./mobilenetv3_large"),
    tflite_path: Optional[Path] = typer.Option(None, help="Target directory to save tflite model file"),
    check: bool = typer.Option(True, "-c", help="Whether to do sanity check"),
    input_shape: str = typer.Option("224,224,3", "-i", help="Input shape, i.e. 224,224,3"),
):
    saved_model_path = fmt_path(saved_model_path)
    if tflite_path is None:
        tflite_path = saved_model_path.parent / (saved_model_path.stem + ".tflite")
    tflite_path = fmt_path(tflite_path)

    input_shape = tuple(map(int, input_shape.split(",")))
    convert_saved_model(saved_model_path, tflite_path, input_shape=input_shape, check=check)


if __name__ == '__main__':
    app()

# python snippets/convert_tflite.py saved_model "~/Downloads/imagenet_mobilenet_v3_large_100_224_classification_5" "~/Downloads/lite_mobilenet_v3_large.tflite"
# python snippets/convert_tflite.py checkpoint "~/Downloads/ckpt_mb3_100" hanser.models.imagenet.mobilenet_v3.mobilenet_v3_large
