import torch
from models import *
from data import *

if __name__ == "__main__":
    n = 50

    model = VADSpectralV2.load_from_checkpoint(
        "./logs/spectralv2-2.0.0/checkpoints/epoch=192-val_loss=0.257.ckpt",
        map_location=torch.device("cpu"),
    )

    model = model.eval()

    torch.onnx.export(
        model,
        torch.randn((1, 1, 1, 160 * n)),
        "../demo/src/model.onnx",
        export_params=True,
        do_constant_folding=False,
        input_names=["input"],
        output_names=["output"],
    )

