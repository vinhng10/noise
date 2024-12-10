import torch
import onnx
from models import *
from data import *

# from onnxconverter_common import float16
from onnxruntime.quantization import (
    quantize_static,
    quantize_dynamic,
    QuantType,
    QuantFormat,
)
from onnxruntime.quantization.calibrate import CalibrationDataReader
from onnxruntime.quantization.shape_inference import quant_pre_process


class DataReader(CalibrationDataReader):
    def __init__(self):
        self.data_module = VADNoiseDataModule(
            data_dir="data",
            sampling_rate=16000,
            length=24000,
            num_workers=1,
            batch_size=1,
        )
        self.data_module.prepare_data()
        self.data_module.setup("fit")
        self.dataset = iter(self.data_module.train_dataloader())

    def get_next(self):
        batch = next(self.dataset, None)
        if batch is None:
            return None
        batch = batch[0].detach().cpu().numpy()
        return {"input": batch}

    def rewind(self):
        self.dataset = iter(self.data_module.train_dataloader())


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

    # model = onnx.load("fp32-model.onnx")
    # model_fp16 = float16.convert_float_to_float16(model)
    # onnx.save(model_fp16, "../demo/src/model.onnx")

    # quantize_static(
    #     "fp32-model.onnx",
    #     "../demo/src/model.onnx",
    #     DataReader(),
    #     activation_type=QuantType.
    # )
