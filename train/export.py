import torch
import onnx
from models import *
from data import NoiseDataModule
from onnxconverter_common import float16
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
        self.data_module = NoiseDataModule(
            data_dir="data-debug",
            sampling_rate=48000,
            length=0,
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
        return {"input": batch[0].detach().cpu().numpy()}

    def rewind(self):
        self.dataset = iter(self.data_module.train_dataloader())


if __name__ == "__main__":
    # checkpoint = torch.load(
    #     "./logs/mobilenetv1-2.0.0/checkpoints/epoch=829-train_loss=1.064.ckpt",
    #     map_location=torch.device("cpu"),
    # )
    # checkpoint["state_dict"] = {
    #     f"model.{k}": v for k, v in checkpoint["state_dict"].items()
    # }
    # checkpoint["hyper_parameters"].pop("_instantiator")
    # model = LightningMobileNetV1(
    #     **checkpoint["hyper_parameters"],
    # )
    # model.load_state_dict(state_dict=checkpoint["state_dict"])

    model = MobileNetV1(
        in_channels=1,
        hidden_channels=32,
        max_channels=256,
        out_channels=1,
        kernel_size=3,
        stride=2,
        padding=0,
        encoder_n_layers=4,
        nhead=8,
        num_layers=1,
        dropout=0.0,
        bias=False,
        src_sampling_rate=48000,
        tgt_sampling_rate=16000,
    )
    model = model.eval()

    torch.onnx.export(
        model,
        torch.randn((1, 1, 1, 48000)),
        "/workspaces/noise/demo/src/model.onnx",
        export_params=True,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )

    # quantize_dynamic(
    #     "fp32-model.onnx",
    #     "/workspaces/noise/demo/src/model.onnx",
    # )
