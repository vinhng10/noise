import torch
import onnx
from models import *
from data import NoiseDataModule
from onnxconverter_common import float16
from onnxruntime.quantization import quantize_static, QuantType
from onnxruntime.quantization.calibrate import CalibrationDataReader
from onnxruntime.quantization.shape_inference import quant_pre_process


class DataReader(CalibrationDataReader):
    def __init__(self):
        self.data_module = NoiseDataModule(
            data_dir="data-debug",
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
    model = Model(
        in_channels=1,
        hidden_channels=64,
        out_channels=1,
        encoder_n_layers=4,
        nhead=4,
        dim_feedforward=512,
        num_layers=2,
        dropout=0.0,
        bias=True,
        mr_stft_lambda=0.5,
        fft_sizes=[512, 1024, 2048],
        hop_lengths=[50, 120, 240],
        win_lengths=[240, 600, 1200],
    )
    # model = Model.load_from_checkpoint(
    #     "./logs/model-1.0.0/checkpoints/epoch=63-train_loss=0.736.ckpt"
    # ).eval()
    model.to_onnx(
        "/workspaces/noise/demo/src/app/model.onnx",
        torch.randn((1, 1, 1, 4000)),
        export_params=True,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
    # model = onnx.load("fp32-model.onnx")
    # model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    # onnx.save(model_fp16, "/workspaces/noise/demo/src/app/model.onnx")

    # quant_pre_process(
    #     input_model="fp32-model.onnx",
    #     output_model_path="prep-model.onnx"
    # )

    # quantize_static(
    #     "fp32-model.onnx",
    #     "/workspaces/noise/demo/src/app/model.onnx",
    #     calibration_data_reader=DataReader(),
    #     quant_format="QDQ",
    #     activation_type=QuantType.QInt8,
    #     weight_type=QuantType.QInt8,
    #     extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
    # )
