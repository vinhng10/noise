import torch
import onnx
from models import Model
from onnxconverter_common import float16
from onnxruntime.quantization import quantize_static, QuantType
from onnxruntime.quantization.calibrate import CalibrationDataReader


class DataReader(CalibrationDataReader):
    def __init__(self):
        self.dataset = iter(
            [
                {"input": (torch.rand((1, 1, 1, 48000)) - 0.5).detach().cpu().numpy()}
                for _ in range(128)
            ]
        )

    def get_next(self):
        return next(self.dataset, None)


if __name__ == "__main__":
    model = Model(
        in_channels=1,
        hidden_channels=32,
        out_channels=1,
        encoder_n_layers=3,
        d_model=128,
        nhead=4,
        dim_feedforward=512,
        num_layers=2,
        dropout=0.0,
        bias=False,
        mr_stft_lambda=0.5,
        fft_sizes=[512, 1024, 2048],
        hop_lengths=[50, 120, 240],
        win_lengths=[240, 600, 1200],
    )
    model.to_onnx(
        "fp32-model.onnx",
        torch.randn((1, 1, 1, 48000)),
        export_params=True,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
    # model = onnx.load("fp32-model.onnx")
    # model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    # onnx.save(model_fp16, "/workspace/noise/demo/src/app/model.onnx")

    quantize_static(
        "fp32-model.onnx",
        "/workspace/noise/demo/src/app/model.onnx",
        calibration_data_reader=DataReader(),
        quant_format="QDQ",
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
    )
