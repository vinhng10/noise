import torchaudio
from data import *
from models import *
from tqdm import tqdm
from pathlib import Path

if __name__ == "__main__":
    sampling_rate = 16000

    checkpoint = torch.load(
        "./logs/mobilenetv1-2.0.0/checkpoints/epoch=829-train_loss=1.064.ckpt",
        map_location=torch.device("cpu"),
    )
    checkpoint["state_dict"] = {
        f"model.{k}": v for k, v in checkpoint["state_dict"].items()
    }
    checkpoint["hyper_parameters"].pop("_instantiator")
    model = LightningMobileNetV1(
        **checkpoint["hyper_parameters"],
    )
    model.load_state_dict(state_dict=checkpoint["state_dict"])

    data = NoiseDataModule(
        data_dir="./data-debug",
        sampling_rate=sampling_rate,
        length=0,
        batch_size=1,
        num_workers=1,
    )
    data.prepare_data()
    data.setup(stage="fit")
    loader = data.train_dataloader()
    Path("./data-debug/enhanced/").mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for i, (noisy_waveforms, clean_waveforms, name) in tqdm(
            list(enumerate(loader))
        ):
            enhanced_waveforms = model.model._forward(noisy_waveforms)
            torchaudio.save(
                f"./data-debug/enhanced/{name[0]}.wav",
                enhanced_waveforms[0, 0],
                sampling_rate,
            )
