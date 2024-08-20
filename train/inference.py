import torchaudio
from data import *
from models import *
from tqdm import tqdm
from pathlib import Path

if __name__ == "__main__":
    sampling_rate = 16000

    model = Model.load_from_checkpoint(
        "./logs/model-1.0.0/checkpoints/epoch=3419-train_loss=1.692.ckpt"
    ).eval()
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
        for i, (noisy_waveforms, clean_waveforms, name) in tqdm(list(enumerate(loader))):
            enhanced_waveforms = model.forward(noisy_waveforms)
            torchaudio.save(
                f"./data-debug/enhanced/{name[0]}.wav",
                enhanced_waveforms[0, 0],
                sampling_rate,
            )
