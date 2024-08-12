import torchaudio
from data import *
from models import *
from tqdm import tqdm

if __name__ == "__main__":
    sampling_rate = 16000

    model = Model.load_from_checkpoint(
        "./logs/model-1.0.5/checkpoints/epoch=479-train_loss=1.761.ckpt"
    ).eval()
    data = NoiseDataModule(
        data_dir="./data-debug",
        sampling_rate=sampling_rate,
        batch_size=1,
        num_workers=1,
    )
    data.prepare_data()
    data.setup(stage="fit")
    loader = data.train_dataloader()
    with torch.no_grad():
        for i, (noisy_waveforms, clean_waveforms) in tqdm(enumerate(loader)):
            enhanced_waveforms = model.forward(noisy_waveforms)
            torchaudio.save(
                f"./data-debug/enhanced/{i}.wav",
                enhanced_waveforms.squeeze(0),
                sampling_rate,
            )
