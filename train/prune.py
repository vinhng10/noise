from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import torch
import torch_pruning as tp
import torch.nn.functional as F

from data import NoiseDataModule
from models import LightningMobileNetV1


def finetune(model, step):
    model.train()
    model.configure_optimizers = lambda: torch.optim.AdamW(
        model.parameters(), lr=1e-5, weight_decay=0.02
    )
    trainer = Trainer(
        max_epochs=1000,
        accelerator="auto",
        strategy="auto",
        devices="auto",
        callbacks=[
            ModelCheckpoint(
                dirpath="./logs/mobilenetv1-2.0.0/checkpoints",
                filename=f"pruned-{step}-" + "{epoch}-{val_loss:.3f}",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
            ),
            EarlyStopping(monitor="val_loss", mode="min", patience=10),
        ],
    )
    trainer.fit(model, datamodule=data_module)


data_module = NoiseDataModule(
    data_dir="./data",
    sampling_rate=16000,
    length=0,
    num_samples=5000,
    num_workers=4,
    batch_size=16,
)
data_module.setup(stage="fit")
val_dataloader = data_module.val_dataloader()

checkpoint = torch.load(
    "./logs/mobilenetv1-2.0.0/checkpoints/epoch=3907-train_loss=0.860.ckpt",
    map_location=torch.device("cpu"),
)
checkpoint["state_dict"] = {
    f"model.{k}": v for k, v in checkpoint["state_dict"].items()
}
checkpoint["hyper_parameters"].pop("_instantiator")
model = LightningMobileNetV1(**checkpoint["hyper_parameters"])
model.load_state_dict(state_dict=checkpoint["state_dict"])

# Importance criteria
noisy_waveforms, clean_waveforms, _ = next(iter(val_dataloader))
imp = tp.importance.TaylorImportance()

ignored_layers = [model.model.bottleneck_attention]

iterative_steps = 5  # progressive pruning
pruner = tp.pruner.MetaPruner(
    model,
    noisy_waveforms,
    importance=imp,
    iterative_steps=iterative_steps,
    pruning_ratio=0.5,
    ignored_layers=ignored_layers,
    global_pruning=True,
    isomorphic=True,
)

base_macs, base_nparams = tp.utils.count_ops_and_params(model, noisy_waveforms)
for i in range(iterative_steps):
    if isinstance(imp, tp.importance.TaylorImportance):
        # Taylor expansion requires gradients for importance estimation
        enhanced_waveforms = model.model._forward(noisy_waveforms)
        l1_loss = F.l1_loss(enhanced_waveforms, clean_waveforms)
        mrstft_loss = model.hparams.mr_stft_lambda * model.multi_resolution_stft_loss(
            enhanced_waveforms, clean_waveforms
        )
        loss = l1_loss + mrstft_loss
        loss.backward()
    pruner.step()
    macs, nparams = tp.utils.count_ops_and_params(model, noisy_waveforms)
    # finetune your model here
    finetune(model, step=iterative_steps)
