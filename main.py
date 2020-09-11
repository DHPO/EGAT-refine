from trainer.amlsim import EGAT_trainer
from trainer.citation import Citation_trainer
import pytorch_lightning as pl
import yaml
import os
from logger import WandbLogger


with open("config.yml", "r") as f:
    hparams = yaml.load(f, Loader=yaml.FullLoader)

Trainer = EGAT_trainer if hparams["net"] == "amlsim" else Citation_trainer

wandb_logger = WandbLogger(project="amlsim-batch", config=hparams)

trainer = pl.Trainer(benchmark=True, early_stop_callback=pl.callbacks.EarlyStopping("val_loss", patience=200, strict=True), gpus=[0], max_epochs=1000, logger=wandb_logger)

# model = MLPTrainer(hparams["mlp"])
model = Trainer(wandb_logger.config)
wandb_logger.watch(model, log="all")

trainer.fit(model)

weight_path = trainer.weights_save_path
filename = os.listdir(weight_path)[0]
model = Trainer.load_from_checkpoint(os.path.join(weight_path, filename))
print("Load from {}\n".format(os.path.join(weight_path, filename)))

trainer.test(model)
wandb_logger.log_metrics(trainer.callback_metrics)
