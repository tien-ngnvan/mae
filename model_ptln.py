from lightning.pytorch import LightningModule
import torch
from typing import Iterable, Optional, Any
# import util.lr_sched as lr_sched
from util.lr_sched import CustomCosineWarmupScheduler
from torchmetrics import MeanMetric

# data_loader: Iterable, device: torch.device, epoch: int
# log_writer=None
class LitMAE(LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        lr,  min_lr, blr,
        weight_decay, warmup_epochs, epochs, 
        sync_dist, devices, accumulate_grad_batches,
        mask_mode_dict=None, list_mask_mode=None,  
        loss_scaler=None,
    ):
        super().__init__()
        self.model = model
        self.loss_scaler = loss_scaler
        self.lr = lr
        self.min_lr = min_lr
        self.blr = blr,
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.mask_mode_dict = mask_mode_dict
        self.list_mask_mode = list_mask_mode
        self.index = 0
        self.mask_mode = self.list_mask_mode[self.index]
        self.sync_dist = sync_dist
        self.mean_valid_loss = MeanMetric()
        self.devices = devices
        self.accumulate_grad_batches = accumulate_grad_batches
        print(f"*** Use strategy {self.mask_mode} ***")
        
    # def setup(self, stage):
    #     if stage == 'fit':
    #         total_devices = len(self.devices) * 1 # self.hparams.n_nodes
    #         train_batches = len(self.train_dataloader()) // total_devices
    #         self.train_steps = (self.epochs * train_batches) // self.accumulate_grad_batches
    
    # def num_training_steps(self) -> int:
    #     """Total training steps inferred from datamodule and devices."""
    #     dataset = self.train_dataloader()
    #     if self.trainer.max_steps:
    #         return self.trainer.max_steps

    #     dataset_size = (
    #         self.trainer.limit_train_batches
    #         if self.trainer.limit_train_batches != 0
    #         else len(dataset)
    #     )

    #     num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
    #     if self.trainer.tpu_cores:
    #         num_devices = max(num_devices, self.trainer.tpu_cores)

    #     effective_batch_size = dataset.batch_size * self.trainer.accumulate_grad_batches * num_devices
    #     return (dataset_size // effective_batch_size) * self.trainer.max_epochs
    
    # def setup(self, stage):
    #     if stage == 'fit':
    #         total_devices = len(self.devices) * 1 # self.n_nodes
    #         # train_batches = len(self.train_dataloader()) // total_devices
    #         train_batches = len(self.trainer._data_connector._train_dataloader_source.dataloader()) // total_devices
    #         self.train_steps = (self.epochs * train_batches) // self.accumulate_grad_batches

    def forward(self, pixel_values, pixel_values_mask, mask_mode):
        loss, _, _ = self.model(pixel_values, pixel_values_mask, mask_mode=mask_mode)
        return loss
    
    def training_step(self, batch, batch_idx):
        # pixel_values_mask = batch['pixel_values_mask'].to(self.device, non_blocking=True)
        # pixel_values = batch['pixel_values'].to(self.device, non_blocking=True)
        pixel_values_mask = batch['pixel_values_mask']
        pixel_values = batch['pixel_values']
        
        # TODO: check mask_mode
        
        loss = self(pixel_values, pixel_values_mask, self.mask_mode)
        
        self.log('train/loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    def on_train_epoch_end(self):
        if self.index < len(self.list_mask_mode) - 1 and self.current_epoch + 1 > self.mask_mode_dict[self.list_mask_mode[self.index]]:
            self.index += 1
            self.mask_mode = self.list_mask_mode[self.index]
        print(f"*** Change mask strategy to {self.mask_mode} ***")
    
    def validation_step(self, batch, batch_ids):
        pixel_values_mask = batch['pixel_values_mask']
        pixel_values = batch['pixel_values']
        
        # TODO: check mask_model
        # mask_mode = None
        
        with torch.no_grad():
            loss = self(pixel_values, pixel_values_mask, self.mask_mode)
        
        self.mean_valid_loss.update(loss, weight=pixel_values.shape[0])
        
        return loss
        
    def on_validation_epoch_end(self):
        
        self.log("valid/loss", self.mean_valid_loss.compute(), prog_bar=True, sync_dist=self.sync_dist, logger=True)
        
        self.mean_valid_loss.reset()
        
    def test_step(self, batch, batch_ids):
        pixel_values_mask = batch['pixel_values_mask']
        pixel_values = batch['pixel_values']
        
        # TODO: check mask_model
        # mask_mode = None
        
        with torch.no_grad():
            loss = self(pixel_values, pixel_values_mask, self.mask_mode)
        
        self.mean_valid_loss.update(loss, weight=pixel_values.shape[0])
        
        return loss
    
    def on_test_epoch_end(self):
        
        self.log("test/loss", self.mean_valid_loss.compute(), prog_bar=True, sync_dist=self.sync_dist, logger=True)
        
        self.mean_valid_loss.reset()
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.training_steps = self.trainer.estimated_stepping_batches
        scheduler = CustomCosineWarmupScheduler(
            optimizer,
            warmup_epochs=self.warmup_epochs,
            total_epochs=self.epochs,
            min_lr=self.min_lr,
            base_lr=self.blr,
            total_steps=self.training_steps
        )
        # return [optimizer]
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "monitor": None
        }
        
        # return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        
    def save_checkpoint(self, file_path, weights_only:bool=False, storage_options:Optional[Any] = None) -> None:
        checkpoint = self._checkpoint_connector.dump_checkpoint(weights_only)
        self.strategy.save_checkpoint(checkpoint, file_path, storage_options=storage_options)
        self.strategy.barrier("Train.save_checkpoint")
        
        