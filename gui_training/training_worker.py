"""QThread worker for cellpose fine-tuning."""
import os
import re
import time
import shutil
import logging
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

log = logging.getLogger(__name__)


class CellposeLogHandler(logging.Handler):
    """Intercepts cellpose logger to extract epoch/loss info."""

    def __init__(self, callback):
        super().__init__()
        self.callback = callback
        self._epoch_re = re.compile(
            r"Epoch\s+(\d+).*?loss[=:]\s*([\d.]+)", re.IGNORECASE)

    def emit(self, record):
        msg = self.format(record)
        m = self._epoch_re.search(msg)
        if m:
            epoch = int(m.group(1))
            loss = float(m.group(2))
            self.callback(epoch, loss)


class TrainingWorker(QThread):
    """Fine-tune cellpose on provided image+mask pairs."""

    progress = pyqtSignal(str, int)
    epoch_done = pyqtSignal(int, float)   # epoch, loss
    log_event = pyqtSignal(str, str)
    finished = pyqtSignal(str)            # model path
    error = pyqtSignal(str)

    def __init__(self, images, masks, config):
        """
        Args:
            images: list of (H, W) uint8 arrays
            masks: list of (H, W) uint16 label arrays
            config: dict with base_model, model_name, n_epochs,
                    learning_rate, batch_size, weight_decay, sgd,
                    augment
        """
        super().__init__()
        self.images = images
        self.masks = masks
        self.config = config
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            from cellpose import models, train, io as cp_io

            cfg = self.config
            base_path = cfg.get("base_model", "")
            model_name = cfg.get("model_name", "cellpose_finetuned")
            n_epochs = cfg.get("n_epochs", 80)
            lr = cfg.get("learning_rate", 5e-6)
            bs = cfg.get("batch_size", 8)
            wd = cfg.get("weight_decay", 1e-5)
            sgd = cfg.get("sgd", False)

            self.log_event.emit("start",
                                f"Training {model_name}: "
                                f"{len(self.images)} pairs, "
                                f"{n_epochs} epochs")

            # Light augmentation if requested
            train_imgs = list(self.images)
            train_masks = list(self.masks)
            if cfg.get("augment", False):
                self.log_event.emit("info", "Applying augmentations")
                aug_imgs, aug_masks = self._augment(
                    train_imgs, train_masks)
                train_imgs.extend(aug_imgs)
                train_masks.extend(aug_masks)
                self.log_event.emit("info",
                                    f"Augmented: {len(train_imgs)} pairs")

            # Setup cellpose log interception
            cp_logger = logging.getLogger("cellpose")
            handler = CellposeLogHandler(
                lambda ep, loss: self.epoch_done.emit(ep, loss))
            cp_logger.addHandler(handler)

            try:
                if base_path and os.path.exists(base_path):
                    base = models.CellposeModel(
                        gpu=True, pretrained_model=base_path)
                else:
                    base = models.CellposeModel(gpu=True)

                save_dir = "data/models"
                train.train_seg(
                    base.net,
                    train_data=train_imgs,
                    train_labels=train_masks,
                    channels=[0, 0],
                    save_path=save_dir,
                    model_name=model_name,
                    n_epochs=n_epochs,
                    learning_rate=lr,
                    weight_decay=wd,
                    SGD=sgd,
                    batch_size=bs,
                    normalize=True,
                    min_train_masks=1,
                )
            finally:
                cp_logger.removeHandler(handler)

            # Handle cellpose model path quirk
            sub = os.path.join(save_dir, "models", model_name)
            target = os.path.join(save_dir, model_name)
            if os.path.exists(sub):
                if os.path.exists(target):
                    shutil.rmtree(target) if os.path.isdir(target) \
                        else os.remove(target)
                shutil.move(sub, target)
                try:
                    os.rmdir(os.path.join(save_dir, "models"))
                except OSError:
                    pass
            else:
                target = sub

            self.log_event.emit("done", f"Model saved: {target}")
            self.finished.emit(target)

        except Exception as e:
            log.exception("Training failed")
            self.error.emit(str(e))

    def _augment(self, images, masks):
        """Light augmentation: noise, gamma, hflip."""
        rng = np.random.default_rng(42)
        aug_imgs, aug_masks = [], []
        for img, msk in zip(images, masks):
            # Noise
            n = (img.astype(np.float32) +
                 rng.normal(0, 6, img.shape)).clip(0, 255).astype(np.uint8)
            aug_imgs.append(n)
            aug_masks.append(msk.copy())
            # Gamma
            g = float(rng.uniform(0.8, 1.25))
            ig = (255 * (img.astype(np.float32) / 255) ** g
                  ).clip(0, 255).astype(np.uint8)
            aug_imgs.append(ig)
            aug_masks.append(msk.copy())
            # Hflip
            aug_imgs.append(np.fliplr(img).copy())
            aug_masks.append(np.fliplr(msk).copy())
        return aug_imgs, aug_masks
