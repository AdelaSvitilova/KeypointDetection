import numpy as np
import tensorflow as tf
from .base_trainer import BaseTrainer

class KerasTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset=None,
        loss_fn=None,
        metrics=None,
        batch_size=16,
        epochs=10,
        lr=0.001,
        val_every_epochs=1
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.loss_fn = loss_fn
        self.metrics = metrics or []
        self.batch_size = batch_size
        self.epochs = epochs
        self.val_every_epochs = val_every_epochs

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # dataset loader
    def _get_batch(self, dataset, start, end):
        images, keypoints = [], []
        for i in range(start, min(end, len(dataset))):
            img, kp = dataset[i]
            img = np.transpose(img, (1, 2, 0))       # C,H,W → H,W,C
            images.append(img.astype(np.float32))
            keypoints.append(kp.astype(np.float32))
        return np.array(images), np.array(keypoints)

    def train(self):
        print("Starting training...")

        for epoch in range(1, self.epochs + 1):
            print(f"\nEpoch {epoch}/{self.epochs}")

            indices = np.random.permutation(len(self.train_dataset))
            total_loss = 0.0

            for i in range(0, len(indices), self.batch_size):
                batch_ids = indices[i:i+self.batch_size]
                images, targets = self._get_batch(self.train_dataset,
                                                  i, i+self.batch_size)

                with tf.GradientTape() as tape:
                    preds = self.model(images, training=True)
                    loss = self.loss_fn(preds, targets)

                grads = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                total_loss += float(loss)

            print(f"Train loss: {total_loss:.4f}")

            if self.val_dataset and epoch % self.val_every_epochs == 0:
                self.validate(epoch)

    def validate(self, epoch):
        total_loss = 0.0

        for i in range(0, len(self.val_dataset), self.batch_size):
            images, targets = self._get_batch(self.val_dataset, i, i+self.batch_size)

            preds = self.model(images, training=False)
            loss = self.loss_fn(preds, targets)
            total_loss += float(loss)

        print(f"[Epoch {epoch}] Validation loss: {total_loss:.4f}")
