import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from typing import Any, Collection, Sequence, cast, Sized
import copy

class TrainProgress():
    def __init__(self):
        self.fold = None
        self.epoch = None
        self.loss = None
        self.best_loss = None

    def set_fold(self, fold: int):
        self.fold = fold

    def set_epoch(self, epoch: int):
        self.epoch = epoch
    
    def set_loss(self, loss: float, best_loss: float):
        self.loss = loss
        self.best_loss = best_loss

    def update(self, steps: int):
        pass

class TqdmTrainProgress(TrainProgress):
    def __init__(self, bar: Any):
        super().__init__()
        self.bar = bar
        self.msg = None
        self.fold = None
        self.epoch = None
        self.loss = None
        self.best_loss = None

    def set_message(self, msg: str):
        self.msg = msg
        self._update_bar()

    def set_fold(self, fold: int):
        self.fold = fold
        self._update_bar()

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self._update_bar()
    
    def set_loss(self, loss: float, best_loss: float):
        self.loss = loss
        self.best_loss = best_loss
        self._update_bar()
        
    def update(self, steps: int):
        if self.bar is not None:
            self.bar.update(steps)

    def _update_bar(self):
        desc = []
        if self.bar is not None:
            if self.msg is not None and self.msg != "":
                desc.append(self.msg)
            if self.fold is not None:
                desc.append(f"fold={self.fold+1}")
            if self.epoch is not None:
                desc.append(f"epoch={self.epoch+1}")
            if self.loss is not None:
                desc.append(f"loss={self.loss:.6f}")
            if self.best_loss is not None:
                desc.append(f"best={self.best_loss:.6f}")
        self.bar.set_description(", ".join(desc))

class Trainer:
    def __init__(self, model, optimizer, criterion, device: str|torch.device="cuda", criterion_kwargs={}):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.criterion_kwargs = criterion_kwargs

    def train_epoch_step(self, inputs):
        input = inputs[0].to(self.device)
        outputs = self.model(input)
        return input.shape[0], outputs

    def train_epoch_loss(self, inputs, outputs):
        expected = inputs[1].to(self.device)
        loss = self.criterion(outputs, expected, **self.criterion_kwargs)
        return loss

    def train_epoch_step_and_loss(self, inputs):
        step, outputs = self.train_epoch_step(inputs)
        return step, self.train_epoch_loss(inputs, outputs)

    def train_epoch(self, train_loader):
        self.model.train()
        for inputs in train_loader:
            step, loss = self.train_epoch_step_and_loss(inputs)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            yield step

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        total = 0

        with torch.no_grad():
            for inputs in val_loader:
                step, loss = self.train_epoch_step_and_loss(inputs)

                val_loss += loss.item() * step
                total += step

        avg_loss = val_loss / total
        return avg_loss

    def train(self, train_loader, validation_loader, num_epochs=100, patience=10, bar: TrainProgress|None=None):
        wait = 0

        best_val_loss = float("inf")
        best_state = None
        epoch = 0
        for epoch in range(num_epochs):
            for steps in self.train_epoch(train_loader):
                if bar: bar.update(steps)
            val_loss = self.validate(validation_loader)
            if bar: bar.set_epoch(epoch+1)
            if bar: bar.set_loss(val_loss, min(best_val_loss, val_loss))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        return best_state, best_val_loss, epoch + 1

def cross_validate(
            trainers: Sequence[Trainer], datasets: Sequence[Dataset], num_epochs=100,
            patience=10, batch_size=32, shuffle=True, seed=42, msg_info: str="",
            debug: bool=False
        ):

    folds = len(trainers)

    if folds != len(datasets):
        raise ValueError("Length of trainers and datasets must be the same.")

    if any(isinstance(d, Sized) == False for d in datasets):
        raise ValueError("All datasets must have a defined length.")

    num_samples_folds = [sum(len(cast(Sized, datasets[i])) for i in range(folds) if i != fold) for fold in range(folds)]
    num_samples_epoch_all_folds = sum(num_samples_folds)
    num_samples_total = num_epochs * num_samples_epoch_all_folds

    if debug: print(f"num_samples_epoch_all_folds = {num_samples_epoch_all_folds}")
    if debug: print(f"num_samples_total = {num_samples_total}")

    # --- Create a generator with fixed seed ---
    g = torch.Generator()
    g.manual_seed(seed)

    fold_losses = []
    fold_models = []
    fold_epochs = []
    try:
        from pre_sal_ii import progress
        bar = TqdmTrainProgress(progress(total=num_samples_total))
        bar.set_message(msg_info)
    except ImportError:
        bar = None

    for fold in range(folds):
        trainer = trainers[fold]
        
        train_subset = ConcatDataset([datasets[i] for i in range(folds) if i != fold])
        val_subset = datasets[fold]

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle, generator=g)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        if bar: bar.set_fold(fold)

        best_state, best_val_loss, epochs = trainer.train(train_loader, val_loader,
                                                          num_epochs=num_epochs,
                                                          patience=patience, bar=bar)

        if epochs < num_epochs:
            if debug: print(f"Early stopping at epoch {epochs} for fold {fold+1}")
            if bar: bar.update((num_epochs-epochs)*num_samples_folds[fold])

        trainers[fold].model.load_state_dict(best_state)
        fold_models.append(trainers[fold].model)
        fold_losses.append(best_val_loss)
        fold_epochs.append(epochs)
        if debug: print(f"Fold {fold+1}: best val loss = {best_val_loss:.4f}")

    if debug: print("Average validation loss:", sum(fold_losses)/len(fold_losses))
    return fold_models, fold_losses, fold_epochs