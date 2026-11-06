"""Reproducible clean/noisy experiments using an explicit grouped manifest."""
import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .audio import SAMPLE_RATE, add_noise, image, load_audio, reconstruct, stft
from .models import Autoencoder, CNN, Discriminator, Generator, VGG


class AudioDataset(Dataset):
    def __init__(self, rows, labels, snr=None, augment=False, seed=42):
        self.rows, self.labels = rows, labels
        self.snr, self.augment, self.seed = snr, augment, seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        wave = load_audio(row["path"])
        if self.augment and random.random() < 0.5:
            wave = add_noise(wave, random.choice([30, 20, 10, 0, -5]))
        elif self.snr is not None:
            wave = add_noise(wave, self.snr, torch.Generator().manual_seed(self.seed + index))
        return image(wave), self.labels[row["label"]]


def read_manifest(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"path", "label", "split", "group"} <= rows[0].keys():
        raise ValueError("Manifest requires path,label,split,group and at least one row")
    groups, files = {}, set()
    for row in rows:
        if row["split"] not in {"train", "val", "test"} or not row["group"]:
            raise ValueError("Invalid split or empty recording group")
        row["path"] = str((path.parent / row["path"]).resolve())
        if not Path(row["path"]).is_file() or row["path"] in files:
            raise ValueError(f"Missing or duplicate recording: {row['path']}")
        files.add(row["path"])
        if row["group"] in groups and groups[row["group"]] != row["split"]:
            raise ValueError("Recording group leaks across splits")
        groups[row["group"]] = row["split"]
    splits = {split: [r for r in rows if r["split"] == split] for split in ("train", "val", "test")}
    if not all(splits.values()):
        raise ValueError("All three splits must be nonempty")
    labels = {name: i for i, name in enumerate(sorted({r["label"] for r in splits["train"]}))}
    if any(r["label"] not in labels for r in rows):
        raise ValueError("Validation/test contains a class absent from training")
    return splits, labels


def loader(rows, labels, args, shuffle=False, **kwargs):
    return DataLoader(AudioDataset(rows, labels, seed=args.seed, **kwargs), batch_size=args.batch_size, shuffle=shuffle)


def fit(model, train, val, args, autoencoder=False, generator=None):
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss() if autoencoder else nn.CrossEntropyLoss()
    best, state = float("inf"), None
    for epoch in range(args.epochs):
        model.train()
        for x, y in train:
            x, y = x.to(args.device), y.to(args.device)
            if generator is not None:
                with torch.no_grad():
                    synthetic = generator(torch.randn(len(y), generator.latent, device=args.device), y)
                x, y = torch.cat([x, synthetic]), torch.cat([y, y])
            optimizer.zero_grad()
            loss = criterion(model(x), x if autoencoder else y)
            loss.backward()
            optimizer.step()
        model.eval()
        total = 0.0
        with torch.no_grad():
            for x, y in val:
                x, y = x.to(args.device), y.to(args.device)
                total += criterion(model(x), x if autoencoder else y).item() * len(x)
        total /= len(val.dataset)
        print(f"epoch={epoch + 1} validation_loss={total:.5f}", flush=True)
        if total < best:
            best, state = total, copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    return model.eval()


def train_gan(train, classes, args):
    generator, discriminator = Generator(classes).to(args.device), Discriminator(classes).to(args.device)
    go = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    do = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(args.gan_epochs):
        for x, y in train:
            x, y = x.to(args.device), y.to(args.device)
            fake = generator(torch.randn(len(y), generator.latent, device=args.device), y)
            do.zero_grad()
            loss = criterion(discriminator(x, y), torch.ones(len(y), device=args.device))
            loss += criterion(discriminator(fake.detach(), y), torch.zeros(len(y), device=args.device))
            loss.backward()
            do.step()
            for p in discriminator.parameters():
                p.requires_grad_(False)
            go.zero_grad()
            criterion(discriminator(fake, y), torch.ones(len(y), device=args.device)).backward()
            go.step()
            for p in discriminator.parameters():
                p.requires_grad_(True)
        print(f"GAN epoch={epoch + 1}", flush=True)
    return generator.eval()


@torch.no_grad()
def evaluate(model, batches, device, classes):
    truth, predictions = [], []
    for x, y in batches:
        predictions.extend(model(x.to(device)).argmax(1).cpu().tolist())
        truth.extend(y.tolist())
    return {"accuracy": accuracy_score(truth, predictions),
            "macro_f1": f1_score(truth, predictions, labels=list(range(classes)), average="macro", zero_division=0),
            "confusion_matrix": confusion_matrix(truth, predictions, labels=list(range(classes))).tolist()}


@torch.no_grad()
def scores(model, batches, device):
    return np.concatenate([((model(x.to(device)) - x.to(device)) ** 2).mean((1, 2, 3)).cpu().numpy() for x, _ in batches])


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    splits, labels = read_manifest(args.manifest)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    val = loader(splits["val"], labels, args)
    clean_train = loader(splits["train"], labels, args, shuffle=True)
    gan = train_gan(clean_train, len(labels), args) if args.with_gan else None
    if gan is not None:
        torch.save(gan.state_dict(), output / "generator.pt")
    results = {"config": vars(args), "labels": labels, "classification": [], "anomalies": []}
    conditions = [None, 30, 20, 10, 0, -5]
    for name, architecture in [("cnn", CNN), ("vgg", VGG)]:
        for augmentation in ["clean", "noise"] + (["gan"] if gan is not None else []):
            # Matched initialization across augmentation variants.
            torch.manual_seed(args.seed)
            random.seed(args.seed)
            train = loader(splits["train"], labels, args, shuffle=True, augment=augmentation == "noise")
            start = time.perf_counter()
            model = fit(architecture(len(labels)), train, val, args, generator=gan if augmentation == "gan" else None)
            elapsed = time.perf_counter() - start
            torch.save(model.state_dict(), output / f"{name}_{augmentation}.pt")
            for split in ["test"] + (["train"] if args.train_diagnostics else []):
                for snr in conditions:
                    metrics = evaluate(model, loader(splits[split], labels, args, snr=snr), args.device, len(labels))
                    results["classification"].append(dict(model=name, augmentation=augmentation, split=split, snr_db=snr,
                        parameters=sum(p.numel() for p in model.parameters()), training_seconds=elapsed, **metrics))
    autoencoder = fit(Autoencoder(), clean_train, val, args, autoencoder=True)
    threshold = float(np.quantile(scores(autoencoder, val, args.device), 0.95))
    torch.save({"state_dict": autoencoder.state_dict(), "threshold": threshold}, output / "autoencoder.pt")
    clean_scores = scores(autoencoder, loader(splits["test"], labels, args), args.device)
    for snr in conditions[1:]:
        noisy_scores = scores(autoencoder, loader(splits["test"], labels, args, snr=snr), args.device)
        truth = np.r_[np.zeros(len(clean_scores)), np.ones(len(noisy_scores))]
        values = np.r_[clean_scores, noisy_scores]
        predicted = values > threshold
        results["anomalies"].append(dict(snr_db=snr, threshold=threshold,
            precision=precision_score(truth, predicted, zero_division=0), recall=recall_score(truth, predicted, zero_division=0),
            f1=f1_score(truth, predicted, zero_division=0), roc_auc=roc_auc_score(truth, values)))
    (output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def export(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    wave = load_audio(args.audio)
    noisy = add_noise(wave, args.snr, torch.Generator().manual_seed(args.seed))
    spectrum = stft(noisy)
    torch.save({"stft": spectrum, "length": len(wave), "sample_rate": SAMPLE_RATE}, output / "noisy_stft.pt")
    # FLOAT preserves levels without clipping, hence preserves requested SNR.
    for name, signal in [("clean", wave), ("noisy", noisy), ("reconstructed", reconstruct(spectrum, len(wave)))]:
        sf.write(output / f"{name}.wav", signal.numpy(), SAMPLE_RATE, subtype="FLOAT")
    import matplotlib.pyplot as plt
    for name, signal in [("clean", wave), ("noisy", noisy)]:
        plt.imsave(output / f"{name}.png", image(signal)[0].numpy(), origin="lower", cmap="magma", vmin=-1, vmax=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    experiment = commands.add_parser("run")
    experiment.add_argument("--manifest", required=True)
    experiment.add_argument("--output", required=True)
    experiment.add_argument("--epochs", type=int, default=20)
    experiment.add_argument("--gan-epochs", type=int, default=30)
    experiment.add_argument("--batch-size", type=int, default=16)
    experiment.add_argument("--lr", type=float, default=0.0002)
    experiment.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    experiment.add_argument("--with-gan", action="store_true")
    experiment.add_argument("--train-diagnostics", action="store_true")
    experiment.add_argument("--seed", type=int, default=42)
    audio = commands.add_parser("export")
    audio.add_argument("--audio", required=True)
    audio.add_argument("--output", required=True)
    audio.add_argument("--snr", type=float, default=10)
    audio.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "run":
        if min(args.epochs, args.gan_epochs, args.batch_size) < 1 or args.lr <= 0:
            parser.error("Epoch counts, batch size and learning rate must be positive")
        run(args)
    else:
        export(args)


if __name__ == "__main__":
    main()
