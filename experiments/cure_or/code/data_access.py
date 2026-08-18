#!/usr/bin/env python3
"""Selective canonical data retrieval and locked ConvNeXt feature extraction."""

from __future__ import annotations

import csv
import json
import os
import struct
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

from common import DATA, MODEL_SEEDS, sha256, stable_unit


TRAIN_URL = "https://zenodo.org/api/records/4299330/files/train.zip/content"
TEST_URL = "https://zenodo.org/api/records/4299330/files/test.zip/content"
TRAIN_SIZE = 1_883_200_293
TEST_SIZE = 1_526_714_682
WEIGHTS_SHA256 = "983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d"


class RemoteZip:
    def __init__(self, url: str, size: int):
        self.url = url
        self.size = size

    def get(self, start: int, end: int, attempts: int = 7) -> bytes:
        last = None
        for attempt in range(attempts):
            request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    data = response.read()
                if len(data) != end - start + 1:
                    raise RuntimeError(f"range length {len(data)}")
                return data
            except Exception as error:
                last = error
                time.sleep(min(20, 2 ** attempt))
        raise RuntimeError(f"range request failed {start}-{end}: {last}")

    def directory(self) -> dict[str, dict]:
        tail = self.get(self.size - 65536, self.size - 1)
        position = tail.rfind(b"PK\x05\x06")
        if position < 0:
            raise RuntimeError("ZIP end-of-central-directory missing")
        values = struct.unpack_from("<4s4H2LH", tail, position)
        count, directory_size, directory_offset = values[4], values[5], values[6]
        raw = self.get(directory_offset, directory_offset + directory_size - 1)
        entries, cursor = {}, 0
        while cursor < len(raw):
            item = struct.unpack_from("<4s6H3L5H2L", raw, cursor)
            if item[0] != b"PK\x01\x02":
                raise RuntimeError("invalid ZIP central-directory record")
            name_length, extra_length, comment_length = item[10], item[11], item[12]
            name = raw[cursor + 46: cursor + 46 + name_length].decode()
            entries[name] = {
                "method": item[4], "crc32": item[7], "compressed_size": item[8],
                "uncompressed_size": item[9], "offset": item[16],
            }
            cursor += 46 + name_length + extra_length + comment_length
        if len(entries) != count:
            raise RuntimeError(f"ZIP directory count {len(entries)} != {count}")
        return entries

    def member(self, name: str, entry: dict) -> tuple[str, bytes, dict]:
        header = self.get(entry["offset"], entry["offset"] + 29)
        values = struct.unpack("<4s5H3L2H", header)
        start = entry["offset"] + 30 + values[9] + values[10]
        compressed = self.get(start, start + entry["compressed_size"] - 1)
        data = compressed if entry["method"] == 0 else zlib.decompress(compressed, -15)
        crc = zlib.crc32(data) & 0xFFFFFFFF
        if crc != entry["crc32"] or len(data) != entry["uncompressed_size"]:
            raise RuntimeError(f"CRC/size mismatch for {name}")
        return name, data, {"bytes": len(data), "crc32": f"{crc:08x}"}


def required_image_ids() -> dict[str, list[int]]:
    streams = json.loads((DATA / "TARGET_STREAMS_FROZEN.json").read_text())
    test_ids = set()
    for role in ("calibration", "confirmation"):
        for stream in streams[role]:
            test_ids.add(int(stream["baseline_image_id"]))
            test_ids.update(int(value) for value in stream["levels"].values())
    with (DATA / "TRAINING_BASELINE_FROZEN.csv").open(newline="", encoding="utf-8") as handle:
        train_ids = {int(row["image_id"]) for row in csv.DictReader(handle)}
    with (DATA / "REPAIR_CANDIDATES_FROZEN.csv").open(newline="", encoding="utf-8") as handle:
        train_ids.update(int(row["image_id"]) for row in csv.DictReader(handle))
    return {"train": sorted(train_ids), "test": sorted(test_ids)}


def fetch_split(split: str, image_ids: list[int], cache: Path, workers: int = 16) -> dict:
    url, size = (TRAIN_URL, TRAIN_SIZE) if split == "train" else (TEST_URL, TEST_SIZE)
    archive = RemoteZip(url, size)
    entries = archive.directory()
    folder = cache / split
    folder.mkdir(parents=True, exist_ok=True)
    names = [f"{split}/{image_id:05d}.jpg" for image_id in image_ids]
    evidence, pending = {}, []
    for name in names:
        if name not in entries:
            raise RuntimeError(f"frozen member absent from canonical ZIP: {name}")
        path = folder / Path(name).name
        if path.exists():
            data = path.read_bytes()
            crc = zlib.crc32(data) & 0xFFFFFFFF
            if crc != entries[name]["crc32"]:
                raise RuntimeError(f"cached CRC mismatch: {path}")
            evidence[path.name] = {"bytes": len(data), "crc32": f"{crc:08x}", "reused": True}
        else:
            pending.append(name)
    print(f"{split}: {len(evidence)} cached, {len(pending)} to retrieve", flush=True)
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        jobs = {executor.submit(archive.member, name, entries[name]): name for name in pending}
        for completed, future in enumerate(as_completed(jobs), 1):
            name, data, item = future.result()
            filename = Path(name).name
            (folder / filename).write_bytes(data)
            item["reused"] = False
            evidence[filename] = item
            if completed % 100 == 0 or completed == len(pending):
                elapsed = max(time.time() - started, 1e-6)
                print(f"{split}: {completed}/{len(pending)} new ({completed / elapsed:.2f}/s)", flush=True)
    return {"archive_url": url, "archive_size": size, "selected_members": evidence}


def fetch_all(cache: Path, workers: int = 16) -> dict:
    ids = required_image_ids()
    return {split: fetch_split(split, ids[split], cache, workers) for split in ("train", "test")}


def image_tensor(path: Path, seed: int | None = None, image_id: int | None = None, size: int = 192):
    import torch

    with Image.open(path) as image:
        image.load()
        image = image.convert("RGB")
        if seed is not None and image_id is not None:
            scale = 0.80 + 0.20 * stable_unit(f"feature|{seed}|{image_id}|scale")
            width, height = max(1, round(image.width * scale)), max(1, round(image.height * scale))
            left = round((image.width - width) * stable_unit(f"feature|{seed}|{image_id}|left"))
            top = round((image.height - height) * stable_unit(f"feature|{seed}|{image_id}|top"))
            image = image.crop((left, top, left + width, top + height))
            if stable_unit(f"feature|{seed}|{image_id}|flip") < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        image = image.resize((size, size), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - np.asarray([0.485, 0.456, 0.406], np.float32)) / np.asarray([0.229, 0.224, 0.225], np.float32)
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


def load_backbone(weights: Path):
    import torch
    from torchvision.models import convnext_tiny

    if sha256(weights) != WEIGHTS_SHA256:
        raise RuntimeError("ConvNeXt-Tiny weight SHA-256 mismatch")
    model = convnext_tiny(weights=None)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    return model


def extract(model, items: list[tuple[int, Path]], seed: int | None, batch_size: int = 16) -> np.ndarray:
    import torch

    output = []
    started = time.time()
    with torch.inference_mode():
        for start in range(0, len(items), batch_size):
            batch_items = items[start:start + batch_size]
            batch = torch.stack([image_tensor(path, seed, image_id) for image_id, path in batch_items])
            values = model.features(batch)
            values = model.avgpool(values)
            values = model.classifier[0](values)
            output.append(torch.flatten(values, 1).cpu().numpy())
            completed = min(start + batch_size, len(items))
            if completed % 400 < batch_size or completed == len(items):
                elapsed = max(time.time() - started, 1e-6)
                print(f"features seed={seed}: {completed}/{len(items)} ({completed / elapsed:.2f}/s)", flush=True)
    return np.concatenate(output).astype(np.float32)


def generate_features(cache: Path, weights: Path, output: Path) -> None:
    import torch

    if output.exists():
        raise RuntimeError(f"refusing to overwrite feature cache: {output}")
    ids = required_image_ids()
    for split in ("train", "test"):
        missing = [image_id for image_id in ids[split] if not (cache / split / f"{image_id:05d}.jpg").is_file()]
        if missing:
            raise RuntimeError(f"{split} cache missing {len(missing)} frozen images")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    model = load_backbone(weights)
    test_items = [(image_id, cache / "test" / f"{image_id:05d}.jpg") for image_id in ids["test"]]
    train_items = [(image_id, cache / "train" / f"{image_id:05d}.jpg") for image_id in ids["train"]]
    payload = {
        "test_ids": np.asarray(ids["test"], dtype=np.int32),
        "test_features": extract(model, test_items, None),
        "train_ids": np.asarray(ids["train"], dtype=np.int32),
    }
    for seed in MODEL_SEEDS:
        payload[f"train_features_seed{seed}"] = extract(model, train_items, seed)
    np.savez_compressed(output, **payload)

