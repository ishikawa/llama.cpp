#!/usr/bin/env python3
"""Repack IQ2_S gate/up experts without rewriting GGUF metadata or other data."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import sys

import numpy as np

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gguf import GGUFReader, GGMLQuantizationType

BLOCK_BYTES = 82
PAYLOAD_BYTES = 80
CHUNK_BLOCKS = 8192


def pack_expert(data: bytes) -> bytes:
    blocks = np.frombuffer(data, dtype=np.uint8).reshape(-1, BLOCK_BYTES)
    return blocks[:, 2:].tobytes() + blocks[:, :2].tobytes()


def unpack_expert(data: bytes) -> bytes:
    if len(data) % BLOCK_BYTES:
        raise ValueError('incomplete IQ2_S_EP expert')
    n = len(data) // BLOCK_BYTES
    blocks = np.empty((n, BLOCK_BYTES), dtype=np.uint8)
    blocks[:, 2:] = np.frombuffer(data, dtype=np.uint8, count=n*PAYLOAD_BYTES).reshape(n, PAYLOAD_BYTES)
    blocks[:, :2] = np.frombuffer(data, dtype=np.uint8, offset=n*PAYLOAD_BYTES).reshape(n, 2)
    return blocks.tobytes()


def shard_paths(first: Path) -> list[Path]:
    match = re.fullmatch(r'(.*)-00001-of-(\d{5})\.gguf', first.name)
    if match is None:
        return [first]
    if int(match[2]) == 0:
        raise ValueError('split count must be positive')
    return [first.with_name(f'{match[1]}-{i:05d}-of-{match[2]}.gguf') for i in range(1, int(match[2])+1)]


def convert_shard(source: Path, output: Path, verify: bool) -> None:
    reader = GGUFReader(source)
    targets = []
    for tensor in reader.tensors:
        if tensor.tensor_type != GGMLQuantizationType.IQ2_S or not re.search(r'(?:^|\.)ffn_(?:gate|up)_exps(?:\.|$)', tensor.name):
            continue
        shape = [int(n) for n in tensor.shape]
        if len(shape) != 3 or shape[0] % 256 or min(shape) <= 0:
            raise ValueError(f'{tensor.name}: expected complete 3D experts, got {shape}')
        blocks = shape[0] // 256 * shape[1]
        if blocks*PAYLOAD_BYTES % 16:
            raise ValueError(f'{tensor.name}: d plane is not 16-byte aligned relative to expert')
        targets.append((tensor, blocks, shape[2]))
    if source.resolve() == output.resolve() or output.exists():
        raise ValueError(f'output already exists or aliases input: {output}')
    # Exclusive creation also prevents following an existing symlink.
    with output.open('xb') as dst, source.open('rb') as src:
        shutil.copyfileobj(src, dst, length=8*1024*1024)
        for tensor, blocks, experts in targets:
            print(f'{source.name}: {tensor.name}', flush=True)
            for expert in range(experts):
                base = tensor.data_offset + expert*blocks*BLOCK_BYTES
                for start in range(0, blocks, CHUNK_BLOCKS):
                    count = min(CHUNK_BLOCKS, blocks-start)
                    src.seek(base + start*BLOCK_BYTES)
                    original = src.read(count*BLOCK_BYTES)
                    if len(original) != count*BLOCK_BYTES:
                        raise ValueError('truncated tensor data')
                    packed = pack_expert(original)
                    dst.seek(base + start*PAYLOAD_BYTES)
                    dst.write(packed[:count*PAYLOAD_BYTES])
                    dst.seek(base + blocks*PAYLOAD_BYTES + start*2)
                    dst.write(packed[count*PAYLOAD_BYTES:])
            parts = tensor.field.parts
            type_offset = tensor.field.offset + sum(part.nbytes for part in parts[:4])
            dst.seek(type_offset)
            dst.write(np.array([GGMLQuantizationType.IQ2_S_EP], dtype=parts[4].dtype).tobytes())
    if verify:
        with source.open('rb') as src, output.open('rb') as dst:
            for tensor, blocks, experts in targets:
                for expert in range(experts):
                    base = tensor.data_offset + expert*blocks*BLOCK_BYTES
                    for start in range(0, blocks, CHUNK_BLOCKS):
                        count = min(CHUNK_BLOCKS, blocks-start)
                        dst.seek(base + start*PAYLOAD_BYTES)
                        payload = dst.read(count*PAYLOAD_BYTES)
                        dst.seek(base + blocks*PAYLOAD_BYTES + start*2)
                        scales = dst.read(count*2)
                        src.seek(base + start*BLOCK_BYTES)
                        if unpack_expert(payload + scales) != src.read(count*BLOCK_BYTES):
                            raise ValueError(f'verification failed: {tensor.name}, expert {expert}, block {start}')
        print(f'{output.name}: verified', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='first shard of the original split GGUF')
    parser.add_argument('output_dir', type=Path, help='new directory for the same shard names')
    parser.add_argument('--verify', action='store_true', help='unpack output experts and compare original bytes')
    args = parser.parse_args()
    sources = shard_paths(args.input)
    for source in sources:
        if not source.is_file():
            parser.error(f'missing shard: {source}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for source in sources:
        convert_shard(source, args.output_dir / source.name, args.verify)


if __name__ == '__main__':
    main()
