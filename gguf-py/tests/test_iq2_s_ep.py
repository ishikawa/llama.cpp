import struct

import numpy as np
import tempfile
from pathlib import Path
import unittest

from gguf import GGUFReader, GGMLQuantizationType
from gguf.scripts.gguf_iq2_s_ep import convert_shard, pack_expert, shard_paths, unpack_expert


def check_roundtrip(nblk, nrow):
    raw = np.random.default_rng(42).integers(0, 256, nblk*nrow*82, dtype=np.uint8).tobytes()
    packed = pack_expert(raw)
    assert len(packed) == len(raw)
    assert len(raw)//82*80 % 16 == 0
    assert unpack_expert(packed) == raw
    assert packed[:80] == raw[2:82]
    assert packed[nblk*nrow*80:nblk*nrow*80+2] == raw[:2]


def make_shard(path, endian):
    names = [b'blk.0.ffn_gate_exps.weight', b'blk.0.ffn_up_exps.weight', b'blk.0.ffn_down_exps.weight']
    raw = np.random.default_rng(11).integers(0, 256, 2*3*82, dtype=np.uint8).tobytes()
    offsets = [0, 512, 1024]
    header = b'GGUF' + struct.pack(endian+'IQQ', 3, len(names), 1)
    header += struct.pack(endian+'Q', 4) + b'test' + struct.pack(endian+'II', 4, 123)
    for name, offset in zip(names, offsets):
        header += struct.pack(endian+'Q', len(name)) + name
        header += struct.pack(endian+'IQQQIQ', 3, 256, 3, 2, 22, offset)
    header += bytes((-len(header)) % 32)
    data = bytearray(1536)
    for offset in offsets:
        data[offset:offset+len(raw)] = raw
    path.write_bytes(header + data)
    return raw


def check_shard_preserves_bytes(tmp_path, endian):
    source = tmp_path / 'model-00001-of-00002.gguf'
    raw = make_shard(source, endian)
    second = tmp_path / 'model-00002-of-00002.gguf'
    make_shard(second, endian)
    assert shard_paths(source) == [source, second]
    output = tmp_path / 'output.gguf'
    convert_shard(source, output, True)
    original = GGUFReader(source)
    converted = GGUFReader(output)
    expected = bytearray(source.read_bytes())
    for tensor in original.tensors[:2]:
        blocks = 3
        for expert in range(2):
            start = tensor.data_offset + expert*blocks*82
            expected[start:start+blocks*82] = pack_expert(raw[expert*blocks*82:(expert+1)*blocks*82])
        type_offset = tensor.field.offset + sum(part.nbytes for part in tensor.field.parts[:4])
        expected[type_offset:type_offset+4] = struct.pack(endian+'I', 43)
    assert output.read_bytes() == expected
    assert converted.tensors[0].tensor_type == GGMLQuantizationType.IQ2_S_EP
    assert converted.tensors[2].data.tobytes() == original.tensors[2].data.tobytes()
    try:
        convert_shard(source, output, False)
    except ValueError:
        pass
    else:
        raise AssertionError('existing output accepted')


class TestIQ2SEP(unittest.TestCase):
    def test_roundtrip(self):
        for nblk, nrow in [(1, 1), (3, 7), (10, 640), (17, 9)]:
            with self.subTest(nblk=nblk, nrow=nrow):
                check_roundtrip(nblk, nrow)

    def test_chunk_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.gguf'
            output = Path(directory) / 'output.gguf'
            name = b'blk.0.ffn_gate_exps.weight'
            blocks = 8197
            header = b'GGUF' + struct.pack('<IQQ', 3, 1, 0)
            header += struct.pack('<Q', len(name)) + name
            header += struct.pack('<IQQQIQ', 3, 256, blocks, 2, 22, 0)
            header += bytes((-len(header)) % 32)
            raw = np.random.default_rng(12).integers(0, 256, blocks*2*82, dtype=np.uint8).tobytes()
            source.write_bytes(header + raw)
            convert_shard(source, output, True)
            converted = GGUFReader(output).tensors[0].data.tobytes()
            expected = b''.join(pack_expert(raw[i*blocks*82:(i+1)*blocks*82]) for i in range(2))
            self.assertEqual(converted, expected)

    def test_shards(self):
        for endian in ['<', '>']:
            with self.subTest(endian=endian), tempfile.TemporaryDirectory() as directory:
                check_shard_preserves_bytes(Path(directory), endian)


if __name__ == '__main__':
    unittest.main()
