"""Project tensor shapes and storage sizes from the pinned original HunyuanVideo SFT cache."""
from collections import OrderedDict
from hashlib import sha256
import io
import json
from pathlib import Path
import pickle
import stat
from zipfile import ZipFile


CACHE = ("/public/home/aclsylqidf/hygon-inference-refactor-20261014/"
         "results/training-hunyuanvideo-original-inputs-v1/sample.pt")
SIZE = 3521630
DIGEST = "9bdbaccaada6be6080ace96aa464f70adc8041bd43e58eae58052c5b2eab541b"
METADATA = "49e7a10cdf84c5daaff9a60c78d02022615ce4ed0ab0a8f5aaa2ba0a2ab8e648"
WIDTH = {"BFloat16Storage": 2, "HalfStorage": 2, "FloatStorage": 4, "DoubleStorage": 8,
         "BoolStorage": 1, "ByteStorage": 1, "CharStorage": 1, "ShortStorage": 2,
         "IntStorage": 4, "LongStorage": 8}


class Storage:
    def __init__(self, kind, key, elements):
        self.kind = kind
        self.key = key
        self.elements = elements


class Tensor:
    def __init__(self, storage, offset, shape, stride, *unused):
        if (not isinstance(storage, Storage) or len(shape) != len(stride) or len(shape) > 8
                or type(offset) is not int or offset < 0
                or any(type(dim) is not int or dim <= 0 for dim in shape)
                or any(type(step) is not int or step < 0 for step in stride)
                or offset + sum((dim-1)*step for dim, step in zip(shape, stride)) >= storage.elements):
            raise ValueError("Tensor descriptor outside the pinned storage")
        self.storage = storage
        self.offset = offset
        self.shape = shape
        self.stride = stride


class ShapeReader(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch" and name in WIDTH:
            return name
        if module == "torch._utils" and name in {"_rebuild_tensor", "_rebuild_tensor_v2"}:
            return Tensor
        if (module, name) == ("collections", "OrderedDict"):
            return OrderedDict
        raise ValueError("Unknown descriptor constructor")

    def persistent_load(self, identity):
        tag, dtype, key, _location, elements = identity
        if (tag != "storage" or dtype not in WIDTH or not isinstance(key, str)
                or not key.isdecimal() or type(elements) is not int or elements <= 0):
            raise ValueError("Unexpected persistent storage descriptor")
        return Storage(dtype, key, elements)


def inspect(path, metadata_only=False):
    path = Path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size != SIZE:
        raise ValueError("Original regular cache file required")
    source = path
    if not metadata_only:
        with path.open("rb") as fd:
            raw = fd.read(SIZE + 1)
        if len(raw) != SIZE or sha256(raw).hexdigest() != DIGEST:
            raise ValueError("Original cache identity changed")
        source = io.BytesIO(raw)
    with ZipFile(source) as archive:
        manifest = [item for item in archive.infolist() if item.filename.endswith("/data.pkl")]
        if len(manifest) != 1 or not 0 < manifest[0].file_size <= 32768:
            raise ValueError("One bounded tensor directory is required")
        descriptor = archive.read(manifest[0])
        descriptor_sha = sha256(descriptor).hexdigest()
        if descriptor_sha != METADATA:
            raise ValueError("Original tensor directory identity changed")
        data = ShapeReader(io.BytesIO(descriptor)).load()
        if not isinstance(data, dict) or not 1 <= len(data) <= 32:
            raise ValueError("Original cache is not a field dictionary")
        prefix = manifest[0].filename.rsplit("/", 1)[0]
        tensors = {}
        other = {}
        for name, value in data.items():
            if (not isinstance(name, str) or len(name) > 64
                    or not name.isascii() or not name.replace("_", "").isalnum()):
                raise ValueError("Unexpected cache field name")
            if not isinstance(value, Tensor):
                other[name] = type(value).__name__
                continue
            storage = value.storage
            size = archive.getinfo(prefix + "/data/" + storage.key).file_size
            if size != storage.elements * WIDTH[storage.kind]:
                raise ValueError("Tensor storage member length differs")
            tensors[name] = dict(shape=value.shape, stride=value.stride, storage_dtype=storage.kind,
                                 storage_offset=value.offset, storage_bytes=size)
        if not tensors:
            raise ValueError("No original tensor descriptors")
        dimensions = {name: data[name] for name in ("frames", "height", "width")}
        if not all(type(value) is int and 0 < value <= 8192 for value in dimensions.values()):
            raise ValueError("Original video dimensions missing")
    after = path.lstat()
    if any(getattr(before, field) != getattr(after, field) for field in
           ("st_dev", "st_ino", "st_size", "st_ctime_ns", "st_mtime_ns")):
        raise ValueError("Original cache changed during inspection")
    return dict(cache_SHA256=DIGEST if not metadata_only else None, cache_bytes=before.st_size,
                metadata_only=metadata_only, metadata_SHA256=descriptor_sha, media_dimensions=dimensions,
                tensor_fields=tensors, other_field_types=other, discloses_tensor_values=False,
                torch_imported=False, actual_GPU=0, training_rows_created=0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(inspect(CACHE, metadata_only=args.metadata_only), sort_keys=True))
