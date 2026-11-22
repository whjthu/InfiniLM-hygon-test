"""Pin the physical sequence and the orientation of every native batch slot."""
from hashlib import sha256
import json


PAIR_PINS = (
    "b98b2473335cecd21c65be9dfb64d908b177802adfa74ce5750a5e4c4efeac50",
    "4ea6cd0cb1327422c4d68986752ea2a4a64776a6dc77746b966745899ba12991",
    "19e7ff4d4632231749c4d48d783a61db290cb6cf8da092c226d46383c0e3ea03",
    "69fbcb5e3057ab58be631856fa893a179734e5a60ca38d1075e25d1ebdd1696a",
    "11d30860cfd684b868b029198a67618677d4dcbdda744e461709339ac8d30b90",
    "94c397ea3f92a6be4551e676764a8034c0da7f5d7b4d08a83786017d3b7246ba",
    "77e5a8ab4176e40ec256644b616803871f089f322c7b0e186db6b8313497541d",
)
PAIR_COUNT = len(PAIR_PINS)
ORDER = tuple((index, "forward" if (index + cycle) % 2 == 0 else "reverse")
              for cycle in range(2) for index in range(PAIR_COUNT))
VIEW_BATCHES = len(ORDER)


def bindings(pins):
    if tuple(pins) != PAIR_PINS:
        raise ValueError("Ordered physical training pairs differ")
    return [
        dict(slot=slot, physical_index=index, source_manifest_SHA256=PAIR_PINS[index],
             direction=direction)
        for slot, (index, direction) in enumerate(ORDER)
    ]


def view_order_sha256():
    serialized = json.dumps(bindings(PAIR_PINS), sort_keys=True, separators=(",", ":"),
                            allow_nan=False) + "\n"
    return sha256(serialized.encode("ascii")).hexdigest()


def validate_observed_views(rows):
    expected = bindings(PAIR_PINS)
    if type(rows) is not list or len(rows) != VIEW_BATCHES:
        raise ValueError("Expected fourteen observed orientation slots")
    forward = {}
    reverse = {}
    for observed, binding in zip(rows, expected):
        if any(observed.get(key) != value for key, value in binding.items()):
            raise ValueError("Observed orientation slot or physical pair differs")
        ids = observed.get("input_frame_ids")
        if (type(ids) is not list or len(ids) != 2 or
                any(type(frame) is not int or frame < 0 for frame in ids)):
            raise ValueError("Observed frame identities differ")
        direction = binding["direction"]
        if (direction == "forward" and ids[0] >= ids[1]) or (direction == "reverse" and ids[0] <= ids[1]):
            raise ValueError("Observed frame order disagrees with declared direction")
        table = forward if direction == "forward" else reverse
        table[binding["physical_index"]] = ids
    if any(reverse[index] != forward[index][::-1] for index in range(PAIR_COUNT)):
        raise ValueError("Reverse view must use the exact same physical frames")
