from functools import lru_cache
import re

from .ezr import DATA, coerce

ARCHIVE = "data/moot.zip"
parse_cell = lru_cache(maxsize=4096)(coerce)


def cells(stream):
    for raw in stream:
        line = re.sub(r"[\n\t\r ]|#.*", "", raw.decode())
        if line:
            yield line.split(",")


def member(archive, name, reference):
    # The probe and cost profile used local variants of a few datasets.
    variant = "reference/" + name
    return variant if reference and variant in archive.namelist() else name


def load_dataset(archive, name, reference=False):
    with archive.open(member(archive, name, reference)) as stream:
        return DATA([parse_cell(x) for x in row] for row in cells(stream))
