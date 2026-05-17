import os
import sys
from multiprocessing import Pool
from typing import List, Optional, Tuple
from PIL import Image
from torchvision.datasets import folder


def _find_classes_static(dir_path: str):
    if sys.version_info >= (3, 5):
        classes = [d.name for d in os.scandir(dir_path) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def _probe_one(args: Tuple[str, str]) -> Tuple[str, int, int, int]:
    root, rel_path = args
    p = os.path.join(root, rel_path)
    try:
        with Image.open(p) as im:
            w, h = im.size
        return rel_path, w, h, min(w, h)
    except Exception:
        # Mark unreadable images with -1; the loader can choose to drop them.
        return rel_path, -1, -1, -1


def _ensure_dirlist(root: str, root_cache: Optional[str]) -> List[Tuple[str, int]]:
    """
    Reuse your existing .txt dirlist (rel_path;label) or build it if missing.
    """
    classes, class_to_idx = _find_classes_static(root)
    cache_root = root if root_cache is None else root_cache
    list_path = cache_root.rstrip("/") + ".txt"

    if os.path.isfile(list_path):
        samples = []
        with open(list_path, "r") as f:
            for line in f:
                rel, idx = line.strip().split(";")
                samples.append((rel, int(idx)))
        return samples

    raw = folder.make_dataset(root, class_to_idx, folder.IMG_EXTENSIONS, None)
    samples = []
    with open(list_path, "w") as f:
        for abs_path, label in raw:
            rel = os.path.relpath(abs_path, root)
            f.write(f"{rel};{label}\n")
            samples.append((rel, label))
    return samples


def build_image_size_index(
        root: str,
        root_cache: Optional[str] = None,
        out_path: Optional[str] = None,
        num_workers: int = 8,
        overwrite: bool = False,
) -> str:
    """
    Writes a TSV with header lines and columns:
    rel_path \t label \t width \t height \t min_side

    File path defaults to (root_cache or root) + ".sizes.tsv".
    """
    cache_root = root if root_cache is None else root_cache
    sizes_path = out_path or (cache_root.rstrip("/") + ".sizes.tsv")

    if os.path.exists(sizes_path) and not overwrite:
        return sizes_path

    samples = _ensure_dirlist(root, root_cache)
    if not samples:
        raise RuntimeError("No files found while building size index.")

    # Parallel probe (I/O bound; modest parallelism helps)
    args = [(root, rel) for (rel, _) in samples]
    with Pool(processes=max(1, num_workers)) as pool:
        results = pool.imap_unordered(_probe_one, args, chunksize=64)

        # Write streaming to avoid holding everything in RAM
        tmp_path = sizes_path + ".tmp"
        with open(tmp_path, "w") as f:
            f.write("# ImageDataset size index v1\n")
            f.write("# rel_path\tlabel\twidth\theight\tmin_side\n")
            # For reproducible label lookup, make a dict once
            label_of = dict(samples)
            written = 0
            for rel, w, h, ms in results:
                lbl = label_of.get(rel, -1)
                f.write(f"{rel}\t{lbl}\t{w}\t{h}\t{ms}\n")
                written += 1

    os.replace(tmp_path, sizes_path)
    return sizes_path
