import os, sys
from typing import Callable, Optional, Dict, List, Tuple
from torchvision.datasets import folder
from torchvision.datasets.vision import VisionDataset
from .compute_sizes_index import build_image_size_index


class ImageDataset(VisionDataset):
    """
    ImageFolder-like dataset with optional cached directory listing.
    Returns a dict with keys: image, target, path (uniform across uses).
    """

    def __init__(
            self,
            root: str,
            root_cache: Optional[str] = None,
            loader: Callable = folder.default_loader,
            extensions=folder.IMG_EXTENSIONS,
            transform: Optional[Callable] = None,
            augmentation_transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable] = None,
            include_labels: bool = False,
            min_side_threshold: Optional[int] = None,  # if set, keep only images with min(w,h) > threshold
            size_index_path: Optional[str] = None,  # defaults to root_cache/../train.sizes.tsv
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.augmentation_transform = augmentation_transform

        classes, class_to_idx = self._find_classes(root)
        cache_root = root if root_cache is None else root_cache
        cache_path = cache_root.rstrip("/") + ".txt"

        if os.path.isfile(cache_path):
            print(f"Using directory list at: {cache_path}")
            samples = []
            with open(cache_path) as f:
                for line in f:
                    rel_path, idx = line.strip().split(";")
                    samples.append((rel_path, int(idx)))
        else:
            print(f"Walking directory: {root}")
            raw = folder.make_dataset(root, class_to_idx, extensions, is_valid_file)
            samples = []
            with open(cache_path, "w") as f:
                for abs_path, label in raw:
                    rel_path = os.path.relpath(abs_path, root)
                    f.write(f"{rel_path};{label}\n")
                    samples.append((rel_path, label))

        if not samples:
            raise RuntimeError(
                f"Found 0 files in subfolders of: {root} "
                f"Supported extensions are: {','.join(extensions)}"
            )

        if min_side_threshold is not None:
            sizes_path = size_index_path or (cache_root.rstrip("/") + ".sizes.tsv")
            if not os.path.isfile(sizes_path):
                build_image_size_index(root, root_cache, sizes_path, num_workers=512)
                # raise FileNotFoundError(
                #     f"Missing size index at '{sizes_path}'. Run build_image_size_index(...)."
                # )

            # Stream the TSV once, keeping only records we need (avoids loading 1.2M rows into a dict)
            needed = {rel for (rel, _) in samples}
            min_side_of: Dict[str, int] = {}
            with open(sizes_path, "r") as f:
                for line in f:
                    if not line or line[0] == "#":
                        continue
                    parts = line.rstrip("\n").split("\t")
                    # Expect 5 columns: rel, label, w, h, ms
                    if len(parts) < 5:
                        continue
                    rel, _, _, _, ms = parts[:5]
                    if rel in needed:
                        try:
                            min_side_of[rel] = int(ms)
                        except ValueError:
                            pass

            def passes(ms: Optional[int]) -> bool:
                if ms is None or ms < 0:
                    return False
                return ms >= min_side_threshold

            filtered: List[Tuple[str, int]] = []
            dropped = 0
            for rel, lbl in samples:
                if passes(min_side_of.get(rel)):
                    filtered.append((rel, lbl))
                else:
                    dropped += 1
            samples = filtered
            if not samples:
                raise RuntimeError(
                    f"After filtering with min_side_threshold={min_side_threshold}, no samples remain."
                )

        self.loader = loader
        self.extensions = extensions
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.include_labels = include_labels

    def _find_classes(self, dir: str):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index: int) -> Dict:
        rel_path, target = self.samples[index]
        path = os.path.join(self.root, rel_path)
        img_ = self.loader(path)  # PIL

        assert self.transform is not None, "transform must be set"
        img = self.transform(img_)  # Tensor[C,H,W]

        if self.augmentation_transform is not None:
            aug_img = self.augmentation_transform(img_)
        else:
            raise ValueError()
            aug_img = img

        if self.target_transform is not None and self.include_labels:
            target = self.target_transform(target)

        return {
            "index": index,
            "image": img,
            "aug_image": aug_img,
            "target": target,
            "path": path,
        }

    def __len__(self):
        return len(self.samples)
