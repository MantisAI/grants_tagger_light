import json
import numpy as np
from transformers import AutoTokenizer
from datasets import Dataset
from loguru import logger

# TODO refactor the two load funcs into a class


def _tokenize(batch, tokenizer: AutoTokenizer, x_col: str):
    return tokenizer(
        batch[x_col],
        padding="max_length",
        truncation=True,
        max_length=512,
    )


def _encode_labels(sample, label2id: dict):
    sample["label_ids"] = []
    for label in sample["meshMajor"]:
        try:
            sample["label_ids"].append(label2id[label])
        except KeyError:
            logger.warning(f"Label {label} not found in label2id")

    return sample


def _get_label2id(dset):
    label_set = set()
    for sample in dset:
        for label in sample["meshMajor"]:
            label_set.add(label)
    label2id = {label: idx for idx, label in enumerate(label_set)}
    return label2id


def load_mesh_json(
    data_path: str,
    tokenizer: AutoTokenizer,
    label2id: dict,
    test_size: float = 0.1,
    num_proc: int = 8,
    max_samples: int = np.inf,
):
    def _datagen(mesh_json_path: str, max_samples: int = np.inf):
        with open(mesh_json_path, "r", encoding="latin1") as f:
            for idx, line in enumerate(f):
                # Skip 1st line
                if idx == 0:
                    continue
                sample = json.loads(line[:-2])

                if idx > max_samples:
                    break

                yield sample

    dset = Dataset.from_generator(
        _datagen,
        gen_kwargs={"mesh_json_path": data_path, "max_samples": max_samples},
    )

    # Remove unused columns to save space & time
    dset = dset.remove_columns(["journal", "year", "pmid", "title"])

    dset = dset.map(
        _tokenize,
        batched=True,
        batch_size=32,
        num_proc=num_proc,
        desc="Tokenizing",
        fn_kwargs={"tokenizer": tokenizer, "x_col": "abstractText"},
        remove_columns=["abstractText"],
    )

    # Generate label2id if None
    if label2id is None:
        label2id = _get_label2id(dset)

    dset = dset.map(
        _encode_labels,
        batched=False,
        num_proc=num_proc,
        desc="Encoding labels",
        fn_kwargs={"label2id": label2id},
        remove_columns=["meshMajor"],
    )

    # Split into train and test
    dset = dset.train_test_split(test_size=test_size)

    return dset["train"], dset["test"], label2id
