from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
from transformers import DataCollatorForSeq2Seq

from constants import (
    MAX_INPUT_LENGTH,
    MAX_TARGET_LENGTH,
    SRC_LANG,
    TGT_LANG,
    TRAIN_SAMPLES,
    VAL_SAMPLES,
)


def build_dataset() -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    """
    Build the dataset.

    Returns:
        The dataset.

    NOTE: You can replace this with your own dataset. Make sure to include
    the `validation` split and ensure that it is the same as the test split from the WMT19 dataset,
    Which means that:
        raw_datasets["validation"] = load_dataset('wmt19', 'zh-en', split="validation")
    """
    dataset = load_dataset("wmt19", "zh-en")

    # Subsample training for manageable compute while keeping the official validation/test
    train_dataset = dataset["train"].select(range(TRAIN_SAMPLES))

    # Validation mirrors the official test split to align evaluation
    validation_dataset = dataset["validation"].select(range(VAL_SAMPLES))

    # NOTE: You should not change the test dataset
    test_dataset = dataset["validation"]
    return DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset,
    })


def create_data_collator(tokenizer, model):
    """
    Create data collator for sequence-to-sequence tasks.

    Args:
        tokenizer: Tokenizer object.
        model: Model object.

    Returns:
        DataCollatorForSeq2Seq instance.
    """
    return DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


def preprocess_function(
    examples,
    prefix,
    tokenizer,
    max_input_length,
    max_target_length,
    src_lang,
    tgt_lang,
):
    """
    Preprocess the data.

    Args:
        examples: Examples.
        prefix: Prefix.
        tokenizer: Tokenizer object.
        max_input_length: Maximum input length.
        max_target_length: Maximum target length.

    Returns:
        Model inputs.
    """
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang

    inputs = [prefix + ex["zh"] for ex in examples["translation"]]
    targets = [ex["en"] for ex in examples["translation"]]

    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
    )
    labels = tokenizer(
        text_target=targets,
        max_length=max_target_length,
        truncation=True,
        padding="max_length",
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def preprocess_data(raw_datasets: DatasetDict, tokenizer) -> DatasetDict:
    """
    Preprocess the data.

    Args:
        raw_datasets: Raw datasets.
        tokenizer: Tokenizer object.

    Returns:
        Tokenized datasets.
    """
    tokenized_datasets: DatasetDict = raw_datasets.map(
        function=lambda examples: preprocess_function(
            examples=examples,
            prefix="",
            tokenizer=tokenizer,
            max_input_length=MAX_INPUT_LENGTH,
            max_target_length=MAX_TARGET_LENGTH,
            src_lang=SRC_LANG,
            tgt_lang=TGT_LANG,
        ),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )
    return tokenized_datasets
