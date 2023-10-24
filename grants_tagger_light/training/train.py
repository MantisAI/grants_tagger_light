from transformers import (
    Trainer,
    TrainingArguments,
    EvalPrediction,
    HfArgumentParser,
    AutoConfig,
    AdamW,
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from grants_tagger_light.models.bert_mesh import BertMesh
from grants_tagger_light.preprocessing.preprocess_mesh import preprocess_mesh
from grants_tagger_light.training.cli_args import (
    BertMeshTrainingArguments,
    WandbArguments,
    BertMeshModelArguments,
)
from grants_tagger_light.training.dataloaders import (
    MultilabelDataCollator,
)
from sklearn.metrics import classification_report
from loguru import logger
from pprint import pformat
import typer
import numpy as np

import os
import transformers
import json
from datasets import load_from_disk

from grants_tagger_light.utils.sharding import Sharding
from grants_tagger_light.utils.years_tags_parser import parse_years, parse_tags

transformers.set_seed(42)


def train_bertmesh(
    model_key: str,
    data_path: str,
    training_args: TrainingArguments,
    model_args: BertMeshModelArguments = None,
    max_samples: int = -1,
    test_size: float = None,
    num_proc: int = os.cpu_count(),
    shards: int = os.cpu_count(),
    from_checkpoint: str = None,
    tags: list = None,
    train_years: list = None,
    test_years: list = None,
):
    if not model_key:
        assert isinstance(model_args, BertMeshModelArguments), (
            "If model_key is not provided, "
            "must provide model_args of type BertMeshModelArguments"
        )  # noqa

    logger.info(f"Preprocessing the dataset at {data_path}...")
    if os.path.isdir(data_path):
        logger.info(
            "Train/test data found in a folder, which means you preprocessed and "
            "save the data before. Loading that split from disk..."
        )
        dset = load_from_disk(os.path.join(data_path, "dataset"))
        with open(os.path.join(data_path, "label2id"), "r") as f:
            label2id = json.load(f)
        with open(os.path.join(data_path, "id2label"), "r") as f:
            id2label = json.load(f)
    else:
        logger.info("Preprocessing the data on the fly...")
        dset, label2id, id2label = preprocess_mesh(
            data_path=data_path,
            model_key=model_key,
            test_size=test_size,
            num_proc=num_proc,
            max_samples=max_samples,
            batch_size=training_args.per_device_train_batch_size,
            tags=tags,
            train_years=train_years,
            test_years=test_years,
        )
    
    train_dset, val_dset = dset["train"], dset["test"]

    metric_labels = []
    for x in train_dset["label_ids"]:
        metric_labels.extend(x)

    train_dset_size = len(train_dset)
    logger.info(f"Training dataset size: {train_dset_size}")
    if max_samples > 0:
        train_dset_size = min(max_samples, train_dset_size)
        logger.info(f"Training max samples: {train_dset_size}.")
        train_dset.filter(
            lambda example, idx: idx < train_dset_size,
            with_indices=True,
            num_proc=num_proc,
        )
    else:
        logger.info("Training with all data...")

    if shards > 0:
        logger.info("Sharding training dataset...")
        train_dset = Sharding(num_shards=shards).shard(train_dset)

    if not model_key:
        logger.info(
            f"Model key not found. "
            f"Training from scratch {model_args.pretrained_model_key}"
        )

        # Instantiate model from scratch
        logger.info(f"Loading `{model_args.pretrained_model_key}` tokenizer...")
        config = AutoConfig.from_pretrained(model_args.pretrained_model_key)

        config.update(
            {
                "pretrained_model": model_args.pretrained_model_key,
                "num_labels": len(label2id),
                "hidden_size": model_args.hidden_size,
                "dropout": model_args.dropout,
                "multilabel_attention": model_args.multilabel_attention,
                "label2id": label2id,
                "id2label": id2label,
                "freeze_backbone": model_args.freeze_backbone,
                "hidden_dropout_prob": model_args.hidden_dropout_prob,
                "attention_probs_dropout_prob": model_args.attention_probs_dropout_prob,
            }
        )
        logger.info(f"Hidden size: {config.hidden_size}")
        logger.info(f"Dropout: {config.dropout}")
        logger.info(f"Multilabel Attention: {config.multilabel_attention}")
        logger.info(f"Freeze Backbone: {config.freeze_backbone}")
        logger.info(f"Num labels: {config.num_labels}")
        logger.info(f"hidden_dropout_prob: {config.hidden_dropout_prob}")
        logger.info(
            f"attention_probs_dropout_prob: {config.attention_probs_dropout_prob}"
        )

        model = BertMesh(config)

    else:
        logger.info(f"Training from pretrained key {model_key}")
        model = BertMesh.from_pretrained(model_key, trust_remote_code=True)

    if model_args.freeze_backbone is None:
        model_args.freeze_backbone = "freeze"

    if model_args.freeze_backbone.lower().strip() == "unfreeze":
        logger.info("Unfreezing weights&biases in the backbone")
        model.unfreeze_backbone()
    elif model_args.freeze_backbone.lower().strip() == "unfreeze_bias":
        logger.info("Unfreezing only biases in the backbone")
        model.unfreeze_backbone(only_bias=True)
    elif model_args.freeze_backbone.lower().strip() == "freeze":
        logger.info("Freezing backbone")
        model.freeze_backbone()

    def sklearn_metrics(prediction: EvalPrediction):
        # This is a batch, so it's an array (rows) of array (labels)
        # Array of arrays with probas [[5.4e-5 1.3e-3...] [5.4e-5 1.3e-3...] ... ]
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        y_pred = np.array([sigmoid(y) for y in prediction.predictions])
        top_indexes = np.argsort(np.sum(prediction.label_ids, axis=0))[::-1][:5]
        #print(y_pred.shape)
        #print('mean probabilties and standard deviations for 10 first examples')
        #print(np.mean(y_pred[:11, :],axis=1), np.std(y_pred[:11, :], axis=1))
        #print('ids')
        #print(np.where(prediction.label_ids[10, :] == 1)[0])
        #print('Probabilities for true value')
        #print(y_pred[10, np.where(prediction.label_ids[10, :] == 1)[0]])
        #print('Higher predictions')
        #print(np.sort(y_pred[10, :])[::-1])
        # Transformed to 0-1 if bigger than threshold [[0 1 0...] [0 0 1...] ... ]
        y_pred = np.int64(y_pred > training_args.threshold)
        #print('Number of predictions and outputs')
        #print(sum(y_pred[10, :]), y_pred[10, :])
        logger.info("predictions made")
        logger.info(np.sum(y_pred))

        # Array of arrays with 0/1 [[0 0 1 ...] [0 1 0 ...] ... ]
        y_true = prediction.label_ids

        # report = classification_report(y_pred, y_true, output_dict=True)

        reports = {index: {'pred_max': np.max(y_pred[[0, 1000, 2500, 10000], :], axis=1)}
                    for index in top_indexes}
        logger.info("highest pred calculated")

        if training_args.prune_labels_in_evaluation:
            mask = np.zeros(y_pred.shape, dtype=bool)
            mask[np.arange(y_pred.shape[0])[:, np.newaxis], metric_labels] = True

            filtered_y_pred = y_pred[mask].reshape(y_pred.shape[0], -1)
            filtered_y_true = y_true[mask].reshape(y_true.shape[0], -1)
        else:
            filtered_y_pred = y_pred
            filtered_y_true = y_true
        
        logger.info("pruned labels in evaluation")

        report = classification_report(
            filtered_y_true, filtered_y_pred, output_dict=True, 
                                       zero_division=0.0,
        )
        logger.info("classification report computed")

        metric_dict = {
            "micro_avg": report["micro avg"],
            "macro_avg": report["macro avg"],
            "weighted_avg": report["weighted avg"],
            "samples_avg": report["samples avg"],
        }

        metric_dict = {**metric_dict, 
                       **{f'index_{index}': reports.get(index)
                        for index in top_indexes}}


        return metric_dict

    logger.info("Collating labels...")
    collator = MultilabelDataCollator(label2id=label2id)

    if shards > 0:
        logger.info("Calculating max steps for IterableDatasets shards...")
        max_steps = Sharding.calculate_max_steps(training_args, train_dset_size)
        training_args.max_steps = max_steps

    optimizer = AdamW(
        model.parameters(),
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        correct_bias=training_args.correct_bias
        if hasattr(training_args, "correct_bias")
        else True,
    )

    if training_args.warmup_steps is None:
        training_args.warmup_steps = 0

    if training_args.scheduler_type.lower().strip() == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.max_steps,
        )
    elif training_args.scheduler_type.lower().strip() == "constant":
        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=training_args.warmup_steps
        )
    elif training_args.scheduler_type.lower().strip() == "cosine_hard_restart":
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.max_steps,
            num_cycles=training_args.num_train_epochs,
        )
    elif training_args.scheduler_type.lower().strip() == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.max_steps,
        )
    else:
        logger.warning(
            f"{training_args.scheduler_type}: not found or not valid. "
            f"Falling back to `linear`"
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.max_steps,
        )

    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Scheduler: {training_args.scheduler_type}")

    training_args.optim = optimizer
    training_args.lr_scheduler_type = scheduler

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dset,
        eval_dataset=val_dset,
        data_collator=collator,
        compute_metrics=sklearn_metrics,
        optimizers=(optimizer, scheduler),
    )
    # logger.info(training_args)

    if from_checkpoint is None:
        logger.info("Training...")
    else:
        logger.info(f"Resuming training from checkpoint: {from_checkpoint}")
    trainer.train()

    logger.info("Saving the model...")
    trainer.save_model(os.path.join(training_args.output_dir, "best"))

    logger.info("Evaluating...")
    metrics = trainer.evaluate(eval_dataset=val_dset)

    logger.info(pformat(metrics))
    with open(os.path.join(training_args.output_dir, "metrics"), "w") as f:
        f.write(pformat(metrics))


train_app = typer.Typer()


@train_app.command()
def train_bertmesh_cli(
    ctx: typer.Context,
    model_key: str = typer.Argument(
        ..., help="Pretrained model key. " "Local path or HF location"
    ),
    data_path: str = typer.Argument(
        ...,
        help="Path to allMeSH_2021.jsonl (or similar) "
        "or to a folder after preprocessing and saving to disk",
    ),
    test_size: float = typer.Option(
        None, help="Fraction of data to use for testing (0,1] or number of rows"
    ),
    num_proc: int = typer.Option(
        os.cpu_count(), help="Number of processes to use for preprocessing"
    ),
    max_samples: int = typer.Option(
        -1,
        help="Maximum number of samples to use from the json",
    ),
    shards: int = typer.Option(
        os.cpu_count(),
        help="Number os shards to divide training "
        "IterativeDataset to (improves performance)",
    ),
    from_checkpoint: str = typer.Option(
        None, help="Name of the checkpoint to resume training"
    ),
    tags: str = typer.Option(
        None,
        help="Comma-separated tags you want to include in the dataset "
        "(the rest will be discarded)",
    ),
    train_years: str = typer.Option(
        None, help="Comma-separated years you want to include in the training dataset"
    ),
    test_years: str = typer.Option(
        None, help="Comma-separated years you want to include in the test dataset"
    ),
):
    parser = HfArgumentParser(
        (
            BertMeshTrainingArguments,
            WandbArguments,
            BertMeshModelArguments,
        )
    )
    (
        training_args,
        wandb_args,
        model_args,
    ) = parser.parse_args_into_dataclasses(ctx.args)

    logger.info("Model args: {}".format(pformat(model_args)))
    logger.info("Training args: {}".format(pformat(training_args)))
    logger.info("Wandb args: {}".format(pformat(wandb_args)))

    train_bertmesh(
        model_key=model_key,
        data_path=data_path,
        training_args=training_args,
        model_args=model_args,
        max_samples=max_samples,
        test_size=test_size,
        num_proc=num_proc,
        shards=shards,
        from_checkpoint=from_checkpoint,
        tags=parse_tags(tags),
        train_years=parse_years(train_years),
        test_years=parse_years(test_years),
    )
