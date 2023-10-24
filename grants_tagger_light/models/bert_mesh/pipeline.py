from transformers import Pipeline
from transformers.pipelines import PIPELINE_REGISTRY
from transformers.modeling_outputs import SequenceClassifierOutput
from grants_tagger_light.models.bert_mesh.model import BertMesh


class BertMeshPipeline(Pipeline):
    def _sanitize_parameters(self, **pipeline_parameters):
        preprocess_kwargs = {}
        forward_kwargs = {}
        postprocess_kwargs = {
            "return_labels": pipeline_parameters.get("return_labels", True),
            "threshold": pipeline_parameters.get("threshold", 0.5),
        }

        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(self, input_txt, **preprocess_kwargs):
        return self.tokenizer(
            input_txt,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )["input_ids"].to(self.device)

    def _forward(self, input_ids, **forward_kwargs):
        return self.model(input_ids)

    def postprocess(
        self,
        model_outputs: SequenceClassifierOutput,
        return_labels: bool,
        threshold: float = 0.5,
    ):
        print('I AM IN THE POSTPROCESS MY GOD')
        if return_labels:
            outs = [
                [
                    self.model.id2label[label_id]
                    for label_id, label_prob in enumerate(logit)
                    if label_prob > threshold
                ]
                for logit in model_outputs.logits
            ]

        else:
            outs = model_outputs.logits

        return outs


PIPELINE_REGISTRY.register_pipeline(
    "grants-tagging", pipeline_class=BertMeshPipeline, pt_model=BertMesh
)
