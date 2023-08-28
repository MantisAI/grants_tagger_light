import datetime
import json
import math
import os
import random
import uuid

from loguru import logger
import openai
from openai_multi_client import OpenAIMultiClient
import numpy as np

from grants_tagger_light.augmentation.JsonParser import JsonParser


class AugmentOpenAI:
    def __init__(self, prompt_template_path, model_key='gpt-3.5-turbo'):
        if 'OPENAI_API_KEY' not in os.environ:
            logger.error("OPENAI_API_KEY not found in env vars. Please define it before running this program.")
        with open(prompt_template_path, 'r') as f:
            self.prompt_template = f.read()
        self.model_key = model_key
        self.api = OpenAIMultiClient(endpoint="chats", data_template={"model": self.model_key})

    def _create_message(self, abstract, tag):
        prompt = self.prompt_template.replace('{TOPIC}', tag)
        prompt = prompt.replace('{ABSTRACT}', abstract)

        return [{"role": "user", "content": prompt}]

    @staticmethod
    def _process_response(result):
        if result.failed:
            logger.warning(f"Failed to get augmentation for {result.metadata['featured_tag']}")
            return

        with open(result.metadata['save_to_path'], 'a') as f:
            for r in result.response['choices']:
                if 'message' in r:
                    if 'content' in r['message']:
                        try:
                            json_response = JsonParser.parse_json(r['message']['content'])
                        except Exception as e:
                            logger.info(f"Error processing output: {e}. Skipping...")
                            continue

                        a = json_response['abstract']
                        tl = json_response['title']

                        f.write(json.dumps({
                            "journal": result.metadata['model_key'],
                            "meshMajor": result.metadata['tags'],
                            "year": result.metadata['year'],
                            "abstractText": a,
                            "pmid": uuid.uuid4().hex,
                            "title": tl,
                            "existing_example": result.metadata['existing_example'],
                            "required_examples": result.metadata['required_examples'],
                            "featured_tag": result.metadata['featured_tag']
                        }))
                        f.write('\n')
                        f.flush()

                        logger.info(f"Data received successfully for {result.metadata['featured_tag']}")

    def _make_requests(self,
                       collect_concurrent_calls,
                       dset,
                       temperature,
                       top_p,
                       presence_penalty,
                       num_proc,
                       model_key,
                       save_to_path):

        year = datetime.date.year

        for num in range(len(collect_concurrent_calls)):
            t = collect_concurrent_calls[num][0]
            n = collect_concurrent_calls[num][1]
            # RAG: I select similar articles to provide them to the LLM, maximum ´n´ (I don't need more)
            tmp_dset = dset.filter(lambda x: any(np.isin([t], x["meshMajor"])), num_proc=num_proc)
            # I remove them from the dataset to process to make it smaller and quicker over time
            # dset = dset.filter(lambda example, idx: idx not in tmp_dset['idx'], with_indices=True,
            #                   num_proc=num_proc)

            required_examples = n
            existing_examples = min(required_examples, len(tmp_dset))

            n_per_example = math.ceil(required_examples / existing_examples)

            for i in range(existing_examples):
                logger.info(f"Augmenting {t} with {n_per_example} examples using 1 already existing example")
                abstract = tmp_dset['abstractText'][i]
                tags = tmp_dset['meshMajor'][i]
                data = {
                    "model": self.model_key,
                    "n": n_per_example,
                    "temperature": temperature,
                    "top_p": top_p,
                    "presence_penalty": presence_penalty,
                    "messages": self._create_message(abstract, t)
                }

                metadata = {
                    'featured_tag': t,
                    'tags': tags,
                    'required_examples': n,
                    'existing_example': abstract,
                    'year': year,
                    'model_key': model_key,
                    'save_to_path': save_to_path
                }

                self.api.request(data=data, metadata=metadata, callback=self._process_response)

    def generate(self, collect_concurrent_calls, dset, save_to_path, model_key,
                 temperature=1.5, top_p=1, presence_penalty=0, num_proc=os.cpu_count()):
        self.api.run_request_function(self._make_requests,
                                      collect_concurrent_calls=collect_concurrent_calls,
                                      dset=dset,
                                      temperature=temperature,
                                      top_p=top_p,
                                      presence_penalty=presence_penalty,
                                      num_proc=num_proc,
                                      model_key=model_key,
                                      save_to_path=save_to_path)

        self.api.pull_all()
