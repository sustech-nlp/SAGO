# import torch
import numpy as np
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
import os
from data.utils import (
    load_hf_dataset,
    add_dataset_index,
    preprocess_pretraining_instance,
)


class CompletionDataset(Dataset):
    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        prefix_key="prompt",
        text_key="text",
        max_length=2048,
        predict_with_generate=False,
        insert_space=False,
    ):
        super(CompletionDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = load_hf_dataset(**hf_args)
        self.data = add_dataset_index(self.data)
        # if either key does not exist in dataset, it is taken as ""
        self.prefix_key = prefix_key
        self.text_key = text_key
        self.predict_with_generate = predict_with_generate
        self.insert_space = insert_space

    def __len__(self):
        return len(self.data)

    def _process_sample(self, prefix, text_content, index=-1):
        tokenized_data = preprocess_pretraining_instance(
            self.tokenizer,
            prefix,
            text_content,
            self.max_length,
            self.predict_with_generate,
            self.insert_space,
        )
        item_dct = {
            "input_ids": tokenized_data["input_ids"],
            "labels": tokenized_data["labels"],
            "attention_mask": tokenized_data["attention_mask"],
        }
        if index != -1:
            item_dct["index"] = index
        return item_dct

    def __getitem__(self, idx):
        pref = self.data[idx].get(self.prefix_key, "")
        text_content = self.data[idx].get(self.text_key, "")
        index = self.data[idx]["index"]
        item = self._process_sample(pref, text_content, index)
        return item


class PretrainingDataset(Dataset):
    def __init__(
        self, hf_args, template_args, tokenizer, text_key="text", max_length=2048
    ):
        super(PretrainingDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length        

        dataset = load_hf_dataset(**hf_args)

        def process_examples(examples):
            texts = [text + "\n\n" for text in examples[text_key]]
            return self.tokenizer(texts, add_special_tokens=False, return_tensors="np")

        tokenized_dataset = dataset.map(process_examples, batched=True)
        self.all_tokens = np.concatenate(tokenized_dataset["input_ids"])

    def __len__(self):
        return self.all_tokens.shape[0] // self.max_length

    def __getitem__(self, idx):
        chunk_tokens = self.all_tokens[idx * self.max_length:(idx + 1) * self.max_length]
        chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        return preprocess_pretraining_instance(
            self.tokenizer, "", chunk_text, self.max_length
        )