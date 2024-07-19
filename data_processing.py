from datasets import load_dataset
from transformers import T5Tokenizer

class DataProcessor:
    def __init__(self, model_name, max_length, num_procs):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.num_procs = num_procs

    def preprocess_function(self, examples):
        inputs = [f"assign tag: {title} {body}" for (title, body) in zip(examples['Title'], examples['Body'])]
        model_inputs = self.tokenizer(inputs, max_length=self.max_length, truncation=True, padding='max_length')

        cleaned_tag = [' '.join(''.join(tag.split('<')).split('>')[:-1]) for tag in examples['Tags']]
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(cleaned_tag, max_length=self.max_length, truncation=True, padding='max_length')

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def load_and_preprocess_data(self, train_file, valid_file):
        dataset_train = load_dataset('csv', data_files=train_file, split='train')
        dataset_valid = load_dataset('csv', data_files=valid_file, split='train')

        tokenized_train = dataset_train.map(self.preprocess_function, batched=True, num_proc=self.num_procs)
        tokenized_valid = dataset_valid.map(self.preprocess_function, batched=True, num_proc=self.num_procs)

        return tokenized_train, tokenized_valid, self.tokenizer
