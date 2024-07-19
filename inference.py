import os
from transformers import T5ForConditionalGeneration, T5Tokenizer

class Inference:
    def __init__(self, model_path, tokenizer_path):
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)

    def do_correction(self, text):
        input_text = f"assign tag: {text}"
        inputs = self.tokenizer.encode(
            input_text,
            return_tensors='pt',
            max_length=256,
            padding='max_length',
            truncation=True
        )

        corrected_ids = self.model.generate(
            inputs,
            max_length=256,
            num_beams=5,
            early_stopping=True
        )

        corrected_sentence = self.tokenizer.decode(
            corrected_ids[0],
            skip_special_tokens=True
        )
        return corrected_sentence

    def process_inference_data(self, data_dir):
        for file in os.listdir(data_dir):
            file_path = os.path.join(data_dir, file)
            with open(file_path, 'r') as f:
                sentence = f.read()
                corrected_sentence = self.do_correction(sentence)
                print(f"QUERY: {sentence}\nTAGS: {corrected_sentence}")
