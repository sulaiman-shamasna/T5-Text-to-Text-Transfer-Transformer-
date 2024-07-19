from transformers import T5ForConditionalGeneration, T5Tokenizer

def save_tokenizer(tokenizer, output_dir):
    tokenizer.save_pretrained(output_dir)

def load_model_and_tokenizer(model_path, tokenizer_path):
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer
