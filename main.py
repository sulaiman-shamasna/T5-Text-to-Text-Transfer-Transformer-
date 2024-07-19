from config import (
    MODEL_NAME,
    BATCH_SIZE,
    NUM_PROCS,
    EPOCHS,
    OUTPUT_DIR,
    MAX_LENGTH,
    TRAIN_FILE,
    VALID_FILE,
    LEARNING_RATE,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    LOGGING_STEPS,
    SAVE_STEPS,
    EVAL_STEPS,
    SAVE_TOTAL_LIMIT,
    DATALOADER_NUM_WORKERS
)
from data_processing import DataProcessor
from model import ModelSetup
from training import ModelTrainer
from utils import save_tokenizer

def main():
    data_processor = DataProcessor(MODEL_NAME, MAX_LENGTH, NUM_PROCS)
    tokenized_train, tokenized_valid, tokenizer = data_processor.load_and_preprocess_data(TRAIN_FILE, VALID_FILE)

    model_setup = ModelSetup(MODEL_NAME)
    model, device = model_setup.setup_model()

    trainer = ModelTrainer(
        model, tokenized_train, tokenized_valid, OUTPUT_DIR, EPOCHS, BATCH_SIZE, LEARNING_RATE,
        WARMUP_STEPS, WEIGHT_DECAY, LOGGING_STEPS, SAVE_STEPS, EVAL_STEPS, SAVE_TOTAL_LIMIT, DATALOADER_NUM_WORKERS
    )
    trainer.train_model()

    save_tokenizer(tokenizer, OUTPUT_DIR)

if __name__ == '__main__':
    main()
