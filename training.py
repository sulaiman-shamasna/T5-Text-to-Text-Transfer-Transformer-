from transformers import TrainingArguments, Trainer

class ModelTrainer:
    def __init__(self, model, train_dataset, valid_dataset, output_dir, epochs, batch_size, learning_rate, warmup_steps, weight_decay, logging_steps, save_steps, eval_steps, save_total_limit, dataloader_num_workers):
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.output_dir = output_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.save_total_limit = save_total_limit
        self.dataloader_num_workers = dataloader_num_workers

    def train_model(self):
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=self.warmup_steps,
            weight_decay=self.weight_decay,
            logging_dir=self.output_dir,
            logging_steps=self.logging_steps,
            evaluation_strategy='steps',
            save_steps=self.save_steps,
            eval_steps=self.eval_steps,
            load_best_model_at_end=True,
            save_total_limit=self.save_total_limit,
            report_to='tensorboard',
            learning_rate=self.learning_rate,
            fp16=True,
            dataloader_num_workers=self.dataloader_num_workers
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset,
        )

        trainer.train()
        return trainer
