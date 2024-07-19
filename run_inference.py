from inference import Inference

def main():
    model_path = 'results_t5small/'
    tokenizer_path = 'results_t5small'
    data_dir = 'inference_data/'

    inference = Inference(model_path, tokenizer_path)
    inference.process_inference_data(data_dir)

if __name__ == '__main__':
    main()
