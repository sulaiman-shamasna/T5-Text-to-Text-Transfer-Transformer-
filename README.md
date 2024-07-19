# T5-Text-to-Text-Transfer-Transformer

This repository provides a step-by-step approach to fine-tune the T5 (**T**ext-**T**o-**T**ext **T**ransfer **T**ransformer) model.

## Setting up Environment

To work with this project, follow these steps:
### ***Prepare CUDA***

You can check this [article](https://medium.com/@kajhanan.1999/setting-up-pytorch-with-cuda-on-windows-11-for-gpu-deeplearning-2023-december-de1da94ddb9e) for details. Or, simply:

1. Download and install [NVIDIA Driver](https://www.nvidia.com/Download/index.aspx).
2. Download and install [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive) or [CUDA Toolkit 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local).

### ***Follow one of the following approaches***:
- With *Docker Container*.
- On local *Virtual Environment*.

### ***Docker Container***
To avoid package inconsistancies, it's preferred to work in a docker container. The following is an illustrative step-by-step approach how this is achieved.
1. [Download](https://docs.docker.com/desktop/install/windows-install/) and install [Docker Desktop](https://docs.docker.com/desktop/install/windows-install/) (in case you're working on *Windows* machine). Then make sure it's running.
2. Run a *docker container*. There's two possibilities
    * ***Pull an existing *docker image*.***
        1. Pull an exisiting docker image, e.g., the image `pytorch:1.12.1-cuda11.3-cudnn8-runtime`:
            ```bash
            docker pull pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
            ```
        2. Run the docker container:
            ```bash
            docker run --gpus all -v C:\\Users\\sulai\\Documents\\GH-Projects\\T5-Text-to-Text-Transfer-Transformer-:/NAME --name NAME -it pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
            ```
    * ***Create and run your *docker image*.***
        1. Prepare your Dockerfile, which might look something like this:
            ```Dockerfile
            # Use the official Python 3.10.9 image as the base image
            FROM python:3.10.9

            RUN apt-get update && apt-get install -y \
                git \
                wget \
                && rm -rf /var/lib/apt/lists/*

            RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html

            WORKDIR /X4
            COPY . .

            CMD ["bash"]
            ```
        2. Navigate to the same directory as of *Dockerfile*, and build the *docker image*.
            ```bash
            docker build -t IMAGE_NAME:3.10.9 .
            ```
        3. Run a container from the newly built image with *GPU* support and volume mapping:
            ```bash
            docker run --gpus all -v C:\\Users\\...\\PATH_TO_THE_PROJECT\\..\\T5-Text-to-Text-Transfer-Transformer-:/X4 --name X4 -it my_pytorch_python:3.10.9
            ```
        ![run_docker](https://github.com/sulaiman-shamasna/T5-Text-to-Text-Transfer-Transformer-/blob/main/images/run_docker.png)
3. Install dependencies inside the container once it is running.
    ```bash
    pip install -r requirements.txt
    ```
4. Install torch with cuda.
    ```bash
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu121
    ```
Be aware here and avoid packages as well as available cuda version and nvidia driver inconsistancies, pleas run the command in the bas ```nvidia-smi```, and check these details:

    +-----------------------------------------------------------------------------------------+
    | NVIDIA-SMI 560.27                 Driver Version: 560.70         CUDA Version: 12.6     |
    |-----------------------------------------+------------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
    |                                         |                        |               MIG M. |
    |=========================================+========================+======================|
    |   0  NVIDIA GeForce RTX 4090 ...    On  |   00000000:01:00.0 Off |                  N/A |
    | N/A   77C    P0            104W /  150W |   16032MiB /  16376MiB |    100%      Default |
    |                                         |                        |                  N/A |
    +-----------------------------------------+------------------------+----------------------+

    +-----------------------------------------------------------------------------------------+
    | Processes:                                                                              |
    |  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
    |        ID   ID                                                               Usage      |
    |=========================================================================================|
    |  No running processes found                                                             |
    +-----------------------------------------------------------------------------------------+


### ***Local Environment***

0. **Setup CUDA**
    You can check this [article](https://medium.com/@kajhanan.1999/setting-up-pytorch-with-cuda-on-windows-11-for-gpu-deeplearning-2023-december-de1da94ddb9e) for details. Or, simply:
    1. Download and install [NVIDIA Driver](https://www.nvidia.com/Download/index.aspx).
    2. Download and install [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive) or [CUDA Toolkit 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local). 

    ```bash
    - pip install torch --index-url https://download.pytorch.org/whl/cu126
    - pip install torchvision==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
    ```

    ```bash
    pip install torch==1.12.1+cu116 torchaudio==0.12.1+cu116 torchvision==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
    ```
    * Pull the docker image:
        ```bash
        docker pull pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
        ```
    * Run the docker container:
        ```bash
        docker run --gpus all -v C:\\Users\\sulai\\Documents\\GH-Projects\\T5-Text-to-Text-Transfer-Transformer-:/X4 --name X4 -it pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
        ```
    * Install the requirements:
        ```bash
        pip install -r requirements.txt
        ```
    * Then enable torch with cuda by installing:
        ```bash
        pip install torch==1.12.1+cu116 torchaudio==0.12.1+cu116 torchvision==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
        ```

1. **Clone the repository:**
    ```bash
    git clone https://github.com/sulaiman-shamasna/T5-Text-to-Text-Transfer-Transformer-.git
    ```
    
2. **Set up Python environment:**
    - Ensure you have **Python 3.10.X** or higher installed.
    - Create and activate a virtual environment:
      - For *Windows* (using Git Bash):
        ```bash
        source env/Scripts/activate
        ```
      - For *Linux* and *macOS*:
        ```bash
        source env/bin/activate
        ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4. **Datasets**
    To be able to train, you must get the dataset, this is done by running the following command in the bash: 

      - wget:
        ```bash
        wget "https://www.dropbox.com/scl/fi/525gv6tmdi3n32mipo6mr/input.zip?rlkey=5jdsxahphk2ped5wxbxnv0n4y&dl=1" -O input.zip
        ```
  
    OR 

      - curl:
        ```bash
        curl "https://www.dropbox.com/scl/fi/525gv6tmdi3n32mipo6mr/input.zip?rlkey=5jdsxahphk2ped5wxbxnv0n4y&dl=1" -O input.zip
        ```
  
    Then, unzip it:

    ```bash
    unzip input.zip
    ```

5. **Training:**
    To run the training pipeline:
    ```bash
    python main.py
    ```

6. **Inference:**
    To run the inference script, you need to have the inference dataset first, to do this, please run the command on git bash:

    ```bash
    !wget "https://www.dropbox.com/scl/fi/9brsjizymq5zvqi7hff09/inference_data.zip?rlkey=ukmdy5egmdld80r5hhmsja78v&dl=1" -O inference_data.zip

    ```
    Then, unzip it:
    ```bash
    !unzip inference_data.zip
    ```

    In so doing, you have now the dataset for the inference! To run it, please use the command:

    ```bash
    python inference.py
    ```

Congratulations! You've just finished finetuning T5 model.

