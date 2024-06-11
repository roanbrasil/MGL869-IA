# How to install and run `images_storage`

## Create virtual environment:

Go to the `images_storage` project root and run the following commands (assuming a Linux OS with Python and Pip installed):

```bash
$ pip install virtualenv
$ python<version> -m venv <virtual-environment-name>
$ source <virtual-environment-name>/bin/activate
```

## Install project dependencies

```bash
$ pip install -r requirements.txt
```

## Run the FastAPI server

```bash
$ fastapi dev main.py
```

# Running model files

## Train CNN1 Model

The model is stored in the `model/repository` folder with the `<model_name>.keras`.
The `files_root` is the path of the directory containing the folders `seg_train`, `seg_test` and `seg_pred`.

```bash
$ python model/train_cnn1_mode.py --model_name <model_name> --files_root <path_to_images_directory>
```

## Train CNN2 Model

The model is stored in the `model/repository` folder with the `<model_name>.keras`.
The `files_root` is the path of the directory containing the folders `seg_train`, `seg_test` and `seg_pred`.

```bash
$ python model/train_cnn2_mode.py --model_name <model_name> --files_root <path_to_images_directory>
```

## Train LinearSVC Model

The model is stored in the `model/repository` folder with the `<model_name>.pkl`.
The `files_root` is the path of the directory containing the folders `seg_train`, `seg_test` and `seg_pred`.

```bash
$ python model/train_linear_svc.py --model_name <model_name> --files_root <path_to_images_directory>
```