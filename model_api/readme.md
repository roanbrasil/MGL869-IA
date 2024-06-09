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

## Data initialization

The `files_root` is the path of the directory containing the folders `seg_train`, `seg_test` and `seg_pred`.

```bash
$ python model/data_initialize.py --files_root <path_to_images_directory>
```
