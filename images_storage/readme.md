# How to install and run `model_api`

## Create virtual environment:

Go to the `model_api` project root and run the following commands (assuming a Linux OS with Python and Pip installed):

```bash
$ pip install virtualenv
$ python<version> -m venv <virtual-environment-name>
$ source <virtual-environment-name>/bin/activate
```

## Install project dependencies

```bash
$ pip install -r requirements.txt
```

## Make Django migrations

```bash
$ python images_storage/manage.py migrate
```

## Create superuser

```bash
$ python images_storage/manage.py createsuperuser
```

## Run the Django Server

```bash
$ python images_storage/manage.py runserver
```

## Delete all images from database

```bash
$ python images_storage/manage.py delete_all_images
```