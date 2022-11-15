# Installation Guide

```bash title="clone the repo"
git clone https://github.com/microsoft/pdearena
```

=== "`conda`"


    ```bash title="create and activate env"
    cd pdearena
    conda env create --file docker/environment.yml
    conda activate pdearena
    ```

    ```bash title="install this package"
    # install so the project is in PYTHONPATH
    pip install -e .
    ```

    If you also want to do data generation:

    ```bash
    pip install -e ".[datagen]"
    ```


=== "`docker`"


    ```bash title="build docker container"
    cd docker
    docker build -t pdearena .
    ```

    ```bash title="run docker container"
    cd pdearena
    docker run --gpus all -it --rm --user $(id -u):$(id -g) \
        -v $(pwd):/code -v /mnt/data:/data --workdir /code -e PYTHONPATH=/code \
        pdearena:latest
    ```


    !!! note

        - `--gpus all -it --rm --user $(id -u):$(id -g)`: enables using all GPUs and runs an interactive session with current user's UID/GUID to avoid `docker` writing files as root.
        - `-v $(pwd):/code -v /mnt/data:/data --workdir /code`: mounts current directory and data directory (i.e. the cloned git repo) to `/code` and `/data` respectively, and use the `code` directory as the current working directory.
