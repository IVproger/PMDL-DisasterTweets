# ZenML set up guide

1. **Set up the environment**
  ```bash
    # REPLACE <project-folder-path> with your project folder path
    # You can use .zshrc insted of .bashrc
    cd <project-folder-path>
    echo "export ZENML_CONFIG_PATH=$PWD/services/zenml" >> ~/.bashrc
    echo "export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=Yes" >> ~/.bashrc

    # Run the file
    source ~/.bashrc

    # Activate the virtual environment again
    source .venv/bin/activate
    ```

2. **Build the docker env for ZenML server**
    ```bash
    zenml up --docker
    ```

3. **ZenML commands**
    ```bash
    zenml up # run the zenml server

    zenml down # shut down the zenml server
    ```
