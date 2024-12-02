### Setup the environment

1. Python version should be greater than to 3.10.13. We do not guarantee that it will work in other versions.

2. Install dependencies:
    ```shell
    pip install -r requirements.txt
    ```

### Run with command line

```shell
python main.py --main Evaluate --data_file data/mulcrr/test.csv --system collaboration --system_config config/systems/collaboration/all_agents.json --task pr --rounds 1
```

### Run with browser

```shell
streamlit run web.py
```

Visit through `http://localhost:8501/`.

Please note that the systems utilizing open-source LLMs or other language models may require a significant amount of memory. These systems have been disabled on machines without CUDA support.
