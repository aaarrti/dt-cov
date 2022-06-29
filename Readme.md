## Time series forecasting of COVID-19 deaths in BERLIN

### Install dependencies
    pip install -r requirements.txt

### Run with
    python3 src/main.py [--debug true] [--epochs <int>]

- `--debug` enables eager execution of TF, and disables XLA
- `--epochs` changes number of training epochs. default is 5
- Just run it, and CLI prompt will guide you through the choice of model and time window