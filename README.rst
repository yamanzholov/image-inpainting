The aim of the project is to predict middle 16x16 segment of every test image.
To run the project first create virtual environment and install dependencies by running:
    python -m venv my_venv
    source my_env/bin/activate
    pip install -r requirements.txt

Each directory contains config file with necessary configurations to replicate the results.
To train the network run train.py script in needed directory.
Inference.py makes predictions on test set.