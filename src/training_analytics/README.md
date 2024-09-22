# Training Analytics

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/training_analytics.git
    cd training_analytics
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements_dev.txt
    ```

## Usage

1. To run the data processing pipeline:
    ```sh
    python src/training_analytics/data.py
    ```

2. To run the Jupyter Notebook for modeling:
    ```sh
    jupyter notebook modelling.ipynb
    ```