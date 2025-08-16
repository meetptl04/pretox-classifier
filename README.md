# PRETOX Text Classifier

A text classification project using BioBERT to predict whether a given text is related to **PRETOX** or not.

## Dataset

This project uses the dataset from [Hugging Face: pretoxtm-dataset](https://huggingface.co/datasets/javicorvi/pretoxtm-dataset).

## Project Structure

```
BioBert_app/
├── data/                     # Dataset files
├── model/                    # Trained model checkpoints
├── venv/                     # Virtual environment
├── app.py                    # Streamlit application
├── requirements.txt          # Required Python packages
└── NLP_BioBert_PRETOX_REL.ipynb   # Notebook with training and evaluation code
```

## Installation

1. Clone the repository:

```
git clone https://github.com/meetptl04/pretox-classifier.git
cd pretox-classifier
```

2. Create and activate a virtual environment:

```
python -m venv venv
source venv/bin/activate     # Linux / macOS
venv\Scripts\activate       # Windows
```

3. Install dependencies:

```
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:

```
streamlit run app.py
```

* Enter a sentence in the input box to get the prediction (**PRETOX\_REL** or **NO\_PRETOX\_REL**).
* Type `exit` in the input box to quit.

## Model

* Uses [BioBERT](https://huggingface.co/dmis-lab/biobert-v1.1) for sequence classification.
* Trained on the PRETOX dataset.
* To retrain the model, use the provided notebook `NLP_BioBert_PRETOX_REL.ipynb`.
