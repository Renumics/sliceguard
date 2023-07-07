<p align="center"><a href="https://github.com/Renumics/sliceguard"><img src="static/img/spotlight.svg" alt="Gray shape shifter" height="60"/></a></p>
<h1 align="center">sliceguard</h1>
<p align="center">Detect problematic data slices in unstructured and structured data fast.</p>

<p align="center">
 	<a href="https://pypi.org/project/sliceguard/"><img src="https://img.shields.io/pypi/pyversions/sliceguard" height="20"/></a>
 	<a href="https://pypi.org/project/sliceguard/"><img src="https://img.shields.io/pypi/wheel/sliceguard" height="20"/></a>
</p>

## 🚀 Introduction

sliceguard is built to quickly discover problematic data segments in your data. It aims at supporting structured data as well as unstructured data like images, text or audio. However, it also tries to keep a simple interface hiding most of its functionality after one simple *find_issues* function.

It also allows for interactive reporting and exploration of found data issues using [Renumics Spotlight](https://github.com/Renumics/spotlight).

## ⏱️ Quickstart

Install sliceguard by running `pip install sliceguard`.

Download the [Example Dataset](example_data.json).

Install the jiwer package for computing the word error rate metric using `pip install jiwer`

Get started by loading your first dataset and let sliceguard do its work:

```python
import pandas as pd
import numpy as np
from jiwer import wer
from sliceguard import SliceGuard

# Load the example data
df = pd.read_json("example_data.json")

# Define a metric function to evaluate your model
def wer_metric(y_true, y_pred):
    return np.mean([wer(s_y, s_pred) for s_y, s_pred in zip(y_true, y_pred)])

# Detect problematic data slices using the features age, gender and accent
sg = SliceGuard()
issue_df = sg.find_issues(
    df,
    ["age", "gender", "accent"],
    "sentence",
    "prediction",
    wer_metric,
    metric_mode="min"
)
sg.report()
```

For a more comprehensive tutorial check the following blog post on Medium:

[Evaluating automatic speech recognition models beyond global metrics — A tutorial using OpenAI’s Whisper as an example](https://medium.com/@daniel-klitzke/evaluating-automatic-speech-recognition-models-beyond-global-metrics-a-tutorial-using-openais-54b63c4dadbd)