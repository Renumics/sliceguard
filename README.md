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

## 🔧 Use case-specific examples
* [Detecting issues in automatic speech recognition (ASR) models](examples/audio_issues_commonvoice_whisper.ipynb)
* [Detecting issues in audio datasets (condition monitoring)](examples/audio_issues_condition_monitoring_dcase.ipynb)
* [Selecting the best (and worst) generated images of Stable Diffusion](examples/stable_diffusion_evaluation.ipynb)


## 🗺️ Public Roadmap
- [x] Detection of problematic data slices
- [x] Basic explanation of found issues via feature importances
- [x] Limited embedding computation for images, audio, text
- [x] Extended embedding support, e.g., more embedding models and allow precomputed embeddings
- [x] Speed up embedding computation using datasets library
- [x] Improved issue detection algorithm, avoiding duplicate detections of similar problems and outliers influencing the segment detection
- [x] Support application on datasets without labels (outlier based)
- [x] Adaptive drop reference for datasets that contain a wide variety of data
- [x] Large data support for detection and reporting, e.g., 500k audio samples with transcriptions
- [ ] Support application without model (by training simple baseline model)
- [ ] Interpretable features for images, audio, text. E.g., dark image, quiet audio, long audio, contains common word x, ...
- [ ] Generation of a summary report doing predefined checks
- [ ] Allow for control features in order to account for expected variations when running checks
- [ ] Different interfaces from min_drop, min_support. Maybe n_slices and sort by criterion?
- [ ] Soft Dependencies for embedding computation as torch dependencies are large
- [ ] Extensive documentation and examples for common cases
- [ ] Data connectors for faster application on common data formats
- [ ] Support embedding generation for remote resources, e.g. audio/images hosted on webservers
- [ ] Improved explanations for found issues, e.g., via SHAP
