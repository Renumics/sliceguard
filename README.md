<p align="center"><a href="https://github.com/Renumics/sliceguard"><img src="static/img/spotlight.svg" alt="Gray shape shifter" height="60"/></a></p>
<h1 align="center">sliceguard</h1>
<p align="center">Detect problematic data slices in unstructured and structured data fast.</p>

<p align="center">
 	<a href="https://pypi.org/project/sliceguard/"><img src="https://img.shields.io/pypi/pyversions/sliceguard" height="20"/></a>
 	<a href="https://pypi.org/project/sliceguard/"><img src="https://img.shields.io/pypi/wheel/sliceguard" height="20"/></a>
</p>

## 🚀 Introduction

sliceguard is built to quickly discover problematic data segments in your data. It aims at supporting structured data as well as unstructured data like images, text or audio. However, it also tries to keep a simple interface hiding most of its functionality behind one simple *find_issues* function.

It also allows for interactive reporting and exploration of found data issues using [Renumics Spotlight](https://github.com/Renumics/spotlight).

## ⏱️ Quickstart

Install sliceguard by running `pip install sliceguard`.

Go straight to our quickstart examples for your use case:

* 🖼️ **[Unstructured data (images, audio, text)](examples/quickstart_unstructured_data.ipynb)**
* 📈 **[Structured data (numerical, categorical variables)](examples/quickstart_structured_data.ipynb)**
* 📊 **[Mixed data (data that contains both)](examples/quickstart_mixed_data.ipynb)**

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
- [x] Different interfaces from min_drop, min_support. Maybe n_slices and sort by criterion?
- [x] Support application without model (by training simple baseline model)
- [x] Improve normalization for mixed type runs e.g. embedding + one categorical or numeric variable.
- [x] Walthroughs for unstructured, structured and mixed data. Also, in depth tutorial explaining all the parameters.
- [ ] Soft Dependencies for embedding computation as torch dependencies are large
- [ ] Allow for model comparisons via intersection, difference, ...
- [ ] Robustify outlier detection algorithm. Probably better parameter choice.
- [ ] Interpretable features for images, audio, text. E.g., dark image, quiet audio, long audio, contains common word x, ...
- [ ] Generation of a summary report doing predefined checks
- [ ] "Supervised" clustering that incorporates classes, probabilities, metrics, not only features
- [ ] Data connectors for faster application on common data formats
- [ ] Support embedding generation for remote resources, e.g. audio/images hosted on webservers
- [ ] Improved explanations for found issues, e.g., via SHAP
