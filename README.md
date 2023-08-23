<p align="center"><a href="https://github.com/Renumics/sliceguard"><img src="static/img/spotlight.svg" alt="Gray shape shifter" height="60"/></a></p>
<h1 align="center">sliceguard</h1>
<p align="center">Detect problematic data slices in unstructured and structured data – fast.</p>

<p align="center">
 	<a href="https://pypi.org/project/sliceguard/"><img src="https://img.shields.io/pypi/pyversions/sliceguard" height="20"/></a>
 	<a href="https://pypi.org/project/sliceguard/"><img src="https://img.shields.io/pypi/wheel/sliceguard" height="20"/></a>
</p>

## 🚀 Introduction

sliceguard is built to quickly **discover problematic data segments** in your data. It aims at supporting structured data as well as unstructured data like images, text or audio. However, it also tries to keep a **simple interface** hiding most of its functionality behind one simple *find_issues* function:

```python
from sliceguard import SliceGuard

sg = SliceGuard()
issues = sg.find_issues(df, features=["image"])

sg.report()
```

It also allows for **interactive reporting and exploration** of found data issues using **[Renumics Spotlight](https://github.com/Renumics/spotlight)**.

## ⏱️ Quickstart

Install sliceguard by running `pip install sliceguard`.

Go straight to our quickstart examples for your use case:

* 🖼️ **[Unstructured data (images, audio, text)](examples/quickstart_unstructured_data.ipynb)**
* 📈 **[Structured data (numerical, categorical variables)](examples/quickstart_structured_data.ipynb)**
* 📊 **[Mixed data (contains both)](examples/quickstart_mixed_data.ipynb)** **–** **[Interactive Demo](https://huggingface.co/spaces/renumics/sliceguard-mixed-data)**

## 🔧 Use case-specific examples
* 🗣️ **[Detecting issues in automatic speech recognition (ASR) models](examples/audio_issues_commonvoice_whisper.ipynb)**
* 🛠️ **[Detecting issues in audio datasets (condition monitoring)](examples/audio_issues_condition_monitoring_dcase.ipynb)**
* 🌆 **[Selecting the best (and worst) generated images of Stable Diffusion](examples/stable_diffusion_evaluation.ipynb)**


## 🗺️ Public Roadmap
We maintain a **[public roadmap](ROADMAP.md)** so you can follow along the development of this library.
