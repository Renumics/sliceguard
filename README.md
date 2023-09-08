<p align="center"><a href="https://github.com/Renumics/sliceguard"><img src="https://github.com/Renumics/sliceguard/raw/main/static/img/sliceguard.svg" alt="Sliceguard Logo" height="80"/></a></p>
<h1 align="center">sliceguard</h1>
<p align="center">Detect problematic data slices in unstructured and structured data – fast.</p>

<p align="center">
 	<a href="https://pypi.org/project/sliceguard/"><img src="https://img.shields.io/pypi/pyversions/sliceguard" height="20"/></a>
 	<a href="https://pypi.org/project/sliceguard/"><img src="https://img.shields.io/pypi/wheel/sliceguard" height="20"/></a>
	<a href="https://sliceguard.readthedocs.io/en/latest/index.html"><img src="https://readthedocs.org/projects/sliceguard/badge/?version=latest&amp;style=flat" height="20"></a>
</p>

<p align="center"><img src="https://github.com/Renumics/sliceguard/raw/main/static/img/sliceguard_github.gif" width="100%"/><img src="https://github.com/Renumics/sliceguard/raw/main/static/img/dropshadow.png" width="100%"/></p>

## 🚀 Introduction

Sliceguard helps you to quickly discover **problematic data segments**. It supports structured data as well as unstructured data like images, text or audio. Sliceguard generates an **interactive report** with just a few lines of code:

```python
from sliceguard import SliceGuard

sg = SliceGuard()
issues = sg.find_issues(df, features=["image"])

sg.report()
```

## ⏱️ Quickstart

Install sliceguard by running `pip install sliceguard`.

Go straight to our quickstart examples for your use case:

* 🖼️ **[Unstructured Data (Images, Audio, Text)](https://github.com/Renumics/sliceguard/blob/main/examples/quickstart_unstructured_data.ipynb)** **–** **[🕹️ Interactive Demo](https://huggingface.co/spaces/renumics/sliceguard-unstructured-data)**
* 📈 **[Structured Data (Numerical, Categorical Variables)](https://github.com/Renumics/sliceguard/blob/main/examples/quickstart_structured_data.ipynb)** **–** **[🕹️ Interactive Demo](https://huggingface.co/spaces/renumics/sliceguard-structured-data)**
* 📊 **[Mixed Data (Contains Both)](https://github.com/Renumics/sliceguard/blob/main/examples/quickstart_mixed_data.ipynb)** **–** **[🕹️ Interactive Demo](https://huggingface.co/spaces/renumics/sliceguard-mixed-data)**

## 🗺️ Public Roadmap
We maintain a **[public roadmap](https://github.com/Renumics/sliceguard/blob/main/ROADMAP.md)** so you can follow along the development of this library.
