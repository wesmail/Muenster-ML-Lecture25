# Installing Core ML Python Packages

This guide walks you through installing key Python packages used in Lecture-1 tutorial for Machine Learning and data analysis:

- [`numpy`](https://numpy.org/)
- [`pandas`](https://pandas.pydata.org/)
- [`scikit-learn`](https://scikit-learn.org/stable/)
- [`matplotlib`](https://matplotlib.org/)
- [`seaborn`](https://seaborn.pydata.org/)

## Method 1: Installing with pip (System-wide or in a venv)

### ✅ Prerequisites
- Python 3.8+ installed
- `pip` available (`python -m ensurepip --upgrade` if unsure)

### Steps

#### 1. Install packages globally (⚠️ Not recommended because dependices may mix)

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Method 2: Recommended: Use a virtual environment  
  
### Create virtual environment
```bash
python -m venv ml-env
```
### Activate environment
#### On Linux/macOS:
```bash
source ml-env/bin/activate
```
#### On Windows:
```bash
ml-env \Scripts\activate
```
### Install packages
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```
### Deactivate when done
```bash
deactivate
```

## 🐍 Method 3: HIGHLY Recommended: Using micromamba (Fast, lightweight conda alternative)  

### ✅ Prerequisites

You need to Install [Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)  
Make sure micromamba is in your `PATH` (Follow instruction procedure in the webpage)  

### Steps:  
#### Create a new environment
```bash
micromamba create -n ml-lecture python=3.10 numpy pandas scikit-learn matplotlib seaborn -y
```
#### Activate the environment
```bash
micromamba activate ml-lecture
```

#### Optional: To install additional packages later:
```bash
micromamba install -n ml-lecture scipy jupyterlab -y
```

#### Once done working, you can deactivate the enviroment:  
```bash
micromamba deactivate
```

# Using VS Code to Run Jupyter Notebooks

## What is VS Code?

**Visual Studio Code (VS Code)** is a free, open-source, and powerful text editor developed by Microsoft. It's widely used for software development, data science, and machine learning because of its:

- Lightweight design
- Extensions for almost every language and framework
- Built-in Git support
- Debugging tools
- Jupyter notebook integration

---

## Running Jupyter Notebooks in VS Code

### ✅ Prerequisites

1. **Install VS Code**
   - Download: https://code.visualstudio.com/

2. **Install VS Code Extensions**
   - Open VS Code
   - Go to the Extensions panel (Ctrl+Shift+X)
   - Search and install:
     - `Python` (by Microsoft)
     - `Jupyter` (by Microsoft)

3. **Open the Jupyter Notebook and Enjoy**
---

# Running Jupyter Notebooks on Google Colab

Google Colab is a free, cloud-based Jupyter notebook environment that runs in your browser—no installation required. You can run Python code, use GPUs/TPUs, and access files stored in your Google Drive or uploaded manually.

---

## Steps to Run Your Notebook on Google Colab

### 1. Prerequisites

- A **Google account** (Gmail)
---

### 2. Open Google Colab

Go to: [https://colab.research.google.com](https://colab.research.google.com)

---

### 3. Upload the Notebook
- First you need to donload the Jupyter notebook locally on your PC or labptop, then  
- Click **File → Open Notebook**
- Go to the **Upload** tab
- Click **"Browse"** or drag and drop your `.ipynb` file
- Click **"Open"**

> Your notebook will now open in a Colab environment, ready to run.

---

### 4. 📁 Upload Data Files

If your notebook needs data:

1. In the left sidebar, click the **folder icon** (📁)
2. Click the **Upload** button (📤)
3. Select your data file(s) (e.g., `sample_pid.csv`) and upload it

The files will appear in `/content/`.
