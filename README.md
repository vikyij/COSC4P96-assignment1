# COSC 4P96 – Assignment 1  
## From Supervised to Semi-Supervised Learning

This repository contains the implementation for **Assignment 1** of COSC 4P96.  
The project builds a complete machine learning pipeline, progressing from **data preparation** to **supervised** and **semi-supervised learning**, with a strong emphasis on **reproducibility and fair evaluation**.

It is structured into five experimental stages:

- **Stage 1** – Data Preparation and Preprocessing  
- **Stage 2** – Supervised Baseline (10% labeled data)  
- **Stage 3** – Regularization (Dropout + Weight Decay)  
- **Stage 4** – Semi-Supervised Learning (Consistency Regularization)  
- **Stage 5** – Final Analysis (Upper Bound, Ablations, Statistics, Cost Analysis)
---

##  Environment Setup

### 1️⃣ Clone the Repository

```bash
git clone <https://github.com/vikyij/COSC4P96-assignment1.git>
cd COSC4P96-assignment1
```
---

### Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
# OR
.venv\Scripts\activate      # Windows

```
---

### Install Dependencies
```bash
pip install -r requirements.txt
```
---

## Stage 1 – Data Preparation
```bash
python data_prep_cosc4p96.py
```
---

## Stage 2 – Supervised Baseline (10% Labels)
```bash
python supervised_baseline.py
```
---

## Stage 3 – Regularization Experiments
```bash
python regularization.py
```
---

## Stage 4 – Semi-Supervised Learning
```bash
python semi_supervised.py
```
---

## Stage 5 – Final Analysis
```bash
python python stage5.py
```
---

## Project Structure

```text
COSC4P96-assignment1/
│
├── data/
│   ├── data_splits.npz
│   └── normalization_stats.pkl
│
├── docs/
│   └── stage1_analysis.txt
│
├── results/
│   ├── supervised.csv
│   ├── regularization.csv
│   ├── semi-supervised.csv
│   └── stage5_*.csv
│
├── src/
│   └── common.py
│
├── notebooks/
│    ├── data_prep_cosc4p96.py
│    ├── supervised_baseline.py
│    ├── regularization.py
│    ├── semi_supervised.py
│    ├── analysis_ablation.py
└── README.md


