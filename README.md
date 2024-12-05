# Analyzing and Improving RAG System Performance Using Statistical Methods

## Table of Contents

1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Components](#components)
   - [Word Token Analysis](#word-token-analysis)
   - [User Prompt Analysis](#user-prompt-analysis)
   - [RAG System Optimization](#rag-system-optimization)
   - [Input-Output Relationship Analysis](#input-output-relationship-analysis)
6. [Results](#results)
7. [Contributing](#contributing)
8. [License](#license)

## Introduction

This project aims to analyze and enhance the performance of a Retrieval-Augmented Generation (RAG) chatbot system using statistical methods from STAT 7010: Modern Data Mining. As Large Language Models (LLMs) and RAG systems become more prevalent, understanding and improving their performance is crucial.

## Project Structure
```

project_root/
│
├── data/
│ ├── raw/
│ └── processed/
│
├── src/
│ ├── word_token_analysis/
│ ├── user_prompt_analysis/
│ ├── rag_optimization/
│ ├── input_output_analysis/
│ └── utils/
│ ├── clustering.py
│ ├── dimensionality_reduction.py
│ ├── embeddings.py
│ └── visualisation.py
│
├── notebooks/
│ └── exploratory_analysis.ipynb
│
├── tests/
│ ├── test_utils.py
│ └── test_word_token_analysis.py
│
├── results/
│ └── word_token_analysis/
│ ├── dendrogram.png
│ ├── hierarchical_clusters.png
│ ├── kmeans_clusters.png
│ ├── pca_components_1_2.png
│ ├── pca_components_3_4.png
│ ├── tsne_2d.png
│ ├── tsne_3d_features_1_2.png
│ ├── tsne_3d_features_1_3.png
│ └── tsne_3d_features_2_3.png
│
├── main.py
├── requirements.txt
└── README.md

````

## Installation

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
````

## Usage

To run the full analysis:

```bash
python main.py
```

## Components

### Word Token Analysis

This component focuses on analyzing word embeddings using dimensionality reduction and clustering techniques.

#### Functionality

- Data Preparation: Generates word lists and creates embeddings
- Dimensionality Reduction: Applies t-SNE and PCA
- Clustering: Implements K-means and hierarchical clustering
- Visualization: Generates various plots for analysis

#### Output

The word token analysis generates several visualizations saved in the `results/word_token_analysis/` directory:

- Dendrogram of hierarchical clustering
- Hierarchical clustering visualization
- K-means clustering visualization
- PCA components plots
- t-SNE 2D and 3D plots

### User Prompt Analysis

[Placeholder for future development]

### RAG System Optimization

[Placeholder for future development]

### Input-Output Relationship Analysis

[Placeholder for future development]

## Results

The results of the analysis can be found in the `results/` directory. These include various plots and visualizations that provide insights into the structure and relationships of the word embeddings.

## Report and Presentation
The detailed report and presentation for this project can be found in the results/report/ and results/presentation/ directories, respectively:

- Report: report/report.pdf
- Presentation: results/presentation/presentation.pdf
