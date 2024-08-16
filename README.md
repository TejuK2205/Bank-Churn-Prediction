# Bank-Churn-Prediction

Welcome to the Bank Churn Predictor Dashboard! This project provides an interactive web-based dashboard for analyzing and predicting customer churn in a banking context. Built with Dash, Plotly, and Python, the dashboard includes various visualizations and model predictions to help understand and mitigate customer churn.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This dashboard allows users to:

- Visualize and explore basic and advanced statistics related to customer data.
- Predict customer churn using a pre-trained Support Vector Machine (SVM) model.
- Evaluate the performance of the SVM model with metrics such as confusion matrix and accuracy score.
- Analyze feature importance and other statistical aspects of the data.

## Features

- **Basic Statistics Page**: 
  - Interactive scatter, bar, density, and line plots.
  - Dropdowns and sliders for dynamic data visualization.
  
- **Advanced Statistics Page**:
  - Violin, parallel coordinates, ridge, bubble, and Sankey plots.
  - Choropleth map for geographical data visualization.

- **Model Prediction Page**:
  - Input fields for customer attributes to get churn predictions.
  - Display of model performance metrics like confusion matrix and accuracy score.
  - Feature importance visualization.

## Installation

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/yourusername/bank-churn-prediction.git
    ```

2. **Navigate to the Project Directory**:

    ```bash
    cd bank-churn-prediction
    ```

3. **Install Dependencies**:

    Ensure you have Python 3.x installed. Then, install the required packages using pip:

    ```bash
    - pandas
    - plotly
    - dash
    - scikit-learn
    - joblib
    - matplotlib
    - numpy
    ```

4. **Download the Dataset and Model**:

    Place your `project_dataset.csv` and `churn_model.pkl` files in the project directory.

## Usage

1. **Run the Dash Application**:

    Execute the following command to start the Dash app:

    ```bash
    python final_dash.py
    ```

2. **Access the Dashboard**:

    Open your web browser and navigate to `http://127.0.0.1:8050/` to view the dashboard.

## Code Structure

- `final_dash.py`: Main application file where the Dash app is defined and callbacks are set up.
- `project_dataset.csv`: Dataset used for analysis and model predictions.
- `churn_model.pkl`: Pre-trained SVM model for churn prediction.

