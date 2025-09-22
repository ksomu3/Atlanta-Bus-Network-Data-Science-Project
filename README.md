# Atlanta-Bus-Network-Data-Science-Project

This project develops a regression model to predict bus travel time on selected low On-Time Performance (OTP) routes operated by MARTA (Metropolitan Atlanta Rapid Transit Authority). The model uses engineered features from multiple sources including weather, traffic, passenger activity, operator behavior, and vehicle maintenance. SHAP (SHapley Additive exPlanations) is used to interpret the model’s predictions. For more in depth context, please review the report. 

## Main Notebook

- **`Consolidated_data.ipynb`**: Contains data processing, feature engineering, model training, and SHAP analysis.

## How to Use

1. Download the full dataset and notebook: [Google Drive Folder](https://drive.google.com/drive/folders/1cQY5cjyrDEXZ-b20WAeYJzlw6TOnhOGP?usp=drive_link)
2. Open `Consolidated_data.ipynb` in Jupyter or Google Colab.
3. Ensure the following Python packages are installed:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, make_scorer
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ParameterGrid, cross_val_score
from tqdm import tqdm
shap.initjs()

shap.initjs()
