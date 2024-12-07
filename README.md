# LetsWatch: Movie Recommendation System

## Overview
LetsWatch is a simple movie recommendation system that provides personalized movie recommendations based on the user's watched movie dataset. The system uses features such as genre, movie duration, and release year to predict and recommend top-rated movies from the IMDb Top 1000 movies dataset.

This project leverages machine learning (Random Forest Regressor) and Streamlit to provide an interactive and easy-to-use interface for users.

---

## Features
- **Upload Personal Dataset**: Users can upload their watched movie dataset (in `.csv` or `.xlsx` format).
- **Feature Selection**: Users can choose the features they want to base the recommendations on (e.g., Genre, Year, Duration).
- **Dynamic Recommendations**: A slider allows users to select the number of recommendations (1–10).
- **Predictions**: Uses a trained Random Forest Regressor model to predict the ratings for unwatched movies.
- **Top Recommendations**: Displays a ranked list of movie recommendations.

---

## Prerequisites
1. **Python 3.x**
2. Required Python libraries:
    - `pandas`
    - `scikit-learn`
    - `streamlit`
    - `openpyxl` (for `.xlsx` file handling)
3. **IMDb Top 1000 Movies Dataset**:
    - Ensure the file `imdb_top_1000.csv` is available in the project directory.

---

## Installation
1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/letswatch.git
    cd letswatch
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## Dataset Format
### **Watched Movies Dataset**
Your uploaded dataset must include the following columns:
- `Title`: Movie title
- `Genre`: Genres of the movie (comma-separated, e.g., "Action, Comedy")
- `Duration`: Movie duration in minutes
- `Year`: Release year of the movie
- `Rating`: Your personal rating for the movie

### **IMDb Dataset**
Ensure the file `imdb_top_1000.csv` includes the following columns:
- `Title`: Movie title
- `Genre`: Genres of the movie (comma-separated)
- `Duration`: Movie duration in minutes
- `Year`: Release year of the movie
- `IMDB_Rating`: IMDb rating of the movie

---

## Usage
1. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

2. Open the URL displayed in your terminal (usually `http://localhost:8501`).

3. Upload your personal watched movie dataset.

4. Select the features to base the recommendations on:
    - **Genre**: Recommends based on the genres you prefer.
    - **Duration**: Normalizes movie durations for recommendations.
    - **Year**: Normalizes release years for recommendations.

5. Use the slider to select the number of recommendations (1–10).

6. View the recommended movies in a table format.

---

## Example
1. Upload a `.csv` file with the following structure:
    | Title           | Genre               | Duration | Year | Rating |
    |-----------------|---------------------|----------|------|--------|
    | Inception       | Action, Sci-Fi     | 148      | 2010 | 9      |
    | Titanic         | Drama, Romance     | 195      | 1997 | 8      |

2. Select features such as **Genre**, **Year**, and **Duration**.

3. Choose the number of recommendations using the slider.

4. Get a table of top movie recommendations.

---

## Output
The output displays the top recommended movies, sorted by predicted ratings, in the following format:
| Title            | Genre               | Year | Duration | IMDB Rating |
|------------------|---------------------|------|----------|-------------|
| The Dark Knight  | Action, Crime, Drama| 2008 | 152      | 9.0         |
| Interstellar     | Adventure, Sci-Fi   | 2014 | 169      | 8.6         |

---

## How It Works
1. **Genre Encoding**:
    - Splits genres into individual columns with binary values (1: Associated, 0: Not associated).
    
2. **Feature Normalization**:
    - Normalizes `Year` and `Duration` values between 0 and 1 using `MinMaxScaler`.

3. **Model Training**:
    - Trains a `RandomForestRegressor` model using the user's watched movie dataset.

4. **Prediction**:
    - Predicts ratings for the unwatched movies in the IMDb dataset.

5. **Recommendation**:
    - Ranks movies based on predicted ratings and excludes already-watched movies.

---

## Customization
- You can adjust the `RandomForestRegressor` parameters (e.g., `n_estimators`, `max_depth`) for better performance.
- Modify the IMDb dataset to include additional movies.

---

## Known Issues
- Ensure that the columns in your dataset are named exactly as specified.
- The IMDb dataset (`imdb_top_1000.csv`) must be preprocessed to ensure consistent formatting.

---

## License
This project is licensed under the MIT License.

---

## Acknowledgments
- IMDb Top 1000 dataset for providing the base movie information.
- Scikit-learn and Streamlit for enabling machine learning and interactive UI development.
