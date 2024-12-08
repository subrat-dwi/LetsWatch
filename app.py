import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import streamlit as st

# function to recommend movies
def recommend_movies(df, imdb_file, features, recommend_no):
    # to seperate different genres as into different columns
    if "Genre" in features:
        df["Genre"] = df["Genre"].apply(lambda x: x.split(", "))
        # assigning binary values, 1 : genre associated with movie, 0 : genre not associated
        mlb = MultiLabelBinarizer()
        genre_encoded = mlb.fit_transform(df["Genre"])
        genre_names = mlb.classes_
    
        genre_df = pd.DataFrame(genre_encoded, columns=genre_names)
        df = pd.concat([df, genre_df], axis=1)

    # normlising duration and year of release of movies between 0 to 1
    scaler = MinMaxScaler()
    if "Duration" in features:
        df["Normalised Duration"] = scaler.fit_transform(df[["Duration"]])
    if "Year" in features:
        df["Normalised Year"] = scaler.fit_transform(df[["Year"]])

    feature_cols = []
    if "Genre" in features_select:
        feature_cols.extend(genre_df.columns)
    if "Year" in features_select:
        feature_cols.append("Normalised Year")
    if "Duration" in features_select:
        feature_cols.append("Normalised Duration")

    X = df[feature_cols]
    y = df["Rating"]
    # partitioning data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # testing the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # reading dataset of unwatched movies
    new_df = pd.read_csv(imdb_file)
    # preprocessing genre
    if "Genre" in features:
        new_df["reGenre"] = new_df["Genre"].apply(lambda x: x.split(", "))
    
        new_genre_encoded = mlb.fit_transform(new_df["reGenre"])
        new_genre_names = mlb.classes_
    
        new_genre_df = pd.DataFrame(new_genre_encoded, columns=new_genre_names)
        new_df = pd.concat([new_df, new_genre_df], axis=1)
    
        for genre in genre_names:  # 'genre_names' from the training phase
            if genre not in new_genre_df.columns:
                new_genre_df[genre] = 0  # add missing genre columns with 0
        new_genre_df = new_genre_df[genre_names]
    # preprocessing duration and year of release
    new_df['reDuration'] = new_df['Duration'].str.extract('(\d+)').astype(int)
    new_df['reYear'] = new_df['Year'].str.extract('(\d+)').astype(float)

    new_df["Normalised Duration"] = scaler.fit_transform(new_df[["reDuration"]])
    new_df["Normalised Year"] = scaler.fit_transform(new_df[["reYear"]])
    # using the trained model to predict rating for unwatched movies
    unwatched_X = new_df[feature_cols]
    new_df['Predicted Rating'] = model.predict(unwatched_X)

    recommended_movies = new_df.sort_values(by="Predicted Rating", ascending=False)
    recommended_movies = recommended_movies[~recommended_movies['Title'].isin(df['Title'])]
    top_recommendations = recommended_movies.head(recommend_no)
    # to rename column IMDB_Rating
    top_recommendations["IMDB Rating"] = top_recommendations["IMDB_Rating"]

    return top_recommendations[['Title', 'Genre', 'Year', 'Duration','IMDB Rating']]


# Streamlit App
st.title("LetsWatch (Movie Recommendation System)")
st.write("Upload your watched movie dataset to get personalized recommendations.")
st.write("Must contain these features : Title, Genre, Duration, Year, Rating (your personal rating of movie)")
st.write("Try out these [Sample Datasets](https://github.com/subrat-dwi/LetsWatch/tree/main/Sample%20Datasets) to get recommendations.")
uploaded_file = st.file_uploader("Upload your watched movies dataset:", type=["xlsx", "csv"])
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        features_select = st.multiselect("Select the features to base recommendations on : ",
            options=["Genre", "Duration", "Year"],
            default=["Genre", "Duration", "Year"])


        imdb_file = "imdb_top_1000.csv"
        
        slider_value = st.slider(label="No. of recommendations : ",
                                value=5,
                                min_value=1,
                                max_value=10)
        recommendations = recommend_movies(df, imdb_file, features_select, slider_value)
        st.write("Recommended Movies:")
        st.table(recommendations)
    except Exception as e:
        st.error(f"An error occurred: {e}")
