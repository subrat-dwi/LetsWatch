import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import streamlit as st

#df = pd.read_excel("user_movie_dataset.xlsx")
def recommend_movies(df, imdb_file):
    print(df.head())
    
    df["Genre"] = df["Genre"].apply(lambda x: x.split(", "))
    
    mlb = MultiLabelBinarizer()
    genre_encoded = mlb.fit_transform(df["Genre"])
    genre_names = mlb.classes_
    
#print(genre_names)

    genre_df = pd.DataFrame(genre_encoded, columns=genre_names)
    df = pd.concat([df, genre_df], axis=1)
    #print(df.head())
    
    scaler = MinMaxScaler()
    df["Normalised Duration"] = scaler.fit_transform(df[["Duration"]])
    df["Normalised Year"] = scaler.fit_transform(df[["Year"]])
    #df = df.drop(["Genre"], axis=1)
    #print(df.head())
    
    X = pd.concat([genre_df, df[["Normalised Year","Normalised Duration"]]], axis=1)
    y = df["Rating"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    
    # Predict on test data
    y_pred = model.predict(X_test)
    
    # Calculate and print MSE
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)
    
    #required_cols = ["Title","Genre","Duration","Year","IMDB_Rating"]
    new_df = pd.read_csv(imdb_file)
    #new_df = new_df[~new_df['Title'].isin(df['Title'])]
    #print(new_df.head())
    
    new_df["Genre"] = new_df["Genre"].apply(lambda x: x.split(", "))
    
    mlb = MultiLabelBinarizer()
    new_genre_encoded = mlb.fit_transform(new_df["Genre"])
    new_genre_names = mlb.classes_
    
    #print(new_genre_names)
    
    new_genre_df = pd.DataFrame(new_genre_encoded, columns=new_genre_names)
    new_df = pd.concat([new_df, new_genre_df], axis=1)
    #new_df
    
    new_df['Duration'] = new_df['Duration'].str.extract('(\d+)').astype(int)
    new_df['reYear'] = new_df['Year'].str.extract('(\d+)').astype(float)
    
    scaler = MinMaxScaler()
    new_df["Normalised Duration"] = scaler.fit_transform(new_df[["Duration"]])
    new_df["Normalised Year"] = scaler.fit_transform(new_df[["reYear"]])
    #new_df =new_df.drop(["Year of Release"], axis=1)
    
    #print(new_df.head())
    
    for genre in genre_names:  # `genre_names` from the training phase
        if genre not in new_genre_df.columns:
            new_genre_df[genre] = 0  # Add missing genre columns with 0
    new_genre_df = new_genre_df[genre_names]
    
    unwatched_X = pd.concat([new_genre_df, new_df[['Normalised Year','Normalised Duration']]], axis=1)
    new_df['Predicted Rating'] = model.predict(unwatched_X)
    #new_df
    
    recommended_movies = new_df.sort_values(by="Predicted Rating", ascending=False)
    recommended_movies = recommended_movies[~recommended_movies['Title'].isin(df['Title'])]
    #recommended_movies
    top_recommendations = recommended_movies.head()
    print(top_recommendations[['Title', 'Genre', 'Year', 'Duration','IMDB_Rating']])
    return top_recommendations[['Title', 'Genre', 'Year', 'Duration','IMDB_Rating']]


# Streamlit App
st.title("Movie Recommendation System")
st.write("Upload your watched movie dataset (Excel or CSV) to get personalized recommendations.")

uploaded_file = st.file_uploader("Upload your watched movies dataset:", type=["xlsx", "csv"])
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        imdb_file = "imdb_top_1000.csv"  # Ensure this file is available
        recommendations = recommend_movies(df, imdb_file)

        st.write("Recommended Movies:")
        st.table(recommendations)
    except Exception as e:
        st.error(f"An error occurred: {e}")
