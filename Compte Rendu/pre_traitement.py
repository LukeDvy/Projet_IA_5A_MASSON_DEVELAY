import pandas as pd

users = pd.read_csv("Compte Rendu/archive/Users.csv", low_memory=False)
ratings = pd.read_csv("Compte Rendu/archive/Ratings.csv", low_memory=False)
books = pd.read_csv("Compte Rendu/archive/Books.csv", low_memory=False)

books = books.drop(columns=["Image-URL-S", "Image-URL-M", "Image-URL-L"])

users = users.dropna()
users = users.drop_duplicates(
    subset="User-ID"
)  # drop des potentiels id en commun sur la clé primaire

ratings = ratings.dropna()
ratings = ratings.drop_duplicates(
    subset=["User-ID", "ISBN"], keep="last"
)  # verifie qu'un lecteur a noté une seule fois un livre = drop des potentiels id en commun sur la clé primaire

books = books.dropna()
books = books.drop_duplicates(
    subset="ISBN"
)  # drop des potentiels id en commun sur la clé primaire

df_final = pd.merge(books, ratings, on="ISBN",how="inner")
df_final = pd.merge(df_final, users, on="User-ID", how="inner")
df_final['Age'] = df_final['Age'].round().astype(int)

df_final.to_csv("archive/datas.csv", index=False)
