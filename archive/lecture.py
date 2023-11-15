import pandas as pd

users = pd.read_csv("Users.csv",low_memory=False)
ratings=pd.read_csv("Ratings.csv",low_memory=False)
books=pd.read_csv("Books.csv",low_memory=False)

books = books.drop(columns=["Image-URL-S","Image-URL-M","Image-URL-L"])
print(users)

users=users.dropna()
ratings=ratings.dropna()
books=books.dropna()

df_final=pd.merge(books, ratings, on="ISBN", how="inner")
df_final=pd.merge(df_final, users, on="User-ID", how="inner")


print(df_final.columns)
print(df_final)

df_final.to_csv('datas.csv', index=False)