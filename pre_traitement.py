import pandas as pd

users = pd.read_csv("archive/Users.csv",low_memory=False)
ratings=pd.read_csv("archive/Ratings.csv",low_memory=False)
books=pd.read_csv("archive/Books.csv",low_memory=False)

books = books.drop(columns=["Image-URL-S","Image-URL-M","Image-URL-L"])
print(users)

users=users.dropna()
users=users.drop_duplicates(subset="User-ID")

ratings=ratings.dropna()
ratings=ratings.drop_duplicates(subset=["User-ID","ISBN"], keep="last") #verifie qu'un lecteur a noté une seule fois un livre

books=books.dropna()
books=books.drop_duplicates(subset="ISBN")


df_final=pd.merge(books, ratings, on="ISBN", how="inner")
df_final=pd.merge(df_final, users, on="User-ID", how="inner")
print(df_final)

#df_final=df_final.drop_duplicates(subset=['Book-Title', 'Book-Author', 'Year-Of-Publication','Publisher'], keep="last"), nous avons remarqué qu'il y a des livres en doublons, publié par plusieurs publisher, cependant ce n'est pas un réelle intérêt à supprimer ces doublons

print(df_final.columns)
print(df_final)

#print(df_final[df_final["Book-Title"]=="The Way Things Work: An Illustrated Encyclopedia of Technology"])

df_final.to_csv('archive/datas.csv', index=False)
