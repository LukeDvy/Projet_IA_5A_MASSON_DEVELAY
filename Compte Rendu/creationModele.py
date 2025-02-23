import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error

# Charger les données
data = pd.read_csv('archive/datas.csv', nrows=100000)
data = data[data['Book-Rating'] > 2]
data_fin = pd.read_csv('archive/datas.csv', nrows=100000)
data_fin = data_fin[data_fin['Book-Rating'] > 2]
data_fin = data_fin.drop(columns=["ISBN","Year-Of-Publication","Publisher","User-ID","Book-Rating","Location","Age","Nouns","Cluster","Cluster-Tag"])

# Prétraitement des données
user_enc = LabelEncoder()
data['User_ID'] = user_enc.fit_transform(data['User-ID'].values)

n_users = data['User_ID'].nunique()
n_books = data['ISBN'].nunique()

# Prétraitement de données pour la prédiction
location_enc = LabelEncoder()
data['Location'] = location_enc.fit_transform(data['Location'].astype(str))

author_enc = LabelEncoder()
data['Book-Author'] = author_enc.fit_transform(data['Book-Author'].astype(str))

book_enc = LabelEncoder()
data['Book-Title'] = book_enc.fit_transform(data['Book-Title'].astype(str))

genre_enc = LabelEncoder()
data['Cluster-Tag'] = genre_enc.fit_transform(data['Cluster-Tag'].astype(str))

# Diviser les données en ensemble d'entraînement et ensemble de test
train, test = train_test_split(data, test_size=0.2, random_state=42)

# Créer le modèle
def book_recommender_model():
    # Entrée pour l'âge de l'utilisateur
    age = Input(shape=(1,))
    a = Dense(1)(age)

    # Entrée pour le lieu de l'utilisateur
    location = Input(shape=(1,))
    l = Embedding(len(location_enc.classes_), 5)(location) # sert a convertir en vecteur => en données lisible par les ordinateurs
    l = Flatten()(l) # change en dimension 1, car les Dense ne peuvent que recup des données en 1 dim

    # Entrée pour l'auteur préféré
    author = Input(shape=(1,))
    au = Embedding(len(author_enc.classes_), 5)(author)
    au = Flatten()(au)

    # Entrée pour le titre du livre préféré
    book_title = Input(shape=(1,))
    bt = Embedding(len(book_enc.classes_), 5)(book_title)
    bt = Flatten()(bt)

    # Entrée pour le titre du livre préféré
    genre = Input(shape=(1,))
    gt = Embedding(len(genre_enc.classes_), 5)(genre)
    gt = Flatten()(gt)

    # Couche de concaténation
    x = Concatenate()([a, l, au, bt, gt])

    # Couches cachées
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    # Couche de sortie
    x = Dense(1)(x)

    model = Model(inputs=[age, location, author, book_title, genre], outputs=x)
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    return model

model = book_recommender_model()

# Entraîner le modèle
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)
train['Age'] = train['Age'].astype(float)
train['Location'] = train['Location'].astype(float)
train['Book-Author'] = train['Book-Author'].astype(float)
train['Book-Title'] = train['Book-Title'].astype(float)
train['Book-Rating'] = train['Book-Rating'].astype(float)
train['Cluster-Tag'] = train['Cluster-Tag'].astype(float)


history = model.fit([train['Age'], train['Location'],
                     train['Book-Author'], train['Book-Title'], train['Cluster-Tag']],
                     train['Book-Rating'],
                    validation_data=([test['Age'], test['Location'],
                                      test['Book-Author'], test['Book-Title'], test['Cluster-Tag']],
                                     test['Book-Rating'],),
                    epochs=5, batch_size=64, callbacks=[checkpoint])

# métrique permettant d'évaluer le modèle
test_predictions = model.predict(
    [test['Age'], test['Location'], test['Book-Author'], test['Book-Title'], test['Cluster-Tag']]
)
mse = mean_squared_error(test['Book-Rating'], test_predictions)
print(f'Mean Squared Error on Test Set: {mse}')



# Charger le meilleur modèle
model.load_weights('best_model.h5')



user_age = 30  # Remplacez cela par l'âge de l'utilisateur
user_location = 'victoria, british columbia, canada'  # Remplacez cela par le lieu de l'utilisateur
user_genre = 'Science Fiction'


user_author = 'Mitch Albom'  # Remplacez cela par l'auteur préféré de l'utilisateur
user_book_title = 'Life of Pi'  # Remplacez cela par le livre préféré de l'utilisateur

## peut etre parcourir tout les livres

user_location_encoded = location_enc.transform([user_location])[0]
user_author_encoded = author_enc.transform([user_author])[0]
user_book_title_encoded = book_enc.transform([user_book_title])[0]
user_genre_encoded = genre_enc.transform([user_genre])[0]


predicted_rating = model.predict([pd.Series([user_age]), pd.Series([user_location_encoded]),
                                  pd.Series([user_author_encoded]), pd.Series([user_book_title_encoded]), pd.Series([user_genre_encoded])])

print(predicted_rating) # affiche une note sur 10
compteur=0
for idx, row in data_fin.iterrows():
    user_author = str(row['Book-Author'])
    user_book_title = str(row['Book-Title'])

    user_author_encoded = author_enc.transform([user_author])[0]
    user_book_title_encoded = book_enc.transform([user_book_title])[0]
    predicted_rating = model.predict([pd.Series([user_age]), pd.Series([user_location_encoded]),
                                    pd.Series([user_author_encoded]), pd.Series([user_book_title_encoded]), pd.Series([user_genre_encoded])])

    print(predicted_rating) # affiche une note sur 10
    data_fin.loc[idx, "rating"] = predicted_rating
    compteur+=1
    if(compteur==10):
        break


data_fin = data_fin.sort_values(by="rating",ascending=False)
print(data_fin.head(10))
