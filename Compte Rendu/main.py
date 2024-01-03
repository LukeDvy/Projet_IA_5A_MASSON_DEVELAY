import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from keras.callbacks import ModelCheckpoint
from flask import Flask, render_template, request

app = Flask(__name__)

df = pd.read_csv('archive/final_datas.csv')


user_age_list = df.groupby('Age').agg(list).to_dict()['book-title']
user_location_list=df.groupby('Location').agg(list).to_dict()['book-title']
user_genre_list=df.groupby('Cluster-Tag').agg(list).to_dict()['book-title']
user_author_list=df.groupby('Book-Author').agg(list).to_dict()['book-title']
user_book_title_list=df.groupby('Book-Title').agg(list).to_dict()


@app.route('/')
def index():
    return render_template('index.html',user_age_list=user_age_list,user_location_list=user_location_list,user_genre_list=user_genre_list,user_author_list=user_author_list,user_book_title_list=user_book_title_list)

# @app.route('/', methods=['POST'])
def test():
    user_age = request.form['user_age']
    return user_age

@app.route('/', methods=['POST'])
def search():


    user_age = request.form['user_age']
    user_location = request.form['user_location']
    user_genre = request.form['user_genre']
    user_author = request.form['user_author']
    user_book_title = request.form['user_book_title']

    if user_age == "" or user_location == "" or user_genre == "" or user_author == "" or user_book_title == "":
        return "Failed, empty value"



    # Charger les données
    data = pd.read_csv('archive/final_datas.csv', nrows=10000)
    data_fin = pd.read_csv('archive/final_datas.csv', nrows=10000)
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


    # Charger le meilleur modèle
    model.load_weights('best_model.h5')

    # user_age = 30  # Remplacez cela par l'âge de l'utilisateur
    # user_location = 'victoria, british columbia, canada'  # Remplacez cela par le lieu de l'utilisateur
    # user_genre = 'Science Fiction'


    # user_author = 'John Wyndham'  # Remplacez cela par l'auteur préféré de l'utilisateur
    # user_book_title = 'The Day of the Triffids'  # Remplacez cela par le livre préféré de l'utilisateur

    user_location_encoded = location_enc.transform([user_location])[0]
    user_author_encoded = author_enc.transform([user_author])[0]
    user_book_title_encoded = book_enc.transform([user_book_title])[0]
    user_genre_encoded = genre_enc.transform([user_genre])[0]


    predicted_rating = model.predict([pd.Series([user_age]), pd.Series([user_location_encoded]),
                                    pd.Series([user_author_encoded]), pd.Series([user_book_title_encoded]), pd.Series([user_genre_encoded])])
    #Ajout d'un compteur pour raccourcir le temps d'execution
    compteur = 0
    for idx, row in data_fin.iterrows():
        user_author = str(row['Book-Author'])
        user_book_title = str(row['Book-Title'])

        user_author_encoded = author_enc.transform([user_author])[0]
        user_book_title_encoded = book_enc.transform([user_book_title])[0]
        predicted_rating = model.predict([pd.Series([user_age]), pd.Series([user_location_encoded]),
                                        pd.Series([user_author_encoded]), pd.Series([user_book_title_encoded]), pd.Series([user_genre_encoded])])
        data_fin.loc[idx, "rating"] = predicted_rating[0]

        compteur = compteur + 1

        if compteur == 1000:
            break

    data_fin = data_fin.sort_values(by="rating",ascending=False)
    print(data_fin.head(10))
    return data_fin.head(10)


if __name__ == '__main__':
    app.run(debug=True)