import pandas as pd
from sklearn.model_selection import train_test_split
from blitz_tests.blitz_test_for_classification import Blitz
from sklearn.preprocessing import StandardScaler
from datavisualization.matplot_visualisation import matplot_visualisation

"""
художник: Ім'я художника.
пісня: Назва треку.
duration_ms: тривалість треку в мілісекундах.
явний: текст або вміст пісні чи музичного відео містить один або кілька критеріїв, які можна вважати образливими або непридатними для дітей.
рік: Рік випуску треку.
популярність: чим вище значення, тим популярніша пісня.
танцювальність: танцювальність описує, наскільки композиція підходить для танцю на основі комбінації музичних елементів, включаючи темп, стабільність ритму, силу удару та загальну регулярність. Значення 0,0 є найменш танцювальним, а 1,0 — найбільш танцювальним.
енергія: енергія є мірою від 0,0 до 1,0 і являє собою перцептивну міру інтенсивності та активності.
key: ключ, у якому знаходиться трек. Цілі числа відображаються на висоту звуку за допомогою стандартної нотації Pitch Class. напр. 0 = C, 1 = C♯/D♭, 2 = D і так далі. Якщо ключ не виявлено, значення дорівнює -1.
loudness: загальна гучність треку в децибелах (дБ). Значення гучності усереднені по всій доріжці і корисні для порівняння відносної гучності треків. Гучність – це якість звуку, яка є основним психологічним корелятом фізичної сили (амплітуди). Значення зазвичай коливаються від -60 до 0 дБ.
режим: Mode вказує модальність (мажорну або мінорну) треку, тип ладу, з якого походить його мелодійний зміст. Мажор представлено 1, а мінор 0.
мовленнєвість: Speechiness виявляє наявність вимовлених слів у д оріжці. Чим більше винятково схожий на мовлення запис (наприклад, ток-шоу, аудіокнига, вірші), тим ближче до 1,0 значення атрибута. Значення вище 0,66 описують треки, які, ймовірно, повністю складаються з вимовлених слів. Значення від 0,33 до 0,66 описують треки, які можуть містити як музику, так і мовлення, як у розділах, так і пошарово, включаючи такі випадки, як реп-музика. Значення нижче 0,33, швидше за все, представляють музику та інші треки, які не схожі на мовлення.
акустичність: міра довіри від 0,0 до 1,0 того, чи є трек акустичним. 1.0 означає високу впевненість, що трек є акустичним.
інструментальність: передбачає, чи не містить трек вокал. Звуки «Ой» та «Ааа» в цьому контексті розглядаються як інструментальні. Реп або розмовні треки явно «голосні». Чим ближче значення інструментальності до 1,0, тим більша ймовірність, що трек не містить вокального вмісту. Значення вище 0,5 призначені для представлення інструментальних треків, але впевненість вища, коли значення наближається до 1,0.
живість: Визначає присутність аудиторії у записі. Більш високі значення живучості представляють підвищену ймовірність того, що трек було виконано наживо. Значення вище 0,8 забезпечує високу ймовірність того, що трек працює.
валентність: показник від 0,0 до 1,0, що описує музичний позитив, який передає трек. Композиції з високою валентністю звучать більш позитивно (наприклад, щасливі, веселі, ейфорійні), тоді як треки з низькою валентністю звучать більш негативно (наприклад, сумний, пригнічений, злий).
tempo: загальний оцінений темп треку в ударах за хвилину (BPM). У музичній термінології темп — це швидкість або темп певної п’єси, що випливає безпосередньо із середньої тривалості удару.
жанр: Жанр треку.
"""


class Spotify:
    def __init__(self):
        self.df = pd.read_csv("datasets/songs_normalize.csv")

    def drop_data(self):
        self.df = self.df.drop(["key"], axis=1)

    def visualisation_popularity_and_loundess(self):
        X = self.df["popularity"]
        Y = self.df["loudness"]

        matplot_visualisation(X, Y).visualisation()

    def visualisation_popularity_and_duration_ms(self):
        X = self.df["popularity"]
        Y = self.df["duration_ms"]

        matplot_visualisation(X, Y).visualisation()

    def visualisation_danceability_by_2000(self):
        X = self.df["year"]
        Y = self.df["danceability"]

        matplot_visualisation(X, Y).visualisation()

    def visualisation_populality_by_years(self):
        X = self.df["year"]
        Y = self.df["popularity"]

        matplot_visualisation(X, Y).visualisation()

    def visualisation_speechiness_by_popularity(self):
        X = self.df["speechiness"]
        Y = self.df["popularity"]

        matplot_visualisation(X, Y).visualisation()

    def visualisation_speechiness_by_year(self):
        X = self.df["speechiness"]
        Y = self.df["year"]

        matplot_visualisation(X, Y).visualisation()

    def create_new_df(self):
        artist = self.df["artist"]
        year = self.df["year"]
        song = self.df["song"]
        danceability = self.df["danceability"]

        d = {
            "artist": pd.Series(artist),
            "year": pd.Series(year),
            "song": pd.Series(song),
            "danceability": pd.Series(danceability)
        }

        df = pd.DataFrame(d)

        df.to_csv("output.csv")

    def learning(self):
        X = self.df.drop(['mode', 'duration_ms', 'year', 'artist', 'song', 'genre'], axis=1)
        Y = self.df["mode"]
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=1)

        rb = StandardScaler()
        x_train = rb.fit_transform(x_train)
        x_test = rb.fit_transform(x_test)

        bt = Blitz(x_train, x_test, y_train, y_test, 100, 10)
        bt.testing_models('accuracy')


if __name__ == '__main__':
    s = Spotify()
    s.drop_data()
    s.learning()