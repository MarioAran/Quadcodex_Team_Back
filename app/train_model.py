# gym_recommender.py
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

class GymRecommender:
    MUSCLE_MAP = {
        "abs": "abs", "abdominals": "abs", "core": "abs",
        "quadriceps": "legs", "quads": "legs", "hamstrings": "legs",
        "calves": "legs", "legs": "legs",
        "chest": "chest", "pecs": "chest", "pectorals": "chest",
        "back": "back", "lats": "back", "latissimus": "back",
        "shoulders": "shoulders", "deltoids": "shoulders",
        "biceps": "biceps", "triceps": "triceps",
        "glutes": "glutes"
    }

    def __init__(self, data_path='../Data/', model_file='modelo_gym.pkl'):
        self.data_path = data_path
        self.model_file = os.path.join(data_path, model_file)
        self.corrMatrix = None
        self.df = None
        self.ratings_df = None
        self.mlb = None
        self.feature_matrix = None
    def normalize_muscle(self, x):
        x = str(x).lower()
        for k in self.MUSCLE_MAP:
            if k in x:
                return self.MUSCLE_MAP[k]
        return None

    def clean_muscles(self, x):
        if pd.isna(x):
            return []
        if isinstance(x, list):
            return [self.normalize_muscle(t) for t in x if self.normalize_muscle(t)]
        if isinstance(x, str):
            tokens = [t.strip() for t in x.replace(";", ",").split(",")]
            return list(set([self.normalize_muscle(t) for t in tokens if self.normalize_muscle(t)]))
        return []

    def entrenar_modelo(self, force=False):
        mega_file = os.path.join(self.data_path, "megaGymDataset.csv")
        ratings_file = os.path.join(self.data_path, "usuarios_ejercicios_valoraciones.csv")

        if not force and os.path.exists(self.model_file):
            print("### Cargando modelo desde archivo...")
            with open(self.model_file, "rb") as f:
                data = pickle.load(f)
                self.corrMatrix = data["corrMatrix"]
                self.df = data["df"]
                self.ratings_df = data["ratings_df"]
                self.mlb = data["mlb"]
                self.feature_matrix = data["feature_matrix"]
            print("Ejercicios totales:", self.df.shape[0])
            print("### Modelo cargado correctamente")
            return

        print("### Entrenando modelo desde cero...")
        gym = pd.read_csv(mega_file)
        self.ratings_df = pd.read_csv(ratings_file)
        self.ratings_df["valoracion"] = self.ratings_df["valoracion"].fillna(1)

        self.df = pd.DataFrame({
            "Exercise_Name": gym["Title"],
            "muscles": gym["BodyPart"].apply(self.clean_muscles),
            "Level": gym["Level"] if "Level" in gym.columns else gym["Difficulty"]
        })
        self.df = self.df[self.df["muscles"].map(len) > 0]
        self.df.reset_index(drop=True, inplace=True)
        self.df["id_ejercicio"] = self.df.index

        self.mlb = MultiLabelBinarizer()
        self.feature_matrix = self.mlb.fit_transform(self.df["muscles"])

        self.ratings_df["user_id"] = range(len(self.ratings_df))
        ratings_pivot = self.ratings_df.pivot_table(
            index="user_id",
            columns="id_ejercicio",
            values="valoracion"
        ).fillna(0)

        self.corrMatrix = ratings_pivot.corr(method="pearson", min_periods=5)

        with open(self.model_file, "wb") as f:
            pickle.dump({
                "corrMatrix": self.corrMatrix,
                "df": self.df,
                "ratings_df": self.ratings_df,
                "mlb": self.mlb,
                "feature_matrix": self.feature_matrix
            }, f)
        print("### Modelo entrenado y guardado ✅")

    def recomendar_ejercicios(self, user_data, nivel_usuario="Beginner", ejercicios_a_recomendar=15):
        if self.corrMatrix is None:
            raise Exception("Modelo no entrenado. Ejecuta entrenar_modelo() primero")

        df_local = self.df.copy()

        ratings_df_clean = self.ratings_df.copy()

        ratings_df_clean['edad'] = ratings_df_clean['edad'].fillna(ratings_df_clean['edad'].mean())
        ratings_df_clean['peso'] = ratings_df_clean['peso'].fillna(ratings_df_clean['peso'].mean())
        ratings_df_clean['altura'] = ratings_df_clean['altura'].fillna(ratings_df_clean['altura'].mean())
        ratings_df_clean['genero'] = ratings_df_clean['genero'].fillna('male')
        ratings_df_clean['valoracion'] = ratings_df_clean['valoracion'].fillna(1)

        ratings_df_clean["genero"] = ratings_df_clean["genero"].map({"male": 1, "female": 0})

        user_vec = np.array([
            1 if user_data.get("genero", "male") == "male" else 0,
            user_data.get("edad", 25),
            user_data.get("peso", 70),
            user_data.get("altura", 170)
        ]).reshape(1, -1)

        other_users = ratings_df_clean[["genero", "edad", "peso", "altura"]].values
        similarities = cosine_similarity(user_vec, other_users)[0]
        ratings_df_clean["user_sim"] = similarities

        weighted = ratings_df_clean.groupby("id_ejercicio").apply(
            lambda x: np.average(x["valoracion"], weights=x["user_sim"])
        ).fillna(0)
        df_local["rating_score"] = df_local["id_ejercicio"].map(weighted).fillna(0)

        content_sim = cosine_similarity(self.feature_matrix, self.feature_matrix).mean(axis=1)
        df_local["content_sim"] = content_sim

        df_local["rating_score"] = df_local["rating_score"].fillna(0)
        scaler = MinMaxScaler()
        df_local["rating_norm"] = scaler.fit_transform(df_local[["rating_score"]])
        df_local["final_score"] = 0.5 * df_local["rating_norm"] + 0.5 * df_local["content_sim"]

        df_local["Level"] = df_local["Level"].astype(str).str.strip().str.capitalize()
        niveles_validos = ["Beginner", "Intermediate", "Expert"]
        nivel_usuario = nivel_usuario.capitalize()
        if nivel_usuario not in niveles_validos:
            nivel_usuario = "Beginner"

        df_local = df_local[df_local["Level"] == nivel_usuario]

        return df_local.sort_values("final_score", ascending=False).head(ejercicios_a_recomendar)[[
            "Exercise_Name",
            "muscles",
            "Level",
            "rating_score",
            "final_score"
        ]]

if __name__ == "__main__":
    recommender = GymRecommender()
    recommender.entrenar_modelo(force=False)

    user_data = {"genero": "male", "edad": 28, "peso": 100, "altura": 178}
    recomendaciones = recommender.recomendar_ejercicios(user_data, nivel_usuario="Expert", ejercicios_a_recomendar=10)
    print(recomendaciones.to_string(index=False))
