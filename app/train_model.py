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

    def __init__(self, data_path='Data/', model_file='modelo_gym.pkl'):
        self.data_path = data_path
        self.model_file = os.path.join(data_path, model_file)
        self.corrMatrix = None
        self.df = None
        self.ratings_df = None
        self.mlb = None
        self.feature_matrix = None

    # ------------------------------
    # Limpieza de grupos musculares
    # ------------------------------
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

    # ------------------------------
    # Entrenamiento
    # ------------------------------
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

        # ------------------------------
        # Incluir equipamiento
        # ------------------------------
        self.df = pd.DataFrame({
            "Exercise_Name": gym["Title"],
            "muscles": gym["BodyPart"].apply(self.clean_muscles),
            "Equipment": gym["Equipment"] if "Equipment" in gym.columns else "None",
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

    # ------------------------------
    # Recomendación mejorada
    # ------------------------------
    def recomendar_ejercicios(self, user_data, nivel_usuario="Beginner", ejercicios_a_recomendar=15):

        if self.corrMatrix is None:
            raise Exception("Modelo no entrenado. Ejecuta entrenar_modelo() primero")

        df = self.df.copy()
        ratings = self.ratings_df.copy()

        # Rellenar valores faltantes
        ratings['edad'] = ratings['edad'].fillna(ratings['edad'].mean())
        ratings['peso'] = ratings['peso'].fillna(ratings['peso'].mean())
        ratings['altura'] = ratings['altura'].fillna(ratings['altura'].mean())
        ratings['genero'] = ratings['genero'].fillna('male')
        ratings['valoracion'] = ratings['valoracion'].fillna(1)

        ratings["genero"] = ratings["genero"].map({"male": 1, "female": 0})

        # Vector del usuario actual
        user_vec = np.array([
            1 if user_data.get("genero", "male") == "male" else 0,
            user_data.get("edad", 25),
            user_data.get("peso", 70),
            user_data.get("altura", 170)
        ]).reshape(1, -1)

        other_users = ratings[["genero", "edad", "peso", "altura"]].values
        similarities = cosine_similarity(user_vec, other_users)[0]
        ratings["user_sim"] = similarities

        # Rating ponderado
        weighted = ratings.groupby("id_ejercicio").apply(
            lambda x: np.average(x["valoracion"], weights=x["user_sim"])
        ).fillna(0)

        df["rating_score"] = df["id_ejercicio"].map(weighted).fillna(0)

        # Similitud de contenido
        content_sim = cosine_similarity(self.feature_matrix, self.feature_matrix).mean(axis=1)
        df["content_sim"] = content_sim

        scaler = MinMaxScaler()
        df["rating_norm"] = scaler.fit_transform(df[["rating_score"]])
        df["final_score"] = 0.5 * df["rating_norm"] + 0.5 * df["content_sim"]

        # ------------------------------
        # SISTEMA DE NIVELES FLEXIBLE
        # ------------------------------
        niveles = ["Beginner", "Intermediate", "Expert"]
        nivel_usuario = nivel_usuario.capitalize()

        if nivel_usuario not in niveles:
            nivel_usuario = "Beginner"

        df["Level"] = df["Level"].astype(str).str.capitalize()

        idx = niveles.index(nivel_usuario)
        niveles_permitidos = niveles[:idx + 1][::-1]

        capas = []
        for lvl in niveles_permitidos:
            sub = df[df["Level"] == lvl].sort_values("final_score", ascending=False)
            if not sub.empty:
                capas.append(sub)

        df_priorizado = pd.concat(capas, ignore_index=True)

        # ------------------------------
        # REPARTO POR GRUPOS MUSCULARES
        # ------------------------------
        grupos = ["chest", "back", "legs", "shoulders", "biceps", "triceps", "glutes", "abs"]

        seleccion = []
        usados = set()

        for g in grupos:
            cand = df_priorizado[df_priorizado["muscles"].apply(lambda x: g in x)]
            if not cand.empty:
                row = cand.iloc[0]
                if row["id_ejercicio"] not in usados:
                    seleccion.append(row)
                    usados.add(row["id_ejercicio"])
            if len(seleccion) >= ejercicios_a_recomendar:
                break

        if len(seleccion) < ejercicios_a_recomendar:
            faltan = ejercicios_a_recomendar - len(seleccion)
            extra = df_priorizado[~df_priorizado["id_ejercicio"].isin(usados)].head(faltan)
            seleccion.extend(extra.to_dict(orient="records"))
        else:
            seleccion = seleccion[:ejercicios_a_recomendar]

        seleccion_df = pd.DataFrame([r if isinstance(r, dict) else r.to_dict() for r in seleccion])

        # ------------------------------
        # DEVOLVER EQUIPAMIENTO
        # ------------------------------
        return seleccion_df[[
            "Exercise_Name",
            "muscles",
            "Equipment",
            "Level",
            "rating_score",
            "final_score"
        ]]
    
if __name__ == "__main__":
    # Inicializar y entrenar modelo
    recommender = GymRecommender()
    recommender.entrenar_modelo(force=False)

    # Datos de usuario de prueba
    user_data = {
        "genero": "male",
        "edad": 28,
        "peso": 100,
        "altura": 178
        }

    # Solicitar recomendaciones
    recomendaciones = recommender.recomendar_ejercicios(
        user_data=user_data,
        nivel_usuario="Expert",
        ejercicios_a_recomendar=10
        )

    # Mostrar resultados en consola
    print("=== Recomendaciones de Ejercicios ===")
    print(recomendaciones.to_string(index=False))
