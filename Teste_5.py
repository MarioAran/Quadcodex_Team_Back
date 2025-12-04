import os
import pickle
import pandas as pd
import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# ================================
# CONFIG
# ================================
MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelo_gym.pkl")

corrMatrix = None
df = None
ratings_df = None
mlb = None
feature_matrix = None

# ================================
# MAPA DE MUSCULOS
# ================================
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

# ================================
# LIMPIEZA MUSCULOS
# ================================
def normalize_muscle(x):
    x = str(x).lower()
    for k in MUSCLE_MAP:
        if k in x:
            return MUSCLE_MAP[k]
    return None

def clean_muscles(x):
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return [normalize_muscle(t) for t in x if normalize_muscle(t)]
    if isinstance(x, str):
        tokens = [t.strip() for t in x.replace(";", ",").split(",")]
        return list(set([normalize_muscle(t) for t in tokens if normalize_muscle(t)]))
    return []


# ================================
# ENTRENAR MODELO
# ================================
def entrenar_modelo(force=False):

    global corrMatrix, df, ratings_df, mlb, feature_matrix

    base_path = os.getcwd()

    mega_file = os.path.join(base_path, "Data", "megaGymDataset.csv")
    ratings_file = os.path.join(base_path, "Data", "usuarios_ejercicios_valoraciones.csv")

    if not force and os.path.exists(MODEL_FILE):
        print("### Cargando modelo desde archivo...")
        with open(MODEL_FILE, "rb") as f:
            data = pickle.load(f)
            corrMatrix = data["corrMatrix"]
            df = data["df"]
            ratings_df = data["ratings_df"]
            mlb = data["mlb"]
            feature_matrix = data["feature_matrix"]
        print("Ejercicios totales:", df.shape[0])
        print("### Modelo cargado correctamente")
        return

    print("### Entrenando modelo desde cero...")

    # SOLO USAMOS megaGymDataset + usuarios
    gym = pd.read_csv(mega_file)
    ratings_df = pd.read_csv(ratings_file)

    # ================================
    # LIMPIEZA DATOS
    # ================================
    ratings_df["valoracion"] = ratings_df["valoracion"].fillna(1)

    df = pd.DataFrame({
        "Exercise_Name": gym["Title"],
        "muscles": gym["BodyPart"].apply(clean_muscles)
    })

    df = df[df["muscles"].map(len) > 0]
    df.reset_index(drop=True, inplace=True)
    df["id_ejercicio"] = df.index



    # ========= MATRIZ DE CONTENIDO =========
    mlb = MultiLabelBinarizer()
    feature_matrix = mlb.fit_transform(df["muscles"])

    # ========= MATRIZ USUARIOS → EJERCICIOS =========
    ratings_df["user_id"] = range(len(ratings_df))

    ratings_pivot = ratings_df.pivot_table(
        index="user_id",
        columns="id_ejercicio",
        values="valoracion"
    ).fillna(0)

    corrMatrix = ratings_pivot.corr(method="pearson", min_periods=5)

    # ===== GUARDAR MODELO =====
    with open(MODEL_FILE, "wb") as f:
        pickle.dump({
            "corrMatrix": corrMatrix,
            "df": df,
            "ratings_df": ratings_df,
            "mlb": mlb,
            "feature_matrix": feature_matrix
        }, f)
        
    print("### Modelo entrenado y guardado ✅")


# ================================
# RECOMENDAR (POR GRUPO)
# ================================
def recomendar_por_grupo(
    user_data,
    user_priorities,
    exercises_per_group=5
):

    global corrMatrix, df, ratings_df, mlb, feature_matrix

    if corrMatrix is None:
        raise Exception("Modelo no entrenado. Ejecuta entrenar_modelo() primero")

    df = df.copy()

    # ========== SIMILITUD CON USUARIOS ==========
    ratings_df["genero"] = ratings_df["genero"].map({"male": 1, "female": 0})

    user_vec = np.array([
        1 if user_data["genero"] == "male" else 0,
        user_data["edad"],
        user_data["peso"],
        user_data["altura"]
    ]).reshape(1, -1)

    other_users = ratings_df[["genero", "edad", "peso", "altura"]].values
    similarities = cosine_similarity(user_vec, other_users)[0]

    ratings_df["user_sim"] = similarities

    weighted = ratings_df.groupby("id_ejercicio").apply(
        lambda x: np.average(x["valoracion"], weights=x["user_sim"])
    ).fillna(0)

    df["rating_score"] = df["id_ejercicio"].map(weighted).fillna(0)

    # ========== SIMILITUD POR MUSCULOS ==========
    user_vector = np.zeros(len(mlb.classes_))

    for muscle, prio in user_priorities.items():
        if muscle in mlb.classes_:
            idx = list(mlb.classes_).index(muscle)
            user_vector[idx] = prio

    content_sim = cosine_similarity([user_vector], feature_matrix)[0]
    df["content_sim"] = content_sim

    # ========= NORMALIZAR =========
    scaler = MinMaxScaler()
    df["rating_norm"] = scaler.fit_transform(df[["rating_score"]])

    # ========= SCORE FINAL =========
    df["final_score"] = 0.6 * df["content_sim"] + 0.4 * df["rating_norm"]

    # ========= AGRUPAR Y RETORNAR ==========
    resultados = []

    for muscle in mlb.classes_:
        df_muscle = df[df["muscles"].apply(lambda x: muscle in x)]
        if df_muscle.empty:
            continue

        top_muscle = df_muscle.sort_values("final_score", ascending=False).head(exercises_per_group).copy()
        top_muscle["grupo_muscular"] = muscle
        resultados.append(top_muscle)

    if not resultados:
        return pd.DataFrame()

    final_df = pd.concat(resultados)

    return final_df[[
        "grupo_muscular",
        "Exercise_Name",
        "muscles",
        "rating_score",
        "final_score"
    ]].sort_values(["grupo_muscular", "final_score"], ascending=[True, False])


# ================================
# USO
# ================================
if __name__ == "__main__":

    entrenar_modelo(force=False)

    user_data = {
        "genero": "male",
        "edad": 28,
        "peso": 100,
        "altura": 178
    }

    user_priorities = {
        "chest": 7,
        "back": 5,
        "legs": 6,
        "abs": 6,
        "biceps": 7
    }

    recomendaciones = recomendar_por_grupo(user_data, user_priorities, exercises_per_group=3)

    print(recomendaciones.to_string(index=False))
