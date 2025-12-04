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
    "muscles": gym["BodyPart"].apply(clean_muscles),
    "Level": gym["Level"] if "Level" in gym.columns else gym["Difficulty"]
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
def recomendar_ejercicios(user_data, nivel_usuario="Beginner", ejercicios_a_recomendar=15):

    global corrMatrix, df, ratings_df, mlb, feature_matrix

    if corrMatrix is None:
        raise Exception("Modelo no entrenado. Ejecuta entrenar_modelo() primero")

    df = df.copy()

    # ================================
    # 1. SIMILITUD CON OTROS USUARIOS
    # ================================
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

    # ================================
    # 2. SIMILITUD POR CONTENIDO
    # ================================
    content_sim = cosine_similarity(feature_matrix, feature_matrix).mean(axis=1)
    df["content_sim"] = content_sim

    # ================================
    # 3. NORMALIZAR Y SCORE FINAL
    # ================================
    scaler = MinMaxScaler()
    df["rating_norm"] = scaler.fit_transform(df[["rating_score"]])

    df["final_score"] = (
        0.5 * df["rating_norm"] +
        0.5 * df["content_sim"]
    )

    # ================================
    # 4. FILTRAR POR NIVEL (REAL)
    # ================================
    if "Level" not in df.columns:
        raise Exception("Tu CSV debe contener una columna 'Level' para filtrar ejercicios.")

    df["Level"] = df["Level"].astype(str).str.strip().str.capitalize()

    niveles_validos = ["Beginner", "Intermediate", "Expert"]

    nivel_usuario = nivel_usuario.capitalize()
    if nivel_usuario not in niveles_validos:
        nivel_usuario = "Beginner"

    df = df[df["Level"] == nivel_usuario]

    # ================================
    # 5. RETORNAR MEJORES EJERCICIOS
    # ================================
    return df.sort_values("final_score", ascending=False).head(ejercicios_a_recomendar)[[
        "Exercise_Name",
        "muscles",
        "Level",
        "rating_score",
        "final_score"
    ]]



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

    recomendaciones = recomendar_ejercicios(
        user_data,
        nivel_usuario="Expert",
        ejercicios_a_recomendar=10
    )

    print(recomendaciones.to_string(index=False))
