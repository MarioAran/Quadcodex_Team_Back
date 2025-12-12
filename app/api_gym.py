# app.py
from flask import Flask, request, jsonify
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
        self.user_file = os.path.join(data_path,"usuarios.csv")
        self.corrMatrix = None
        self.df = None
        self.ratings_df = None
        self.user_df = None
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
            with open(self.model_file, "rb") as f:
                data = pickle.load(f)
                self.corrMatrix = data["corrMatrix"]
                self.df = data["df"]
                self.ratings_df = data["ratings_df"]
                self.mlb = data["mlb"]
                self.feature_matrix = data["feature_matrix"]
            print("### Modelo cargado")
            return

        print("### Entrenando modelo desde cero...")

        gym = pd.read_csv(mega_file)
        self.ratings_df = pd.read_csv(ratings_file)
        self.ratings_df["valoracion"] = self.ratings_df["valoracion"].fillna(1)

        # Asignar id_usuario si no existe
        if "id_usuario" not in self.ratings_df.columns:
            num_usuarios = 500
            self.ratings_df["id_usuario"] = np.random.randint(1, num_usuarios + 1, size=len(self.ratings_df))

        # DataFrame ejercicios
        self.df = pd.DataFrame({
            "Exercise_Name": gym["Title"],
            "muscles": gym["BodyPart"].apply(self.clean_muscles),
            "Equipment": gym["Equipment"] if "Equipment" in gym.columns else "None",
            "Level": gym["Level"] if "Level" in gym.columns else gym["Difficulty"]
        })
        self.df = self.df[self.df["muscles"].map(len) > 0].reset_index(drop=True)
        self.df["id_ejercicio"] = self.df.index

        # Matriz características
        self.mlb = MultiLabelBinarizer()
        self.feature_matrix = self.mlb.fit_transform(self.df["muscles"])

        # Matriz colaborativa
        ratings_pivot = self.ratings_df.pivot_table(
            index="id_usuario",
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
    # NUEVO: Recomendación por ID
    # ------------------------------
    def recomendar_por_id(self, user_id, ejercicios_a_recomendar=15):
        """Recomendación personalizada basada en gustos reales del usuario."""
        user_ratings = self.ratings_df[self.ratings_df["id_usuario"] == user_id]

        if user_ratings.empty:
            print("Usuario sin datos → usando recomendación general.")
            return None

        # Ejercicios valorados por el usuario
        rated_ids = user_ratings[user_ratings["valoracion"] >= 3]["id_ejercicio"].tolist()
        if not rated_ids:
            print("Usuario sin valoraciones útiles → recomendación general.")
            return None

        # Similaridad entre ejercicios
        scores = pd.Series(dtype=float)
        for ej in rated_ids:
            if ej in self.corrMatrix.columns:
                corr = self.corrMatrix[ej].dropna()
                scores = scores.add(corr, fill_value=0)

        scores = scores.sort_values(ascending=False)

        # Evitar recomendar ejercicios ya vistos
        scores = scores[~scores.index.isin(rated_ids)]

        top_ids = scores.head(ejercicios_a_recomendar).index
        return self.df[self.df["id_ejercicio"].isin(top_ids)]


    # ------------------------------
    # Recomendación híbrida (AJUSTADA)
    # ------------------------------
    def recomendar_ejercicios(self, user_data, nivel_usuario="Beginner", ejercicios_a_recomendar=15):

        if self.corrMatrix is None:
            raise Exception("Modelo no entrenado. Ejecuta entrenar_modelo() primero")

        df = self.df.copy()
        ratings = self.ratings_df.copy()

        # ------------------------------
        # ¿VIENE CON USER ID?
        # ------------------------------
        user_id = user_data.get("user_id", None)

        tiene_id = user_id is not None and user_id in ratings["user_id"].unique()

        # ------------------------------
        # LIMPIEZA RÁPIDA
        # ------------------------------
        ratings['edad'] = ratings['edad'].fillna(ratings['edad'].mean())
        ratings['peso'] = ratings['peso'].fillna(ratings['peso'].mean())
        ratings['altura'] = ratings['altura'].fillna(ratings['altura'].mean())
        ratings['genero'] = ratings['genero'].fillna('male')
        ratings['valoracion'] = ratings['valoracion'].fillna(1)
        ratings["genero"] = ratings["genero"].map({"male": 1, "female": 0})

        # ============================================================
        #  CASO 1 — RECOMENDACIÓN POR USER ID (COLLABORATIVE)
        # ============================================================
        if tiene_id:

            print("### Modo colaborativo — usando user_id:", user_id)

            # Ratings del usuario
            user_ratings = ratings[ratings["user_id"] == user_id]

            # Ejercicios que ya valoró
            rated_items = user_ratings[user_ratings["valoracion"] > 0]["id_ejercicio"]

            # Sumar similitudes basado en correlación
            score_total = {}
            for item in rated_items:
                if item not in self.corrMatrix:
                    continue
                similares = self.corrMatrix[item].dropna()
                for sim_item, sim_value in similares.items():
                    score_total[sim_item] = score_total.get(sim_item, 0) + sim_value

            # Convertir a DataFrame
            score_df = pd.DataFrame(score_total.items(), columns=["id_ejercicio", "score"])
            score_df.sort_values("score", ascending=False, inplace=True)

            df = df.merge(score_df, on="id_ejercicio", how="left")
            df["score"] = df["score"].fillna(0)
            df["final_score"] = df["score"]

        # ============================================================
        #  CASO 2 — SIN USER ID → RECOMENDACIÓN POR FÍSICO + CONTENIDO
        # ============================================================
        else:

            print("### Modo general — sin user_id, usando físicas + contenido")

            # Vector físico del usuario
            user_vec = np.array([
                1 if user_data.get("genero", "male") == "male" else 0,
                user_data.get("edad", 25),
                user_data.get("peso", 70),
                user_data.get("altura", 170)
            ]).reshape(1, -1)

            other_users = ratings[["genero", "edad", "peso", "altura"]].values
            similarities = cosine_similarity(user_vec, other_users)[0]
            ratings["user_sim"] = similarities

            # Rating ponderado por similitud física
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

        # ============================================================
        #  FILTRADO POR NIVEL
        # ============================================================
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

        # ============================================================
        #  REPARTO POR GRUPOS MUSCULARES
        # ============================================================
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

        return seleccion_df[[
            "Exercise_Name",
            "muscles",
            "Equipment",
            "Level",
            "final_score"
        ]]

# ------------------------------
# FLASK APP
# ------------------------------
app = Flask(__name__)

recommender = GymRecommender()
recommender.entrenar_modelo(force=False)

@app.route('/')
def index():
    return "API de Recomendación de Ejercicios Gym OK"

@app.route('/recomendar', methods=['GET','POST'])
def recomendar():
    try:
        if recommender.ratings_df.empty:
            return jsonify({"status":"error","mensaje":"No hay datos de ratings"}), 500

        # Leer id_user enviado desde cliente (login previo)
        id_user = None
        if request.method == "POST":
            data = request.json
            if not data or "id_user" not in data:
                return jsonify({"status":"error","mensaje":"Debe enviar id_user"}), 400
            id_user = int(data.get("id_user"))
            nivel = data.get("nivel", "Beginner")
            cantidad = int(data.get("cantidad", 10))
        else:  # GET
            id_user = request.args.get("id_user")
            if not id_user:
                return jsonify({"status":"error","mensaje":"Debe enviar id_user"}), 400
            id_user = int(id_user)
            nivel = request.args.get("nivel", "Beginner")
            cantidad = int(request.args.get("cantidad", 10))

        # Buscar datos del usuario en ratings_df
        if id_user not in recommender.ratings_df['id_usuario'].values:
            return jsonify({"status":"error","mensaje":"Usuario no tiene datos de ratings"}), 404

        user_ratings = recommender.ratings_df[recommender.ratings_df['id_usuario'] == id_user]
        user_data = {
            "genero": user_ratings['genero'].iloc[0] if 'genero' in user_ratings.columns else 'male',
            "edad": user_ratings['edad'].iloc[0] if 'edad' in user_ratings.columns else 25,
            "peso": user_ratings['peso'].iloc[0] if 'peso' in user_ratings.columns else 70,
            "altura": user_ratings['altura'].iloc[0] if 'altura' in user_ratings.columns else 170
        }

        # Recomendaciones
        recomendaciones = recommender.recomendar_ejercicios(
            user_data=user_data,
            nivel_usuario=nivel,
            ejercicios_a_recomendar=cantidad
        )

        return jsonify({
            "status":"success",
            "id_user": id_user,
            "cantidad_recomendaciones": len(recomendaciones),
            "recomendaciones": recomendaciones.to_dict(orient='records'),
            "user_data": user_data,
            "nivel": nivel,
            "total_ejercicios": recommender.df.shape[0]
        })

    except Exception as e:
        return jsonify({"status":"error","mensaje":str(e)}),500

# ------------------------------
# NUEVO ENDPOINT: LOGIN POR DNI
# ------------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    # Leer dni
    dni = None
    if request.method == "POST":
        data = request.json
        if data:
            dni = data.get("dni")
    else:  # GET
        dni = request.args.get("dni")

    if not dni:
        return jsonify({"status":"error", "mensaje":"Debe enviar el DNI"}), 400

    # Leer CSV de usuarios
    try:
        usuarios = pd.read_csv(recommender.user_file)
    except Exception as e:
        return jsonify({"status":"error","mensaje":f"No se pudo leer usuarios.csv: {str(e)}"}), 500

    # Buscar usuario
    usuario = usuarios[usuarios['dni'] == dni]
    if usuario.empty:
        return jsonify({"status":"error","mensaje":"Usuario no encontrado"}), 404

    usuario_info = usuario[['id_user','nombre','apellido','dni']].iloc[0].to_dict()
    return jsonify({"status":"success","usuario": usuario_info})



@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status":"ok","mensaje":"API funcionando"}),200

@app.route('/info', methods=['GET'])
def info():
    return jsonify({
        "total_ejercicios": recommender.df.shape[0] if recommender.df is not None else 0,
        "total_usuarios": recommender.ratings_df["id_usuario"].nunique() if recommender.ratings_df is not None else 0,
        "niveles_disponibles": recommender.df["Level"].unique().tolist() if recommender.df is not None else []
    })

if __name__=='__main__':
    app.run(debug=True, port=5000)
