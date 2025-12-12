# app.py
from flask import Flask, request, jsonify
import os
import csv
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------
# CLASE GYM RECOMMENDER
# ------------------------------
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
        self.user_rating = os.path.join(data_path,"usuarios_ejercicios_valoraciones.csv")
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
    # Actualización de valoraciones
    # ------------------------------
    def update_rating(self, id_usuario, genero, edad, peso, altura, id_ejercicio, valoracion):
        ratings_file = self.user_rating
        rows = []
        updated = False

        # Leer CSV existente
        if os.path.exists(ratings_file):
            with open(ratings_file, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 6 and row[0] == str(id_usuario) and row[5] == str(id_ejercicio):
                        # Actualizar registro existente
                        row[1] = genero
                        row[2] = edad
                        row[3] = peso
                        row[4] = altura
                        row[6] = valoracion
                        updated = True
                    rows.append(row)

        # Agregar nuevo registro si no se actualizó
        if not updated:
            rows.append([id_usuario, genero, edad, peso, altura, id_ejercicio, valoracion])

        # Reescribir CSV
        with open(ratings_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    # ------------------------------
    # Entrenamiento del modelo
    # ------------------------------
    def entrenar_modelo(self, force=False):
        mega_file = os.path.join(self.data_path, "megaGymDataset.csv")
        ratings_file = self.user_rating

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

        # DataFrame de ejercicios
        self.df = pd.DataFrame({
            "Exercise_Name": gym["Title"],
            "muscles": gym["BodyPart"].apply(self.clean_muscles),
            "Equipment": gym["Equipment"] if "Equipment" in gym.columns else "None",
            "Level": gym["Level"] if "Level" in gym.columns else gym["Difficulty"]
        })
        self.df = self.df[self.df["muscles"].map(len) > 0].reset_index(drop=True)
        self.df["id_ejercicio"] = self.df.index

        # Matriz de características de músculos
        self.mlb = MultiLabelBinarizer()
        self.feature_matrix = self.mlb.fit_transform(self.df["muscles"])

        # Matriz colaborativa de ratings por usuario
        ratings_pivot = self.ratings_df.pivot_table(
            index="id_usuario",
            columns="id_ejercicio",
            values="valoracion"
        ).fillna(0)
        self.corrMatrix = ratings_pivot.corr(method="pearson", min_periods=5)

        # Guardar modelo
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
    # Recomendación de ejercicios
    # ------------------------------
    def recomendar_ejercicios(self, user_data, nivel_usuario="Beginner", ejercicios_a_recomendar=15):
        if self.corrMatrix is None:
            raise Exception("Modelo no entrenado")

        df = self.df.copy()
        ratings = self.ratings_df.copy()

        # Rellenar valores faltantes
        ratings['edad'] = ratings['edad'].fillna(ratings['edad'].mean())
        ratings['peso'] = ratings['peso'].fillna(ratings['peso'].mean())
        ratings['altura'] = ratings['altura'].fillna(ratings['altura'].mean())
        ratings['genero'] = ratings['genero'].fillna('male')
        ratings['valoracion'] = ratings['valoracion'].fillna(1)
        ratings["genero"] = ratings["genero"].map({"male": 1, "female": 0})

        # Vector usuario actual
        user_vec = np.array([
            1 if user_data.get("genero","male")=="male" else 0,
            user_data.get("edad",25),
            user_data.get("peso",70),
            user_data.get("altura",170)
        ]).reshape(1,-1)

        # Similitud con usuarios existentes (atributos físicos)
        other_users = ratings[["genero","edad","peso","altura"]].values
        similarities = cosine_similarity(user_vec, other_users)[0]
        ratings["user_sim"] = similarities

        # Rating ponderado por id_ejercicio
        weighted = ratings.groupby("id_ejercicio").apply(
            lambda x: np.average(x["valoracion"], weights=x["user_sim"])
        ).fillna(0)
        df["rating_score"] = df["id_ejercicio"].map(weighted).fillna(0)

        # Similitud de contenido
        content_sim = cosine_similarity(self.feature_matrix, self.feature_matrix).mean(axis=1)
        df["content_sim"] = content_sim

        # Combinación final
        scaler = MinMaxScaler()
        df["rating_norm"] = scaler.fit_transform(df[["rating_score"]])
        df["final_score"] = 0.5 * df["rating_norm"] + 0.5 * df["content_sim"]

        # Filtrado por nivel
        niveles = ["Beginner","Intermediate","Expert"]
        nivel_usuario = nivel_usuario.capitalize()
        if nivel_usuario not in niveles:
            nivel_usuario = "Beginner"

        df["Level"] = df["Level"].astype(str).str.capitalize()
        idx = niveles.index(nivel_usuario)
        niveles_permitidos = niveles[:idx+1][::-1]

        capas = []
        for lvl in niveles_permitidos:
            sub = df[df["Level"]==lvl].sort_values("final_score", ascending=False)
            if not sub.empty:
                capas.append(sub)
        df_priorizado = pd.concat(capas, ignore_index=True)

        # Reparto por grupos musculares
        grupos = ["chest","back","legs","shoulders","biceps","triceps","glutes","abs"]
        seleccion = []
        usados = set()
        for g in grupos:
            cand = df_priorizado[df_priorizado["muscles"].apply(lambda x: g in x)]
            if not cand.empty:
                row = cand.iloc[0]
                if row["id_ejercicio"] not in usados:
                    seleccion.append(row)
                    usados.add(row["id_ejercicio"])
            if len(seleccion)>=ejercicios_a_recomendar:
                break

        if len(seleccion)<ejercicios_a_recomendar:
            faltan = ejercicios_a_recomendar-len(seleccion)
            extra = df_priorizado[~df_priorizado["id_ejercicio"].isin(usados)].head(faltan)
            seleccion.extend(extra.to_dict(orient="records"))
        else:
            seleccion = seleccion[:ejercicios_a_recomendar]

        seleccion_df = pd.DataFrame([r if isinstance(r,dict) else r.to_dict() for r in seleccion])

        return seleccion_df[["Exercise_Name","muscles","Equipment","Level","rating_score","final_score"]]


# ------------------------------
# FLASK APP
# ------------------------------
app = Flask(__name__)
recommender = GymRecommender()
recommender.entrenar_modelo(force=False)

# ------------------------------
# ENDPOINTS
# ------------------------------
@app.route('/')
def index():
    return "API de Recomendación de Ejercicios Gym OK"

@app.route('/recomendar', methods=['GET','POST'])
def recomendar():
    try:
        if recommender.ratings_df.empty:
            return jsonify({"status":"error","mensaje":"No hay datos de ratings"}), 500

        # Leer id_user y parámetros
        data = request.json if request.method == "POST" else request.args
        id_user = data.get("id_user")
        nivel = data.get("nivel", "Beginner")
        cantidad = int(data.get("cantidad", 10))

        if not id_user:
            return jsonify({"status":"error","mensaje":"Debe enviar id_user"}), 400
        id_user = int(id_user)

        # Datos del usuario
        if id_user not in recommender.ratings_df['id_usuario'].values:
            return jsonify({"status":"error","mensaje":"Usuario no tiene datos de ratings"}), 404
        user_ratings = recommender.ratings_df[recommender.ratings_df['id_usuario'] == id_user]
        user_data = {
            "genero": user_ratings['genero'].iloc[0],
            "edad": user_ratings['edad'].iloc[0],
            "peso": user_ratings['peso'].iloc[0],
            "altura": user_ratings['altura'].iloc[0]
        }

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


@app.route('/update', methods=['GET', 'POST'])
def update():
    # ------------------------
    # Obtener datos
    # ------------------------
    if request.method == "POST":
        data = request.json
        if not data:
            return jsonify({"status": "error", "mensaje": "Debe enviar datos JSON"}), 400
        id_usuario = data.get("id_usuario")
        genero = data.get("genero")
        edad = data.get("edad")
        peso = data.get("peso")
        altura = data.get("altura")
        id_ejercicio = data.get("id_ejercicio")
        valoracion = data.get("valoracion")
    else:  # GET
        id_usuario = request.args.get("id_usuario")
        genero = request.args.get("genero")
        edad = request.args.get("edad")
        peso = request.args.get("peso")
        altura = request.args.get("altura")
        id_ejercicio = request.args.get("id_ejercicio")
        valoracion = request.args.get("valoracion")

    # ------------------------
    # Validar campos
    # ------------------------
    required_fields = [id_usuario, genero, edad, peso, altura, id_ejercicio, valoracion]
    if not all(required_fields):
        return jsonify({"status":"error","mensaje":"Faltan datos requeridos"}),400

    try:
        # ------------------------
        # Actualizar CSV
        # ------------------------
        recommender.update_rating(id_usuario, genero, edad, peso, altura, id_ejercicio, valoracion)
        # Recargar ratings_df y reentrenar
        recommender.ratings_df = pd.read_csv(recommender.user_rating)
        recommender.entrenar_modelo(force=True)

        return jsonify({
            "status": "success",
            "mensaje": "Valoración actualizada y modelo re-entrenado correctamente",
            "datos_recibidos": {
                "id_usuario": id_usuario,
                "genero": genero,
                "edad": edad,
                "peso": peso,
                "altura": altura,
                "id_ejercicio": id_ejercicio,
                "valoracion": valoracion
            }
        }), 200

    except Exception as e:
        return jsonify({"status":"error","mensaje":f"Error actualizando CSV o modelo: {str(e)}"}),500


@app.route('/login', methods=['GET', 'POST'])
def login():
    dni = request.json.get("dni") if request.method=="POST" else request.args.get("dni")
    if not dni:
        return jsonify({"status":"error", "mensaje":"Debe enviar el DNI"}), 400
    try:
        usuarios = pd.read_csv(recommender.user_file)
        rating = pd.read_csv(recommender.user_rating)
    except Exception as e:
        return jsonify({"status":"error", "mensaje":f"No se pudieron leer CSV: {str(e)}"}),500

    usuario = usuarios[usuarios['dni']==dni]
    if usuario.empty:
        return jsonify({"status":"error", "mensaje":"Usuario no encontrado"}),404

    usuario_info = usuario[['id_user','nombre','apellido','dni']].iloc[0].to_dict()
    id_user = usuario_info['id_user']
    user_ratings = rating[rating['id_user']==id_user]
    usuario_info["ratings"] = user_ratings.to_dict(orient="records") if not user_ratings.empty else []

    return jsonify({"status":"success","usuario":usuario_info})


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


# ------------------------------
# RUN APP
# ------------------------------
if __name__=='__main__':
    app.run(debug=True, port=5000)
