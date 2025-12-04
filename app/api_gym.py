# app.py
from flask import Flask, request, jsonify
from train_model import GymRecommender

app = Flask(__name__)
##
# ejemplo url api http://localhost:5000/recomendar?genero=male&edad=30&peso=120&altura=160&nivel=Intermediate&cantidad=5
##

recommender = GymRecommender()
recommender.entrenar_modelo(force=False)

@app.route('/')
def index():
    return "API de Recomendación de Ejercicios Gym OK"

# ================================
# ENDPOINT DE RECOMENDACIÓN
# ================================
@app.route('/recomendar', methods=['GET', 'POST'])
def recomendar():
    try:
        if request.method == 'GET':
            genero = request.args.get('genero', default='male')
            edad = int(request.args.get('edad', default=25))
            peso = float(request.args.get('peso', default=70))
            altura = float(request.args.get('altura', default=170))
            nivel = request.args.get('nivel', default='Beginner')
            cantidad = int(request.args.get('cantidad', default=10))
        elif request.method == 'POST':
            data = request.json
            if not data:
                return jsonify({"status":"error","mensaje":"No se enviaron datos"}), 400
            genero = data.get('genero', 'male')
            edad = int(data.get('edad', 25))
            peso = float(data.get('peso', 70))
            altura = float(data.get('altura', 170))
            nivel = data.get('nivel', 'Beginner')
            cantidad = int(data.get('cantidad', 10))
        user_data = {"genero": genero, "edad": edad, "peso": peso, "altura": altura}
        recomendaciones = recommender.recomendar_ejercicios(
            user_data=user_data,
            nivel_usuario=nivel,
            ejercicios_a_recomendar=cantidad
        )
        recomendaciones_json = recomendaciones.to_dict(orient='records')
        return jsonify({
            "status": "success",
            "cantidad_recomendaciones": len(recomendaciones_json),
            "recomendaciones": recomendaciones_json,
            "user_data": user_data,
            "nivel": nivel,
            "total_ejercicios": recommender.df.shape[0]
        })

    except Exception as e:
        return jsonify({"status": "error", "mensaje": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "mensaje": "API funcionando"}), 200

@app.route('/info', methods=['GET'])
def info():
    return jsonify({
        "total_ejercicios": recommender.df.shape[0] if recommender.df is not None else 0,
        "total_usuarios": recommender.ratings_df.shape[0] if recommender.ratings_df is not None else 0,
        "niveles_disponibles": recommender.df["Level"].unique().tolist() if recommender.df is not None else []
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)