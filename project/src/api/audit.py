from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/api/app', methods=['GET', 'POST'])
def audit():
    if request.method == 'GET':
        # Route de test pour vérifier que le serveur fonctionne
        return jsonify({
            "message": "Server is running",
            "status": "success"
        })
    
    elif request.method == 'POST':
        try:
            data = request.get_json()
            return jsonify({
                "message": "Analysis completed",
                "data_received": data,
                "status": "success"
            })
        except Exception as e:
            print(f"Error processing request: {str(e)}")  # Log l'erreur
            return jsonify({
                "message": "Error processing request",
                "error": str(e),
                "status": "error"
            }), 500

# Route de test additionnelle
@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Test endpoint working"}), 200

if __name__ == '__main__':
    app.run(debug=True, port=8000)
