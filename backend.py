from flask import Flask, jsonify

app = Flask('backend.py')

@app.route('/api/data', methods=['GET'])
def get_data():
    data = {'message': 'Hello from Python!'}
    return jsonify(data)

if 'backend.py' == 'main':
    app.run(debug=True)