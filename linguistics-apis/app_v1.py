import base64
import os
from flask import Flask, json, jsonify, request, send_file, abort
from datetime import datetime
from urllib.parse import unquote
import logging
from database.SelectTables import SelectTables
from flask_cors import CORS
from search.NewsCypherGraph import NewsCypherGraph

app = Flask(__name__)
CORS(app, origins=["http://10.42.0.1:4200", "http://127.0.0.1:4200", "http://localhost:4200", "http://10.42.0.1:3000", "http://localhost:4200"])


@app.route("/news_search_retrieve_actions_by_query")
def news_search_retrieve_actions_by_query():
    try:
        query: str = unquote(request.args.get('query'))
        select = NewsCypherGraph().generate_graphs(query)
        return jsonify(select), 200
    except Exception as e:
        endpoint: str = "/news_search_retrieve_actions_by_query"
        logging.error(f"Error: {e} | {endpoint} | at {datetime.now()}")
        return jsonify({"status": "error", "message": f"Error: {e} | {endpoint} | at {datetime.now()}"})
    
@app.route("/news_search_by_query")
def news_search_by_query():
    try:
        query: str = unquote(request.args.get('query'))
        select = SelectTables().filter_news_api_table_by_search(query)
        return jsonify(select), 200
    except Exception as e:
        endpoint: str = "/news_search_by_query"
        logging.error(f"Error: {e} | {endpoint} | at {datetime.now()}")
        return jsonify({"status": "error", "message": f"Error: {e} | {endpoint} | at {datetime.now()}"})
    
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)