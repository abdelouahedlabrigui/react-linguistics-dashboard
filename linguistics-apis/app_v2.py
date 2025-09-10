from elasticsearch8 import Elasticsearch
from urllib.parse import unquote
from collections import defaultdict
from datetime import datetime
import html
import os
import json
import re
from bs4 import BeautifulSoup
import oracledb
from flask import Flask, json, jsonify, request
import pandas as pd
import requests
import logging
from flask_cors import CORS
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=["http://10.42.0.1:4200", "http://127.0.0.1:4200", "http://localhost:4200", "http://10.42.0.1:3000", "http://127.0.0.1:3000", "http://localhost:3000"])


# linguistics Elasticsearch
from repos.linguistics.LinguisticsTextInterpreter import LinguisticsTextInterpreter
@app.route("/news/linguistics/v1", methods=["POST"])
def news_linguistics_v1():
    try:
        article: dict = request.get_json()
        keys = ["source", "author", "title", "description", "url", "urlToImage", "publishedAt", "content"]
        missing_keys = [key for key in keys if key not in article]
        lang = LinguisticsTextInterpreter()
        lang.create_index()
        if missing_keys:
            raise ValueError(f"Missing keys in article: {', '.join(missing_keys)}")
        es = Elasticsearch("http://10.42.0.243:9200") # Replace with your Elasticsearch host
        index_name = "linguistic_analysis" # Replace with your index name
        response_before = es.count(index=index_name)
        
        lang.interpret_text(news_article=article)
        response_after = es.count(index=index_name)
        document_counts = {
            "index_name": index_name,
            "count_before": response_before["count"],
            "count_after": response_after["count"]
        }
        return jsonify({"status": "success", "message": f"Total documents in '{index_name}': {document_counts}"}), 200
    except Exception as e:
        endpoint: str = "/news/linguistics/v1"
        logging.error(f"Error: {e} | {endpoint} | at {datetime.now()}")
        return jsonify({"status": "error", "message": f"Error: {e} | {endpoint} | at {datetime.now()}"})

@app.route("/news/linguistics/v1/search", methods=["GET"])
def news_linguistics_v1_search():
    try:
        search: str = unquote(request.args.get("query"))
        if len(search) > 2:
            es = Elasticsearch("http://10.42.0.243:9200") # Replace with your Elasticsearch host
            index_name = "linguistic_analysis" # Replace with your index name
            query = {
                "query": {
                    "query_string": {
                        "query": f"*{search}*",
                        "fields": ["news_article.description"]
                    }
                }
            }
            response = es.search(index=index_name, body=query)
            return jsonify(response["hits"]["hits"]), 200
        else:
            return jsonify({"status": "error", "message": f"Param search string empty | at {datetime.now()}"})
    except Exception as e:
        endpoint: str = "/news/linguistics/v1/search"
        logging.error(f"Error: {e} | {endpoint} | at {datetime.now()}")
        return jsonify({"status": "error", "message": f"Error: {e} | {endpoint} | at {datetime.now()}"})

# linguistics Elasticsearch
from repos.linguistics.LinguisticsTextInterpreter import LinguisticsTextInterpreter
@app.route("/news/psycho_linguistics/v1/search", methods=["GET"])
def news_psycho_linguistic_analysis_v1_search():
    try:
        search: str = unquote(request.args.get("query"))
        if len(search) > 2:
            es = Elasticsearch("http://10.42.0.243:9200") # Replace with your Elasticsearch host
            index_name = "psycho_linguistic_analysis" # Replace with your index name
            query = {
                "query": {
                    "query_string": {
                        "query": f"*{search}*",
                        "fields": ["news_article.description"]
                    }
                }
            }
            response = es.search(index=index_name, body=query)
            return jsonify(response["hits"]["hits"]), 200
        else:
            return jsonify({"status": "error", "message": f"Param search string empty | at {datetime.now()}"})
    except Exception as e:
        endpoint: str = "/news/psycho_linguistics/v1/search"
        logging.error(f"Error: {e} | {endpoint} | at {datetime.now()}")
        return jsonify({"status": "error", "message": f"Error: {e} | {endpoint} | at {datetime.now()}"})

# psycholinguistics Elasticsearch
from repos.linguistics.PsychoLinguisticsTextInterpreter import PsychoLinguisticsTextInterpreter, PsychoLinguisticsTextInterpreterCreateIndex
@app.route("/news/psycho_linguistics/v1", methods=["POST"])
def news_psycho_linguistics_v1():
    try:
        article: dict = request.get_json()
        keys = ["source", "author", "title", "description", "url", "urlToImage", "publishedAt", "content"]
        missing_keys = [key for key in keys if key not in article]
        create_index = PsychoLinguisticsTextInterpreterCreateIndex()
        create_index.create_index()
        if missing_keys:
            raise ValueError(f"Missing keys in article: {', '.join(missing_keys)}")
        es = Elasticsearch("http://10.42.0.243:9200") # Replace with your Elasticsearch host
        index_name = "psycho_linguistic_analysis" # Replace with your index name
        response_before = es.count(index=index_name)
        lang = PsychoLinguisticsTextInterpreter(news_article=article)
        lang.interpret_text()
        response_after = es.count(index=index_name)
        document_counts = {
            "index_name": index_name,
            "count_before": response_before["count"],
            "count_after": response_after["count"]
        }
        return jsonify({"status": "success", "message": f"Total documents in '{index_name}': {document_counts}"}), 200
    except Exception as e:
        endpoint: str = "/news/psycho_linguistics/v1"
        logging.error(f"Error: {e} | {endpoint} | at {datetime.now()}")
        return jsonify({"status": "error", "message": f"Error: {e} | {endpoint} | at {datetime.now()}"})

# linguistics relationships NEO4j + Elasticsearch
from repos.linguistics.LinguisticAnalyzerNeo4j import LinguisticAnalyzerNeo4j
@app.route("/news/linguistics_analyze_relationships/v1", methods=["POST"])
def news_linguistics_analyze_relationships_v1():
    try:
        arguments: dict = request.get_json()
        keys = ["linguistics_analysis_id"]
        missing_keys = [key for key in keys if key not in arguments]
        analyzer = LinguisticAnalyzerNeo4j(
            neo4j_uri="bolt://10.42.0.243:7687",
            neo4j_user="neo4j",
            neo4j_password="rootroot",
            elasticsearch_host="http://10.42.0.243:9200"
        )
        analyzer.create_index_mapping()
        if missing_keys:
            raise ValueError(f"Missing keys in article: {', '.join(missing_keys)}")
        es = Elasticsearch("http://10.42.0.243:9200") # Replace with your Elasticsearch host
        index_name = "linguistic_relationships" # Replace with your index name
        
        response_before = es.count(index=index_name)
        analyzer.generate_linguistic_insight(arguments["linguistics_analysis_id"])
        response_after = es.count(index=index_name)
        document_counts = {
            "index_name": index_name,
            "count_before": response_before["count"],
            "count_after": response_after["count"]
        }
        return jsonify({"status": "success", "message": f"Total documents in '{index_name}': {document_counts}"}), 200
    except Exception as e:
        endpoint: str = "/news/psycho_linguistics/v1"
        logging.error(f"Error: {e} | {endpoint} | at {datetime.now()}")
        return jsonify({"status": "error", "message": f"Error: {e} | {endpoint} | at {datetime.now()}"})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)