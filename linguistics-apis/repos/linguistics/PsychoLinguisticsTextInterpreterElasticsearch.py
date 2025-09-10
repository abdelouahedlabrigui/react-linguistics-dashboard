from elasticsearch8 import Elasticsearch
import json
from datetime import datetime
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class PsychoLinguisticsTextInterpreterElasticsearch:
    def __init__(self, host='10.42.0.243', port=9200):
        self.es = Elasticsearch([{'host': host, 'port': port, 'scheme': 'http'}])
        self.index_name = 'psycho_linguistic_analysis'
    
    def create_index_mapping(self):
        """
        Create Elasticsearch index mapping for TextInterpreter psycho-linguistic analysis results
        """
        mapping = {
            "mappings": {
                "properties": {
                    "timestamp": {
                        "type": "date"
                    },
                    "document_id": {
                        "type": "keyword"
                    },
                    "news_article": {
                        "properties": {
                            "source": {
                                "type": "keyword"
                            },
                            "author": {
                                "type": "keyword"
                            },
                            "title": {
                                "type": "keyword"
                            },
                            "description": {
                                "type": "keyword"
                            },
                            "url": {
                                "type": "keyword"
                            },
                            "urlToImage": {
                                "type": "keyword"
                            },
                            "publishedAt": {
                                "type": "keyword"
                            },
                            "content": {
                                "type": "keyword"
                            },
                            "timestamp": {
                                "type": "date"
                            },
                            "document_id": {
                                "type": "keyword"
                            }
                        },
                    },
                    "psycholinguistic_interpretation": {
                        "properties": {
                            
                            "processing_fluency": {
                                "properties": {
                                "level": { "type": "keyword" },
                                "contributing_factors": { "type": "text" }
                                }
                            },
                            "cognitive_load": {
                                "properties": {
                                "score": { "type": "float" },
                                "contributing_elements": { "type": "text" }
                                }
                            },
                            "emotional_valence": {
                                "properties": {
                                "velance": { "type": "keyword" },
                                "contributing_words_phrases": { "type": "text" }
                                }
                            },
                            "priming_effects": { "type": "text" },
                            "working_memory_demands": { "type": "text" },
                            "attention_allocation": { "type": "text" },
                            "reader_model": { "type": "keyword" }
                        }
                    }
                }
            }
        }
        # Create the index
        if not self.es.indices.exists(index=self.index_name):        
            response = self.es.indices.create(index=self.index_name, body=mapping)
            return response
    def index_linguistic_analysis(self, document_id, news_article: dict, psycho_linguistic_data):
        """
        Index a linguistic analysis result
        """
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"news_article.title": news_article['title']}}
                    ]
                }
            }
        }
        result = self.es.search(index=self.index_name, body=query)
        if result['hits']['total']['value'] > 0:
            logger.info(f"Document with index {self.index_name} text: '{news_article['title']}' already exists, skipping.")
        else:
            doc = {
                'document_id': document_id,
                'news_article': news_article,
                'timestamp': str(datetime.utcnow().isoformat()),  # You can use datetime.utcnow().isoformat()
                'psycholinguistic_interpretation': psycho_linguistic_data["psycholinguistic_interpretation"]
            }
            
            response = self.es.index(index=self.index_name, id=document_id, body=doc)
            return response