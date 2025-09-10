from elasticsearch8 import Elasticsearch
import json
from datetime import datetime
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TextInterpreterElasticsearch:
    def __init__(self, host='10.42.0.243', port=9200):
        self.es = Elasticsearch([{'host': host, 'port': port, 'scheme': 'http'}])
        self.index_name = 'linguistic_analysis'
    
    def create_index_mapping(self):
        """
        Create Elasticsearch index mapping for TextInterpreter linguistic analysis results
        """
        mapping = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 1,
                "analysis": {
                    "analyzer": {
                        "linguistic_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "stop"]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
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
                    "linguistic_interpretation": {
                        "properties": {
                            # Phonological Features
                            "phonological_features": {
                                "properties": {
                                    "vowel_consonant_ratio": {"type": "float"},
                                    "total_vowels": {"type": "integer"},
                                    "total_consonants": {"type": "integer"},
                                    "alliteration_instances": {
                                        # "type": "nested",
                                        # "properties": {
                                        #     "word1": {"type": "keyword"},
                                        #     "word2": {"type": "keyword"},
                                        #     "initial_sound": {"type": "keyword"}
                                        # }
                                        "type": "object",
                                        "dynamic": True
                                    },
                                    "rhyme_patterns": {
                                        # "type": "nested",
                                        # "properties": {
                                        #     "word1": {"type": "keyword"},
                                        #     "word2": {"type": "keyword"},
                                        #     "rhyme_ending": {"type": "keyword"}
                                        # }
                                        "type": "object",
                                        "dynamic": True
                                    },
                                    "estimated_syllable_count": {"type": "integer"},
                                    "phonetic_density": {"type": "float"}
                                }
                            },
                            
                            # Morphological Components
                            "morphological_components": {
                                "properties": {
                                    "prefix_usage": {
                                        "type": "object",
                                        "dynamic": True
                                    },
                                    "suffix_usage": {
                                        "type": "object",
                                        "dynamic": True
                                    },
                                    "morphological_complexity": {"type": "integer"},
                                    "root_word_variations": {
                                        # "type": "nested",
                                        "properties": {
                                            "original_word": {"type": "keyword"},
                                            "root_word": {"type": "keyword"},
                                            "variations": {"type": "keyword"}
                                        }
                                    },
                                    "pos_tag_distribution": {
                                        "properties": {
                                            "tag": {"type": "keyword"},
                                            "distribution": {"type": "keyword"}
                                        }
                                    },
                                    "inflectional_variants": {
                                        "type": "object",
                                        "dynamic": True
                                    }
                                }
                            },
                            
                            # Syntactic Structures
                            "syntactic_structures": {
                                "properties": {
                                    "sentence_structures": {
                                        "type": "nested",
                                        "properties": {
                                            "pos_sequence": {"type": "keyword"},
                                            "structure_id": {"type": "integer"}
                                        }
                                    },
                                    "average_sentence_length": {"type": "float"},
                                    "clause_analysis": {
                                        "type": "nested",
                                        "properties": {
                                            "clause_pos": {"type": "keyword"},
                                            "clause_id": {"type": "integer"}
                                        }
                                    },
                                    "syntactic_complexity_score": {"type": "float"},
                                    "common_syntactic_patterns": {
                                        "type": "object",
                                        "dynamic": True
                                        # "properties": {
                                        #     "pattern": {"type": "keyword"},
                                        #     "count": {"type": "integer"}
                                        # }
                                    },
                                    "sentence_types": {
                                        "properties": {
                                            "declarative": {"type": "integer"},
                                            "interrogative": {"type": "integer"},
                                            "exclamatory": {"type": "integer"},
                                            "imperative": {"type": "integer"}
                                        }
                                    }
                                }
                            },
                            
                            # Semantic Elements
                            "semantic_elements": {
                                "properties": {
                                    "named_entities": {
                                        # "type": "nested",
                                        # "properties": {
                                        #     "entity_text": {"type": "keyword"},
                                        #     "entity_type": {"type": "keyword"}
                                        # }
                                        "type": "object",
                                        "dynamic": True
                                    },
                                    "semantic_fields": {
                                        "properties": {
                                            "emotion": {"type": "keyword"},
                                            "temporal": {"type": "keyword"},
                                            "spatial": {"type": "keyword"}
                                        }
                                    },
                                    "concept_density": {
                                        "properties": {
                                            "noun_ratio": {"type": "float"},
                                            "verb_ratio": {"type": "float"},
                                            "adjective_ratio": {"type": "float"}
                                        }
                                    },
                                    "high_frequency_concepts": {
                                        "type": "object",
                                        "dynamic": True
                                    },
                                    "semantic_richness": {"type": "float"},
                                    "content_words": {"type": "keyword"}
                                }
                            },
                            
                            # Lexical Diversity
                            "lexical_diversity": {
                                "properties": {
                                    "type_token_ratio": {"type": "float"},
                                    "total_words": {"type": "integer"},
                                    "unique_words": {"type": "integer"},
                                    "hapax_legomena": {"type": "integer"},
                                    "average_word_length": {"type": "float"},
                                    "vocabulary_sophistication": {"type": "float"},
                                    "lexical_density": {"type": "float"},
                                    "most_frequent_words": {
                                        "type": "object",
                                        "dynamic": True
                                    }
                                }
                            },
                            
                            # Register Analysis
                            "register_analysis": {
                                "properties": {
                                    "formality_score": {"type": "float"},
                                    "register_classification": {"type": "keyword"},
                                    "formal_language_indicators": {"type": "integer"},
                                    "informal_language_indicators": {"type": "integer"},
                                    "technical_vocabulary_usage": {"type": "integer"},
                                    "contraction_usage": {"type": "integer"},
                                    "average_sentence_complexity": {"type": "float"},
                                    "passive_voice_indicators": {"type": "integer"},
                                    "register_features": {
                                        "properties": {
                                            "academic": {"type": "boolean"},
                                            "conversational": {"type": "boolean"},
                                            "technical": {"type": "boolean"},
                                            "literary": {"type": "boolean"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        # Create the index
        if not self.es.indices.exists(index=self.index_name):        
            response = self.es.indices.create(index=self.index_name, body=mapping)
            return response
    
    def index_linguistic_analysis(self, document_id, linguistic_data):
        """
        Index a linguistic analysis result
        """
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"news_article.titles": linguistic_data["news_article"]["title"]}}
                    ]
                }
            }
        }
        result = self.es.search(index=self.index_name, body=query)
        if result['hits']['total']['value'] > 0:
            logger.info(f"Document with index {self.index_name} text: '{linguistic_data['news_article']['title']}' already exists, skipping.")
        else:
            doc = {
                'document_id': document_id,
                'news_article': linguistic_data["news_article"],
                'timestamp': '2024-01-01T00:00:00',  # You can use datetime.utcnow().isoformat()
                'linguistic_interpretation': self._transform_data_for_elasticsearch(linguistic_data["linguistic_interpretation"])
            }
            
            response = self.es.index(index=self.index_name, id=document_id, body=doc)
            return response
        
    def _transform_data_for_elasticsearch(self, data):
        transformed = {}

        for section, data in data.items():
            if section == 'phonological_features':
                transformed[section] = {
                    **data,
                    'alliteration_instances': data.get('alliteration_instances', []),
                    'rhyme_patterns': data.get('rhyme_patterns', [])
                }
            elif section == 'morphological_components':
                transformed[section] = {
                    **data,
                    'root_word_variations': data.get('root_word_variations', []),
                    'inflectional_variants': data.get('inflectional_variants', [])
                }
            elif section == 'syntactic_structures':
                transformed[section] = {
                    **data,
                    'sentence_structures': [
                        {'pos_sequence': pos_tags, 'structure_id': idx}
                        for idx, pos_tags in enumerate(data.get('sentence_structures', []))
                    ],
                    'clause_analysis': [
                        {'clause_pos': clause, 'clause_id': idx}
                        for idx, clause in enumerate(data.get('clause_analysis', []))
                    ]
                }
            elif section == 'semantic_elements':
                transformed[section] = {
                    **data,
                    'named_entities': data.get('named_entities', [])
                }
            elif section == 'pragmatic_inferences':
                transformed[section] = {
                    **data,
                    'modal_verb_usage': [
                        {'text_excerpt': item[0], 'modal_verbs': item[1]}
                        for item in data.get('modal_verb_usage', [])
                        if isinstance(item, (list, tuple))
                    ]
                }
            else:
                transformed[section] = data
        
        return transformed
    
    def search_by_complexity(self, min_complexity=None, max_complexity=None):
        """
        Search documents by syntactic complexity score
        """
        query = {"query": {"range": {"linguistic_interpretation.syntactic_structures.syntactic_complexity_score": {}}}}
        
        if min_complexity is not None:
            query["query"]["range"]["linguistic_interpretation.syntactic_structures.syntactic_complexity_score"]["gte"] = min_complexity
        if max_complexity is not None:
            query["query"]["range"]["linguistic_interpretation.syntactic_structures.syntactic_complexity_score"]["lte"] = max_complexity
        
        return self.es.search(index=self.index_name, body=query)
    
    def search_by_register(self, register_type):
        """
        Search documents by register classification
        """
        query = {
            "query": {
                "term": {
                    "linguistic_interpretation.register_analysis.register_classification": register_type
                }
            }
        }
        return self.es.search(index=self.index_name, body=query)
    
    def get_lexical_diversity_stats(self):
        """
        Get aggregated lexical diversity statistics
        """
        query = {
            "size": 0,
            "aggs": {
                "avg_type_token_ratio": {
                    "avg": {"field": "linguistic_interpretation.lexical_diversity.type_token_ratio"}
                },
                "avg_vocabulary_sophistication": {
                    "avg": {"field": "linguistic_interpretation.lexical_diversity.vocabulary_sophistication"}
                },
                "complexity_distribution": {
                    "histogram": {
                        "field": "linguistic_interpretation.syntactic_structures.syntactic_complexity_score",
                        "interval": 0.5
                    }
                }
            }
        }
        return self.es.search(index=self.index_name, body=query)

