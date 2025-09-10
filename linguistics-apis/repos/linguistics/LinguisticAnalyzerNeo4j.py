import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import re
import uuid
from elasticsearch8 import Elasticsearch

# Required imports (install with: pip install neo4j elasticsearch spacy textstat)
from neo4j import GraphDatabase
from elasticsearch8 import Elasticsearch
import spacy
import textstat
from collections import Counter, defaultdict
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@dataclass
class LinguisticInsight:
    """Data class for storing linguistic insights in JSON-serializable format"""
    document_id: str
    original_text: str
    timestamp: str
    phonological_complexity: float
    morphological_richness: float
    syntactic_sophistication: float
    semantic_density: float
    lexical_diversity_score: float
    register_formality: float
    # narrative_summary: str
    key_linguistic_features: List[str]
    neo4j_relationships: List[Dict[str, Any]]

class LinguisticAnalyzerNeo4j:
    """
    A comprehensive linguistic analysis system that combines Elasticsearch data retrieval,
    Neo4j graph relationships, and NLP processing to generate informative linguistic paragraphs.
    """
    
    def __init__(self, 
                 neo4j_uri: str, 
                 neo4j_user: str, 
                 neo4j_password: str,
                 elasticsearch_host: str = 'http://10.42.0.243:9200',
                 spacy_model: str = 'en_core_web_sm'):
        """
        Initialize the linguistic analyzer with database connections.
        
        Args:
            neo4j_uri: Neo4j database URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            elasticsearch_host: Elasticsearch host URL
            spacy_model: SpaCy model name to load
        """
        # Initialize Neo4j connection
        self.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        # Initialize Elasticsearch connection
        self.es_client = Elasticsearch([elasticsearch_host])
        
        # Load SpaCy model for NLP processing
        try:
            self.nlp = spacy.load(spacy_model)
        except IOError:
            logging.error(f"Could not load SpaCy model '{spacy_model}'. Please install it with: python -m spacy download {spacy_model}")
            raise
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
        self.es = Elasticsearch([{'host': "10.42.0.243", 'port': 9200, 'scheme': 'http'}])
        self.index_name = 'linguistic_relationships'
    
    def create_index_mapping(self):
        """
        Create Graph Database index mapping for TextInterpreter linguistic relationships results
        """
        mapping = {
            "mappings": {
                "properties": {
                    "document_id":{
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
                    "timestamp": {
                        "type": "keyword"
                    },
                    "phonological_complexity": {
                        "type": "float"
                    },
                    "morphological_richness": {
                        "type": "float"
                    },
                    "syntactic_sophistication": {
                        "type": "float"
                    },
                    "semantic_density": {
                        "type": "float"
                    },
                    "lexical_diversity_score": {
                        "type": "float"
                    },
                    "register_formality": {
                        "type": "float"
                    },
                    "key_linguistic_features": {
                        "type": "keyword" 
                    },
                    "neo4j_relationships": {
                        "properties": {
                            "relationship_type": {
                                "type": "keyword" 
                            },
                            "node_labels": {
                                "type": "keyword" 
                            },
                            "node_type": {
                                "type": "keyword"
                            },
                            "word": {
                                "type": "keyword"
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

    def index_linguistic_analysis(self, document_id, insight):
        """
        Index a linguistic analysis result
        """
        self.create_index_mapping()
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"news_article.title": insight["news_article"]["title"]}}
                    ]
                }
            }
        }
        random_uuid = uuid.uuid4()
        document_id = str(random_uuid)
        result = self.es.search(index=self.index_name, body=query)
        if result['hits']['total']['value'] > 0:
            logger.info(f"Document with index {self.index_name} text: '{insight['news_article']['title']}' already exists, skipping.")
        else:
            
            response = self.es.index(index=self.index_name, id=document_id, body=insight)
            return response

    def close_connections(self):
        """Close database connections"""
        if self.neo4j_driver:
            self.neo4j_driver.close()
    
    def fetch_from_elasticsearch(self, document_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch linguistic data from Elasticsearch index.
        
        Args:
            index_name: Name of the Elasticsearch index
            document_id: Specific document ID to fetch (optional)
            
        Returns:
            Dictionary containing document data
        """
        try:
            index_name = "linguistic_analysis"
            if document_id:
                response = self.es_client.get(index=index_name, id=document_id)
                return response['_source']
            else:
                # Fetch the most recent document if no ID specified
                query = {
                    "query": {"match_all": {}},
                    "sort": [{"timestamp": {"order": "desc"}}],
                    "size": 1
                }
                response = self.es_client.search(index=index_name, body=query)
                if response['hits']['hits']:
                    return response['hits']['hits'][0]['_source']
                else:
                    raise ValueError(f"No documents found in index {index_name}")
        except Exception as e:
            self.logger.error(f"Error fetching from Elasticsearch: {str(e)}")
            raise
    
    def create_neo4j_relationships(self, linguistic_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create and query Neo4j relationships based on linguistic features.
        
        Args:
            linguistic_data: Processed linguistic analysis data
            
        Returns:
            List of relationship data from Neo4j
        """
        relationships = []
        
        with self.neo4j_driver.session() as session:
            # Create document node
            doc_id = linguistic_data['document_id']
            article: dict = linguistic_data['news_article']
            text: str = f"{article['title']} {article['description']} {article['content']}"

            # Create main document node
            session.run("""
                MERGE (d:Document {id: $doc_id})
                SET d.original_text = $text,
                    d.timestamp = $timestamp,
                    d.word_count = $word_count
            """, doc_id=doc_id, 
                text=text,
                timestamp=linguistic_data['timestamp'],
                word_count=len(text.split()))
            
            # Create semantic field nodes and relationships
            semantic_fields = linguistic_data.get('linguistic_interpretation', {}).get('semantic_elements', {}).get('semantic_fields', {})
            for field_type, words in semantic_fields.items():
                for word in words:
                    session.run("""
                        MERGE (s:SemanticField {type: $field_type, word: $word})
                        MERGE (d:Document {id: $doc_id})
                        MERGE (d)-[:CONTAINS_SEMANTIC_FIELD]->(s)
                    """, field_type=field_type, word=word, doc_id=doc_id)
            
            # Create named entity nodes
            named_entities = linguistic_data.get('linguistic_interpretation', {}).get('semantic_elements', {}).get('named_entities', [])
            for entity in named_entities:
                session.run("""
                    MERGE (e:Entity {text: $entity_text, type: $entity_type})
                    MERGE (d:Document {id: $doc_id})
                    MERGE (d)-[:MENTIONS_ENTITY]->(e)
                """, entity_text=entity['entity_text'], entity_type=entity['entity_type'], doc_id=doc_id)
            
            # Query for linguistic patterns
            result = session.run("""
                MATCH (d:Document {id: $doc_id})-[r]-(n)
                RETURN type(r) as relationship_type, labels(n) as node_labels, n.type as node_type, n.word as word
            """, doc_id=doc_id)
            
            for record in result:
                if record['word'] != None:
                    relationships.append({
                        'relationship_type': record['relationship_type'],
                        'node_labels': record['node_labels'],
                        'node_type': record['node_type'],
                        'word': record['word']
                    })
        
        return relationships
    
    def enhanced_nlp_analysis(self, text: str) -> Dict[str, Any]:
        """
        Perform enhanced NLP analysis using SpaCy and additional metrics.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of NLP analysis results
        """
        doc = self.nlp(text)
        
        # Extract linguistic features
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        pos_tags = [(token.text, token.pos_) for token in doc]
        dependencies = [(token.text, token.dep_, token.head.text) for token in doc]
        
        # Calculate readability metrics
        flesch_reading_ease = textstat.flesch_reading_ease(text)
        flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
        
        # Lexical diversity
        words = [token.lemma_.lower() for token in doc if token.is_alpha]
        unique_words = set(words)
        type_token_ratio = len(unique_words) / len(words) if words else 0
        
        return {
            'entities': entities,
            'pos_tags': pos_tags,
            'dependencies': dependencies,
            'flesch_reading_ease': flesch_reading_ease,
            'flesch_kincaid_grade': flesch_kincaid_grade,
            'type_token_ratio': type_token_ratio,
            'sentence_count': len(list(doc.sents)),
            'word_count': len(words),
            'unique_word_count': len(unique_words)
        }
    
    def calculate_linguistic_scores(self, es_data: Dict[str, Any], nlp_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate comprehensive linguistic complexity scores.
        
        Args:
            es_data: Elasticsearch linguistic data
            nlp_data: Enhanced NLP analysis data
            
        Returns:
            Dictionary of linguistic scores
        """
        ling_interp = es_data.get('linguistic_interpretation', {})
        
        # Phonological complexity
        phono = ling_interp.get('phonological_features', {})
        phonological_complexity = (
            phono.get('phonetic_density', 0) * 0.4 +
            len(phono.get('alliteration_instances', [])) * 0.3 +
            len(phono.get('rhyme_patterns', [])) * 0.3
        )
        
        # Morphological richness
        morpho = ling_interp.get('morphological_components', {})
        morphological_richness = (
            morpho.get('morphological_complexity', 0) * 0.5 +
            len(morpho.get('root_word_variations', [])) * 0.3 +
            sum(morpho.get('suffix_usage', {}).values()) * 0.2
        )
        
        # Syntactic sophistication
        syntactic = ling_interp.get('syntactic_structures', {})
        syntactic_sophistication = (
            syntactic.get('syntactic_complexity_score', 0) * 0.6 +
            syntactic.get('average_sentence_length', 0) / 20 * 0.4  # Normalize to ~20 words
        )
        
        # Semantic density
        semantic = ling_interp.get('semantic_elements', {})
        concept_density = semantic.get('concept_density', {})
        semantic_density = (
            concept_density.get('noun_ratio', 0) * 0.4 +
            concept_density.get('verb_ratio', 0) * 0.3 +
            concept_density.get('adjective_ratio', 0) * 0.3
        )
        
        # Lexical diversity
        lexical = ling_interp.get('lexical_diversity', {})
        lexical_diversity_score = (
            lexical.get('type_token_ratio', 0) * 0.4 +
            lexical.get('vocabulary_sophistication', 0) * 0.3 +
            lexical.get('lexical_density', 0) * 0.3
        )
        
        # Register formality
        register = ling_interp.get('register_analysis', {})
        register_formality = register.get('formality_score', 0)
        
        return {
            'phonological_complexity': round(phonological_complexity, 3),
            'morphological_richness': round(morphological_richness, 3),
            'syntactic_sophistication': round(syntactic_sophistication, 3),
            'semantic_density': round(semantic_density, 3),
            'lexical_diversity_score': round(lexical_diversity_score, 3),
            'register_formality': round(register_formality, 3)
        }
    
    def generate_narrative_summary(
        self,
        scores: Dict[str, float],
        nlp_data: Dict[str, Any],
        neo4j_relationships: List[Dict[str, Any]],
        es_data: Dict[str, Any]
    ) -> str:
        """
        Generate a comprehensive narrative summary of linguistic features,
        using real textual evidence from Elasticsearch analysis.
        """

        src = es_data
        article: dict = es_data['news_article']
        text: str = f"{article['title']} {article['description']} {article['content']}"
        original_text = text
        ling = src["linguistic_interpretation"]

        narrative_parts = []

        # === Phonological analysis ===
        phon = ling["phonological_features"]
        if scores['phonological_complexity'] > 2.0:
            if phon["alliteration_instances"]:
                ex = phon["alliteration_instances"][0]
                narrative_parts.append(
                    f"The text exhibits high phonological complexity, visible in alliteration such as "
                    f"'{ex['word1']}' and '{ex['word2']}' sharing the {ex['initial_sound']} sound."
                )
            elif phon["rhyme_patterns"]:
                rp = phon["rhyme_patterns"][0]
                narrative_parts.append(
                    f"The text shows phonological complexity with rhyme between '{rp['word1']}' and '{rp['word2']}'."
                )
        elif scores['phonological_complexity'] > 1.0:
            narrative_parts.append(
                f"The text demonstrates moderate phonological complexity with {phon['estimated_syllable_count']} syllables "
                f"distributed across {len(original_text.split())} words."
            )
        else:
            narrative_parts.append("The text shows relatively simple phonological structure.")

        # === Morphological analysis ===
        morph = ling["morphological_components"]
        if scores['morphological_richness'] > 15:
            if morph["suffix_usage"]:
                suffixes = ", ".join(list(morph["suffix_usage"].keys()))
                narrative_parts.append(
                    f"The text displays rich morphological diversity, with productive suffixes such as {suffixes}."
                )
        elif scores['morphological_richness'] > 8:
            if morph["root_word_variations"]:
                rw = morph["root_word_variations"][0]
                narrative_parts.append(
                    f"The text demonstrates moderate morphological complexity, as seen in the root '{rw['root_word']}' "
                    f"appearing with variations like {', '.join(rw['variations'][:3])}."
                )
        else:
            narrative_parts.append("The text presents a simple morphological structure.")

        # === Syntactic analysis ===
        synt = ling["syntactic_structures"]
        if scores['syntactic_sophistication'] > 25:
            narrative_parts.append(
                f"The syntactic structure is highly sophisticated, with an average sentence length of "
                f"{synt['average_sentence_length']} words and patterns like {synt['common_syntactic_patterns'][0]['pattern']}."
            )
        elif scores['syntactic_sophistication'] > 15:
            narrative_parts.append(
                f"The syntax demonstrates moderate complexity with {len(synt['clause_analysis'])} embedded clauses."
            )
        else:
            narrative_parts.append("The syntactic structure is relatively straightforward.")

        # === Semantic and lexical analysis ===
        sem = ling["semantic_elements"]
        lex = ling["lexical_diversity"]

        if scores['semantic_density'] > 0.4:
            ents = [e["entity_text"] for e in sem["named_entities"]]
            narrative_parts.append(
                f"The text is semantically dense, covering multiple fields (e.g., social: 'neighbors', "
                f"motion: 'volunteers') and named entities such as {', '.join(ents)}."
            )

        if scores['lexical_diversity_score'] > 0.7:
            narrative_parts.append(
                f"The vocabulary usage is exceptionally diverse with {lex['unique_words']} unique words "
                f"out of {lex['total_words']} total, yielding a TTR of {lex['type_token_ratio']:.2f}."
            )

        # === Graph relationships ===
        entity_count = len([
            r for r in neo4j_relationships
            if 'MENTIONS_ENTITY' in r.get('relationship_type', '')
        ])
        if entity_count > 0:
            narrative_parts.append(
                f"The graph analysis reveals {entity_count} distinct entity relationships."
            )

        # === Reading level ===
        if nlp_data['flesch_reading_ease'] > 70:
            narrative_parts.append("The text is highly accessible with easy readability.")
        elif nlp_data['flesch_reading_ease'] > 30:
            narrative_parts.append("The text maintains moderate readability.")
        else:
            narrative_parts.append("The text presents reading challenges with complex structure.")

        # === Register analysis ===
        reg = ling["register_analysis"]
        style = reg["register_classification"]

        if style == "neutral":
            narrative_parts.append(
                f"The register of the text is neutral, balancing between formal and informal tones. "
                f"This is supported by a formality score of {reg['formality_score']:.2f}, with "
                f"{reg['formal_language_indicators']} formal and {reg['informal_language_indicators']} informal markers detected."
            )
        elif style == "formal":
            narrative_parts.append(
                f"The text adopts a formal register, as shown by the absence of contractions and colloquial forms."
            )
        elif style == "informal":
            narrative_parts.append(
                f"The text leans toward an informal register, with contractions and conversational markers."
            )

        # Explicit supporting proof
        if reg["contraction_usage"] > 0:
            narrative_parts.append(
                f"A contraction was detected (contraction_usage={reg['contraction_usage']}), "
                f"which is a hallmark of conversational style."
            )

        if reg["technical_vocabulary_usage"] == 0:
            narrative_parts.append(
                "No technical or specialized vocabulary was found, reinforcing the neutral/conversational character of the passage."
            )

        if reg["register_features"]["conversational"]:
            narrative_parts.append(
                "The presence of conversational features aligns with the narrative flow, "
                "favoring directness over academic or literary style."
            )

        return " ".join(narrative_parts)


    
    def extract_key_features(self, es_data: Dict[str, Any], scores: Dict[str, float]) -> List[str]:
        """
        Extract key linguistic features from the analysis.
        
        Args:
            es_data: Elasticsearch data
            scores: Linguistic scores
            
        Returns:
            List of key linguistic features
        """
        features = []
        
        ling_interp = es_data.get('linguistic_interpretation', {})
        
        # High-scoring features
        if scores['phonological_complexity'] > 1.5:
            features.append("Complex phonological patterns")
        
        if scores['morphological_richness'] > 10:
            features.append("Rich morphological structure")
        
        if scores['syntactic_sophistication'] > 20:
            features.append("Sophisticated syntax")
        
        # Specific linguistic elements
        alliterations = ling_interp.get('phonological_features', {}).get('alliteration_instances', [])
        if alliterations:
            features.append(f"Alliteration: {len(alliterations)} instances")
        
        entities = ling_interp.get('semantic_elements', {}).get('named_entities', [])
        if entities:
            entity_types = set(e['entity_type'] for e in entities)
            features.append(f"Named entities: {', '.join(entity_types)}")
        
        register = ling_interp.get('register_analysis', {}).get('register_classification', 'unknown')
        if register != 'unknown':
            features.append(f"Register: {register}")
        
        return features
    
    def generate_linguistic_insight(self, document_id: Optional[str] = None) -> LinguisticInsight:
        """
        Generate comprehensive linguistic insight from Elasticsearch data with Neo4j relationships.
        
        Args:
            index_name: Elasticsearch index name
            document_id: Optional specific document ID
            
        Returns:
            LinguisticInsight object with all analysis data
        """
        try:
            # Fetch data from Elasticsearch
            es_data = self.fetch_from_elasticsearch(document_id)
            
            # Enhanced NLP analysis
            article: dict = es_data['news_article']
            text: str = f"{article['title']} {article['description']} {article['content']}"
            original_text = text
            nlp_data = self.enhanced_nlp_analysis(original_text)
            
            # Create Neo4j relationships
            neo4j_relationships = self.create_neo4j_relationships(es_data)
            
            # Calculate linguistic scores
            scores = self.calculate_linguistic_scores(es_data, nlp_data)
            
            # Generate narrative summary
            # narrative = self.generate_narrative_summary(scores, nlp_data, neo4j_relationships, es_data)
            
            # Extract key features
            key_features = self.extract_key_features(es_data, scores)
            
            insight = {
                "document_id": es_data.get('document_id', ''),
                "news_article": article,
                "timestamp": es_data.get('timestamp', datetime.now().isoformat()),
                "phonological_complexity": scores['phonological_complexity'],
                "morphological_richness": scores['morphological_richness'],
                "syntactic_sophistication": scores['syntactic_sophistication'],
                "semantic_density": scores['semantic_density'],
                "lexical_diversity_score": scores['lexical_diversity_score'],
                "register_formality": scores['register_formality'],
                "key_linguistic_features": key_features,
                "neo4j_relationships": neo4j_relationships,
            }
            self.index_linguistic_analysis(insight["document_id"], insight)
            self.logger.info(f"Successfully generated linguistic insight for document {insight['document_id']}")
            return insight
            
        except Exception as e:
            self.logger.error(f"Error generating linguistic insight: {str(e)}")
            raise
    
    def to_json(self, insight: dict) -> str:
        """
        Convert LinguisticInsight to JSON string.
        
        Args:
            insight: LinguisticInsight object
            
        Returns:
            JSON string representation
        """
        return json.dumps(insight, indent=2, ensure_ascii=False)

# Example usage
if __name__ == "__main__":
    # Initialize the analyzer
    analyzer = LinguisticAnalyzerNeo4j(
        neo4j_uri="bolt://10.42.0.243:7687",
        neo4j_user="neo4j",
        neo4j_password="rootroot",
        elasticsearch_host="http://10.42.0.243:9200"
    )
    
    try:
        # Generate linguistic insight
        insight = analyzer.generate_linguistic_insight("linguistic_analysis", "15cb7cb7-8605-4272-a9a4-3706b6cc85ed")
        
        # Convert to JSON
        json_output = analyzer.to_json(insight)
        print("Generated Linguistic Insight:")
        print(json_output)
        
        # Access specific components
        # print(f"\nNarrative Summary: {insight.narrative_summary}")
        print(f"Key Features: {', '.join(insight['key_linguistic_features'])}")
        
    finally:
        # Clean up connections
        analyzer.close_connections()