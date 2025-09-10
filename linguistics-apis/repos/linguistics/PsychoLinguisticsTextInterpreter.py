import re
import uuid
import nltk
from textblob import TextBlob
from collections import Counter
import numpy as np
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from sklearn.cluster import KMeans
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize
from nltk.corpus import brown
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from repos.linguistics.PsychoLinguisticsTextInterpreterElasticsearch import PsychoLinguisticsTextInterpreterElasticsearch
import logging
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')
nltk.download('brown')
nltk.download('sentiwordnet')

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

class PsychoLinguisticsTextInterpreterCreateIndex:
    def __init__(self):
        self.es_handler = PsychoLinguisticsTextInterpreterElasticsearch()
    def create_index(self):
        self.es_handler.create_index_mapping()


class PsychoLinguisticsTextInterpreter:
    def __init__(self, news_article: dict):
        self.stop_words = set(stopwords.words('english'))
        
        self.news_article: dict = news_article
        title: str = self.news_article["title"]
        description: str = self.news_article["description"]
        content: str = self.news_article["content"]
        
        self.text: str = f"{title} {description} {content}."
        self.blob = TextBlob(self.text)
        self.word_vectors = {} # load real embeddings here
        self.classifier, self.label_map = self._train_semantic_classifier()
        self.learned_intensifiers = self._learn_intensifiers()
    
    def _tokenize(self):
        """Tokenize & lowercase."""
        return re.findall(r'\b[a-z]+\b', self.text.lower())
    
    def _get_sentiment(self, word, pos):
        """Use SentiWordNet to classify positive/negative dynamically."""
        try:
            synsets = list(swn.senti_synsets(word, pos))
            if not synsets:
                return None
            score = synsets[0]
            if score.pos_score() > score.neg_score():
                return "positive"
            elif score.neg_score() > score.pos_score():
                return "negative"
        except:
            return None
        return None
    
    def categorize_words(self):
        tokens = self._tokenize()
        tagged = pos_tag(tokens)
        
        categories = {
            "positive_words": [],
            "negative_words": [],
            "complex_conjunctions": [],
            "attention_words": []
        }
        
        # dynamic attention/conjunction heuristics
        discourse_markers = {"however", "therefore", "moreover", "furthermore", "consequently"}
        attention_patterns = {"urgent", "important", "critical", "suddenly", "immediately"}
        
        for word, tag in tagged:
            # Sentiment classification
            wn_pos = None
            if tag.startswith("J"): wn_pos = wordnet.ADJ
            elif tag.startswith("V"): wn_pos = wordnet.VERB
            elif tag.startswith("N"): wn_pos = wordnet.NOUN
            elif tag.startswith("R"): wn_pos = wordnet.ADV
            
            sentiment = self._get_sentiment(word, wn_pos) if wn_pos else None
            if sentiment == "positive":
                categories["positive_words"].append(word)
            elif sentiment == "negative":
                categories["negative_words"].append(word)
            
            # Complex conjunctions
            if word in discourse_markers:
                categories["complex_conjunctions"].append(word)
            
            # Attention words
            if word in attention_patterns:
                categories["attention_words"].append(word)
        
        return categories
    def interpret_text(self):
        
        sentences = sent_tokenize(self.text)
        words = word_tokenize(self.text.lower())
        
        # Get POS tags
        pos_tags = nltk.pos_tag(words)
        
        data: dict = {
            'psycholinguistic_interpretation': {
                'processing_fluency': self._analyze_processing_fluency(sentences, words, pos_tags),
                'cognitive_load': self._analyze_cognitive_load(sentences, words, pos_tags),
                'emotional_valence': self._analyze_emotional_valence(words),
                'priming_effects': self._analyze_priming_effects(words, pos_tags),
                'working_memory_demands': self._analyze_working_memory_demands(sentences, pos_tags),
                'attention_allocation': self._analyze_attention_allocation(words, sentences),
                'reader_model': self._analyze_reader_model(words, sentences, pos_tags)
            }
        }
        es_handler = PsychoLinguisticsTextInterpreterElasticsearch()
        logger.info("Creating index...")
        response = es_handler.create_index_mapping()
        logger.info(f"Index created: {response}")
        random_uuid = uuid.uuid4()
        document_id = str(random_uuid)
        index_response = es_handler.index_linguistic_analysis(document_id=document_id, news_article=self.news_article, psycho_linguistic_data=data)
        logger.info(f"Document indexed: {index_response}")
    
    def _analyze_processing_fluency(self, sentences, words, pos_tags):
        factors = []
        
        # Analyze sentence length
        avg_sentence_length = np.mean([len(word_tokenize(s)) for s in sentences])
        if avg_sentence_length > 25:
            factors.append("long sentences reduce fluency")
            fluency = "low"
        elif avg_sentence_length < 12:
            factors.append("short sentences enhance fluency")
            fluency = "high"
        else:
            factors.append("moderate sentence length")
            fluency = "medium"
            
        # Analyze vocabulary complexity
        long_words = [word for word in words if len(word) > 6 and word.isalpha()]
        if len(long_words) / len(words) > 0.3:
            factors.append("complex vocabulary reduces fluency")
            if fluency == "high":
                fluency = "medium"
            elif fluency == "medium":
                fluency = "low"
        else:
            factors.append("accessible vocabulary")
            
        # Check for passive voice
        passive_indicators = ['was', 'were', 'been', 'being'] + [word for word, tag in pos_tags if tag == 'VBN']
        if len(passive_indicators) > len(sentences):
            factors.append("frequent passive constructions")
            
        # Check for complex conjunctions
        categorized = self.categorize_words()
        complex_conj_count = sum(1 for word in words if word in categorized["complex_conjunctions"])
        if complex_conj_count > 2:
            factors.append("complex logical connectors")
            
        return {
            'level': fluency,
            'contributing_factors': factors
        }
    
    def _analyze_cognitive_load(self, sentences, words, pos_tags):
        load_factors = []
        load_score = 0.0
        
        # Sentence complexity
        avg_sentence_length = np.mean([len(word_tokenize(s)) for s in sentences])
        if avg_sentence_length > 20:
            load_factors.append("long sentences increase processing demands")
            load_score += 0.3
            
        # Subordinate clauses
        subordinate_markers = ['that', 'which', 'who', 'whom', 'whose', 'where', 'when', 'while', 'although', 'because', 'since', 'if', 'unless']
        subordinate_count = sum(1 for word in words if word in subordinate_markers)
        if subordinate_count > len(sentences) * 0.5:
            load_factors.append("multiple subordinate clauses")
            load_score += 0.2
            
        # Pronoun resolution complexity
        pronouns = [word for word, tag in pos_tags if tag in ['PRP', 'PRP$']]
        if len(pronouns) > len(sentences) * 0.3:
            load_factors.append("frequent pronoun resolution required")
            load_score += 0.15
            
        # Garden path potential
        garden_path_indicators = ['the horse raced past', 'the man the woman', 'since jay always']
        for indicator in garden_path_indicators:
            if indicator.lower() in self.text.lower():
                load_factors.append("garden-path sentence structure")
                load_score += 0.25
                break
                
        # Anaphoric references
        anaphoric_words = ['this', 'that', 'these', 'those', 'such', 'it', 'they']
        anaphoric_count = sum(1 for word in words if word in anaphoric_words)
        if anaphoric_count > len(sentences) * 0.2:
            load_factors.append("multiple anaphoric references")
            load_score += 0.1
            
        # Technical terminology
        technical_pos = [word for word, tag in pos_tags if tag == 'NN' and len(word) > 7]
        if len(technical_pos) > len(words) * 0.1:
            load_factors.append("specialized terminology")
            load_score += 0.2
            
        return {
            'score': min(load_score, 1.0),
            'contributing_elements': load_factors
        }
    
    def _learn_intensifiers(self):
        """Extract common intensifier candidates from Brown corpus"""
        words = brown.words(categories='reviews')
        tagged = pos_tag(words)

        # Intensifiers are usually adverbs (RB) before adjectives (JJ)
        candidates = []
        for i in range(len(tagged)-1):
            word, tag = tagged[i]
            next_word, next_tag = tagged[i+1]
            if tag.startswith('RB') and next_tag.startswith('JJ'):
                candidates.append(str(word).lower())
        
        # return frequent ones
        return {w for w in candidates if len(w) > 2 and candidates.count(w) > 2}
    
    def _analyze_emotional_valence(self, words: list):
        contributing_words = []

        categories = self.categorize_words()
        positive_found = categories["positive_words"]
        negative_found = categories["negative_words"]

        contributing_words.extend(positive_found)
        contributing_words.extend(negative_found)

        # TextBlob sentiment
        polarity = self.blob.sentiment.polarity
        if polarity > 0.1:
            valence = "positive"
        elif polarity < -0.1:
            valence = "negative"
        else:
            valence = "neutral"

        # Dynamically learned intensifiers
        found_intensifiers = [w for w in words if w in self.learned_intensifiers]
        contributing_words.extend(found_intensifiers)

        return {
            "velance": valence,
            "contributing_words_phrases": list(set(contributing_words)) if contributing_words else ["neutral tone thought"]
        }
    
    def get_vector(self, word):
        """Fetch word embedding vector, fallback to random if missing"""
        if word in self.word_vectors:
            return self.word_vectors[word]
        
        # fallback: random vector
        return np.random.rand(50)

    def _train_semantic_classifier(self, categories=None):
        """
        Train decision tree classifier on semantic categories dynamically.
        Uses WordNet to expand category seed words instead of hardcoding.
        """
        if categories is None:
            categories = [
                "business", "academic", "technology", "sports",
                "science", "health", "politics", "art", 
                "finance", "travel", "food"
            ]

        X, y = [], []
        label_map = {}

        for idx, label in enumerate(categories):
            label_map[idx] = label

            # Expand category with WordNet synonyms & hyponyms
            synonyms = set()
            for synset in wn.synsets(label):
                # Synonyms
                for lemma in synset.lemmas():
                    synonyms.add(lemma.name().lower().replace("_", " "))
                # Hyponyms (more specific terms)
                for hyponym in synset.hyponyms():
                    for lemma in hyponym.lemmas():
                        synonyms.add(lemma.name().lower().replace("_", " "))

            # Convert words into vectors
            for word in synonyms:
                vec = self.get_vector(word)
                if vec is not None:  # safeguard: skip OOV
                    X.append(vec)
                    y.append(idx)

        if not X:  # safeguard: no training data
            return None, {}

        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X, y)

        return clf, label_map
    
    def _classify_word(self, word, min_conf=0.35):
        """Classify a word into a semantic category with confidence threshold."""
        if not self.classifier:
            return None
        
        vec = self.get_vector(word).reshape(1, -1)
        probs = self.classifier.predict_proba(vec)[0]
        label_idx = probs.argmax()
        confidence = probs[label_idx]

        if confidence < min_conf:  # reject weak matches
            return None
        return self.label_map[label_idx]
    
    def _detect_metaphors(self, words: list, pos_tags: list):
        """Detect metaphorical constructions dynamically."""
        metaphors = []
        for i in range(len(pos_tags)-2):
            w1, t1 = pos_tags[i]
            w2, t2 = pos_tags[i+1]
            w3, t3 = pos_tags[i+2]

            # "X is Y" metaphor
            if str(t1).startswith("NN") and str(w2).lower() == "is" and str(t3).startswith("NN"):
                metaphors.append(f"metaphorical priming: {w1} is {w3}")
            
            # "X like Y" simile
            if str(w2).lower() == "like" and str(t1).startswith("NN") and str(t3).startswith("NN"):
                metaphors.append(f"metaphorical priming: {w1} like {w3}")

            # "as ADJ as"
            if str(w1).lower() == "as" and str(t2).startswith("JJ") and str(w3).lower() == "as":
                metaphors.append(f"metaphorical priming: {w2} as ...")
            
        return metaphors
    
    def _filter_word(self, word):
        if len(word) < 3:
            return False
        if word.lower() in self.stop_words:
            return False
        if all(ch in ".,!?;:'\"-–—()" for ch in word):
            return False
        return True

    def _analyze_priming_effects(self, words, pos_tags, n_clusters=5):
        priming_elements = []

        # --- Repeated concepts ---
        word_freq = Counter([w for w in words if w not in self.stop_words and len(w) > 3])
        repeated_words = [w for w, freq in word_freq.most_common(10) if freq > 1]
        priming_elements.extend([f"repeated concept: {w}" for w in repeated_words[:5]])

        # --- Semantic fields (dynamic clustering, no fixed categories) ---
        vectors, valid_words = [], []
        for w in words:
            if self._filter_word(w):  # skip stopwords, punctuation, etc.
                vec = self.get_vector(w)
                if vec is not None:
                    vectors.append(vec)
                    valid_words.append(w)

        if vectors:
            X = np.array(vectors)
            n_clusters = min(n_clusters, len(valid_words))  # safeguard
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)

            cluster_map = {}
            for word, label in zip(valid_words, labels):
                cluster_map.setdefault(label, []).append(word)

            # Keep only "activated" clusters (≥3 members)
            for cluster_id, cluster_words in cluster_map.items():
                if len(cluster_words) > 2:
                    priming_elements.append(
                        f"semantic field activation (cluster {cluster_id}): {', '.join(set(cluster_words[:5]))}"
                    )

        # --- Metaphorical language ---
        metaphors = self._detect_metaphors(words, pos_tags)
        priming_elements.extend(metaphors)

        # --- Cultural references (proper nouns) ---
        proper_nouns = [w for w, t in pos_tags if t == "NNP"]
        if proper_nouns:
            priming_elements.extend([f"cultural reference: {noun}" for noun in proper_nouns[:3]])

        return priming_elements if priming_elements else ['minimal priming effects detected']


    
    def _analyze_working_memory_demands(self, sentences, pos_tags):
        memory_demands = []
        
        # Long noun phrases
        noun_phrases = []
        current_np = []
        for word, tag in pos_tags:
            if tag.startswith('N') or tag in ['DT', 'JJ', 'JJR', 'JJS']:
                current_np.append(word)
            else:
                if len(current_np) > 3:
                    noun_phrases.append(' '.join(current_np))
                current_np = []
        
        if noun_phrases:
            memory_demands.extend([f"long noun phrase: {np}" for np in noun_phrases[:3]])
            
        # Embedded clauses
        for sentence in sentences:
            clause_depth = sentence.count('(') + sentence.count(',') + sentence.count(';')
            if clause_depth > 2:
                memory_demands.append(f"deeply embedded clauses in: '{sentence[:50]}...'")
                
        # Postponed information
        for sentence in sentences:
            if 'not only' in sentence.lower() or 'either' in sentence.lower():
                memory_demands.append("postponed information structure")
                
        # Lists requiring retention
        list_indicators = [';', ':', 'first', 'second', 'third', 'finally']
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in list_indicators):
                if len(word_tokenize(sentence)) > 15:
                    memory_demands.append("complex enumeration requiring retention")
                    
        return memory_demands if memory_demands else ['minimal working memory demands']
    
    
    def _analyze_attention_allocation(self, words, sentences):
        attention_elements = []

        # --- 1. Topic shifts (discourse markers at sentence start) ---
        if len(sentences) > 3:
            topic_shift_markers = {
                'however', 'meanwhile', 'suddenly', 'in contrast',
                'on the other hand', 'alternatively', 'nevertheless',
                "conversely", "on the contrary", "by contrast", "instead", 
                "as for", "regarding", "with respect to", "later", "next", 
                "subsequently", "abruptly", "all of a sudden", "at the same time", "incidentally", "by the way"
            }
            for i, sentence in enumerate(sentences[1:], 1):
                sentence_lower = sentence.strip().lower()
                # only look at beginning
                for marker in topic_shift_markers:
                    if sentence_lower.startswith(marker):
                        attention_elements.append(f"topic shift at sentence {i+1} (marker: {marker})")
                        break

        # --- 2. Attention-grabbing vocabulary (dynamic) ---
        categorized = self.categorize_words()
        attention_grabbers = categorized.get("attention_words", [])
        if attention_grabbers:
            attention_elements.extend([f"attention-grabbing: {word}" for word in set(attention_grabbers)])

        # --- 3. Unusual punctuation or formatting ---
        if '!' in self.text:
            attention_elements.append("exclamation points draw attention")
        if self.text.count('?') > len(sentences) * 0.2:
            attention_elements.append("frequent questions engage attention")
        if re.search(r'\b[A-Z]{2,}\b', self.text):
            attention_elements.append("capitalized words for emphasis")

        # --- 4. Quotations ---
        quote_count = self.text.count('"') + self.text.count("'")
        if quote_count > 2:
            attention_elements.append("quoted material draws focus")

        # --- 5. Numbers and statistics ---
        numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', self.text)
        if numbers:
            attention_elements.extend([f"numerical data: {num}" for num in numbers[:3]])

        return attention_elements if attention_elements else ['standard attention flow']
    
    def _analyze_reader_model(self, words, sentences, pos_tags):
        # Analyze complexity indicators
        avg_sentence_length = np.mean([len(word_tokenize(s)) for s in sentences])
        complex_words = [word for word in words if len(word) > 8 and word.isalpha()]
        technical_ratio = len(complex_words) / len(words) if words else 0
        
        # Check for explanatory elements
        explanatory_markers = ['that is', 'in other words', 'specifically', 'for example', 'such as', 'namely', "i.e.", "e.g.", "to put it another way", "in simpler terms", "to be more precise", "particularly", "in particular", "to illustrate", "what this means is", "in fact", "as a result", "consequently", "due to", "because of this"]
        has_explanations = any(marker in self.text.lower() for marker in explanatory_markers)
        
        # Check for assumed knowledge
        technical_terms = [word for word, tag in pos_tags if tag == 'NN' and len(word) > 6]
        acronyms = re.findall(r'\b[A-Z]{2,}\b', self.text)
        
        # Determine reader model
        if avg_sentence_length > 20 and technical_ratio > 0.15 and not has_explanations:
            if acronyms or len(technical_terms) > len(sentences):
                return "expert reader (assumes specialized knowledge)"
            else:
                return "educated reader (assumes general sophistication)"
        elif has_explanations or any(word in self.text.lower() for word in ['basically', 'simply', 'easy', 'straightforward']):
            return "novice reader (provides explanatory support)"
        elif avg_sentence_length < 15 and technical_ratio < 0.05:
            return "general audience (accessible language)"
        else:
            return "educated general reader (balanced complexity)"
        
# if __name__ == "__main__":
#     text = """The ICJ issued an opinion on climate change that stated the obligations of the Paris Agreement come with legal liability. A concern raised by Trump as early as 2017."""
#     news_article: dict = {
#         "source": "None",
#         "author": "Stephanie Mencimer",
#         "title": "They Survived Katrina and Started to Rebuild. Now Trump’s Cuts May Flood Them Out Again.",
#         "description": "After Hurricane Katrina and its related flooding inundated much of New Orleans in 2005, residents of a large Vietnamese community in the Algiers neighborhood were some of the first to return and rebuild. But 20 years later, their neighborhood is still prone t…",
#         "url": "https://www.motherjones.com/politics/2025/08/they-survived-katrina-and-started-to-rebuild-now-trumps-cuts-may-flood-them-out-again/",
#         "urlToImage": "https://www.motherjones.com/wp-content/uploads/2025/08/20250828_new-orleans-flooding_2000.png?w=1200&h=630&crop=1",
#         "publishedAt": "2025-08-28T10:00:00Z",
#         "content": "Mother Jones; Justin Sullivan/Getty; Aaron Schwartz/Pool/Cnp/Zuma Get your news from a source thats not owned and controlled by oligarchs. Sign up for the free Mother Jones Daily. After Hurricane K… [+6817 chars]"
#     }
#     lang = PsychoLinguisticsTextInterpreter(news_article=news_article)
#     data = lang.interpret_text()
#     import json
#     print(json.dumps(data, indent=4))