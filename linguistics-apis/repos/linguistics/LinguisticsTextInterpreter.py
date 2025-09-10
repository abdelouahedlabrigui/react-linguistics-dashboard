import json
import re
import nltk
from nltk import RegexpParser
from collections import Counter, defaultdict
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.corpus import stopwords, wordnet, cmudict
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import spacy
from textstat import flesch_reading_ease, flesch_kincaid_grade
import string
from repos.linguistics.LinguisticsTextInterpreterElasticsearch import TextInterpreterElasticsearch
import uuid

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
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')
try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker')
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
nltk.download("cmudict")
nltk.download('omw-1.4')

class LinguisticsTextInterpreter:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # self.cmu_dict = cmudict.dict()

    def create_index(self):
        es_handler = TextInterpreterElasticsearch()
        # Create the index
        logger.info("Creating index...")
        response = es_handler.create_index_mapping()
        logger.info(f"Index created: {response}")
        
    def interpret_text(self, news_article: dict):
        title: str = news_article["title"]
        description: str = news_article["description"]
        content: str = news_article["content"]
        
        text: str = f"{title} {description} {content}."

        tokens = word_tokenize(text.lower())
        sentences = sent_tokenize(text)
        pos_tags = pos_tag(word_tokenize(text))
        words = [word for word in tokens if word.isalpha()]
        
        data: dict = {
            "news_article": {
                "source": news_article["source"],
                "author": news_article["author"],
                "title": news_article["title"],
                "description": news_article["description"],
                "url": news_article["url"],
                "urlToImage": news_article["urlToImage"],
                "publishedAt": news_article["publishedAt"],
                "content": news_article["content"],
            },
            'linguistic_interpretation': {
                'phonological_features': self._analyze_phonological_features(text, words),
                'morphological_components': self._analyze_morphological_components(words, pos_tags),
                'syntactic_structures': self._analyze_syntactic_structures(sentences, pos_tags),
                'semantic_elements': self._analyze_semantic_elements(text, words, pos_tags),
                'lexical_diversity': self._analyze_lexical_diversity(words, tokens),
                'register_analysis': self._analyze_register(text, words, sentences)
            }
        }
        data = json.dumps(data, indent=4)

        es_handler = TextInterpreterElasticsearch()
        random_uuid = uuid.uuid4()
        document_id = str(random_uuid)

        

        with open('/room/labrigui/Software/microservices/python-software/newsapp/apis/newsquery/repos/linguistics/linguistics.json', 'w') as f:
            f.writelines(data)

        with open('/room/labrigui/Software/microservices/python-software/newsapp/apis/newsquery/repos/linguistics/linguistics.json', 'r') as file:
            python_dict_from_file = json.load(file)

        logger.info("Indexing document...")
        index_response = es_handler.index_linguistic_analysis(document_id, python_dict_from_file)
        logger.info(f"Document indexed: {index_response}")
        # Example searches
        logger.info("\nSearching by complexity...")
        complexity_results = es_handler.search_by_complexity(min_complexity=3.0)
        logger.info(f"Found {complexity_results['hits']['total']['value']} documents with complexity >= 3.0")
        
        logger.info("\nGetting lexical diversity stats...")
        stats = es_handler.get_lexical_diversity_stats()
        logger.info(f"Stats: {stats['aggregations']}")
    
    def _analyze_phonological_features(self, text, words):
        vowels = set("AEIOU")
        consonants = set("BCDFGHJKLMNPQRSTVWXYZ")
        cmu_dict = cmudict.dict()

        # Count vowels/consonants from text (orthographic fallback)
        vowel_count = sum(1 for char in text.upper() if char in vowels)
        consonant_count = sum(1 for char in text.upper() if char in consonants)

        # Get phonetic forms (fall back to spelling if not found)
        phonetics = {}
        for word in words:
            word_lower = word.lower()
            phonetics[word] = cmu_dict.get(word_lower, [[c for c in word_lower]])
            # logger.info(f"Word: {word}, Phonetics: {phonetics[word]}")  # Debugging

        # --- Alliteration (same initial phoneme) ---
        alliterative_pairs = []
        for i in range(len(words) - 1):
            p1, p2 = phonetics[words[i]][0], phonetics[words[i+1]][0]
            if p1 and p2 and p1[0] == p2[0]:  # same starting sound
                if words[i] != words[i+1]:
                    alliterative_pairs.append({
                        "word1": words[i],
                        "word2": words[i+1],
                        "initial_sound": p1[0]
                    })
                # logger.info(f"Alliteration found: {words[i]}, {words[i+1]}, Sound: {p1[0]}")  # Debugging
        logger.debug("alliterative_pairs")
        logger.info(alliterative_pairs)
        # --- Rhyme (match last stressed vowel + coda) ---
        def rhyme_part(pron):
            """Return from last stressed vowel to end (modified to handle more cases)."""
            stressed_vowels = [ph for ph in pron if re.search(r'\d', ph)]
            if stressed_vowels:
                # Return from the last stressed vowel to the end
                last_stressed_index = pron.index(stressed_vowels[-1])
                return pron[last_stressed_index:]
            else:
                # If no stressed vowel is found, try to find the last vowel
                vowels = [ph for ph in pron if re.search(r'[AEIOU]', ph, re.IGNORECASE)]
                if vowels:
                    last_vowel_index = pron.index(vowels[-1])
                    return pron[last_vowel_index:]
                else:
                    # If no vowels are found, return the last two phonemes as a last resort
                    return pron[-2:]

        rhyming_pairs = []
        for i, word1 in enumerate(words):
            for word2 in words[i+1:]:
                if word1 != word2:
                    p1, p2 = phonetics[word1][0], phonetics[word2][0]
                    if rhyme_part(p1) == rhyme_part(p2):
                        rhyming_pairs.append({
                            "word1": word1,
                            "word2": word2,
                            "rhyme_ending": " ".join(rhyme_part(p1))
                        })
                    # logger.info(f"Rhyme found: {word1}, {word2}, Ending: {' '.join(rhyme_part(p1))}")  # Debugging
        logger.debug("rhyming_pairs")
        logger.info(rhyming_pairs)
        # --- Syllable estimation (from CMU dictionary) ---
        def count_syllables(pron):
            return sum(1 for ph in pron if any(v in ph for v in vowels))

        total_syllables = sum(count_syllables(phonetics[w][0]) for w in words)

        return {
            'vowel_consonant_ratio': round(vowel_count / max(consonant_count, 1), 3),
            'total_vowels': vowel_count,
            'total_consonants': consonant_count,
            'alliteration_instances': alliterative_pairs,
            'rhyme_patterns': rhyming_pairs,
            'estimated_syllable_count': total_syllables,
            'phonetic_density': round(total_syllables / max(len(words), 1), 2)
        }
    
    
    def _analyze_morphological_components(self, words, pos_tags):
        prefixes = ['un', 're', 'pre', 'dis', 'mis', 'over', 'under', 'out', 'up', 'anti', 
                    'co', 'de', 'fore', 'inter', 'mid', 'non', 'semi', 'sub', 'super', 'trans', 'ultra']
        suffixes = ['ing', 'ed', 'er', 'est', 'ly', 'tion', 'sion', 'ness', 'ment', 
                    'ful', 'less', 'able', 'ible', 'al', 'ial', 'ic', 'ous', 'ious', 'ive']
        
        prefix_analysis = defaultdict(int)
        suffix_analysis = defaultdict(int)
        root_words = []

        # Mapping from Penn POS → WordNet POS
        def _pos_to_wordnet(treebank_tag):
            if treebank_tag.startswith('J'):
                return wordnet.ADJ
            elif treebank_tag.startswith('V'):
                return wordnet.VERB
            elif treebank_tag.startswith('N'):
                return wordnet.NOUN
            elif treebank_tag.startswith('R'):
                return wordnet.ADV
            return wordnet.NOUN  # default
        for word, pos in pos_tags:
            # Prefix analysis
            for prefix in prefixes:
                if word.lower().startswith(prefix) and len(word) > len(prefix):
                    prefix_analysis[prefix] += 1

            # Suffix analysis
            for suffix in suffixes:
                if word.lower().endswith(suffix) and len(word) > len(suffix):
                    suffix_analysis[suffix] += 1

            # POS-aware lemmatization
            wn_pos = _pos_to_wordnet(pos)
            lemma = self.lemmatizer.lemmatize(word.lower(), pos=wn_pos)

            variations = set([lemma])  # collect root + variations

            # Expand using WordNet (derivationally related forms)
            for syn in wordnet.synsets(lemma, pos=wn_pos):
                for lemma_obj in syn.lemmas():
                    for related in lemma_obj.derivationally_related_forms():
                        variations.add(related.name())
            
            if lemma != word.lower() or len(variations) > 1:
                root_words.append({
                    "original_word": word,
                    "root_word": lemma,
                    "variations": list(variations)
                })
        
        logger.debug("root_word_variations")
        logger.info(root_words)

        # POS distribution
        pos_distribution = Counter([tag for _, tag in pos_tags if "." not in tag])
        pos_distribution_list = [{'tag': tag, 'distribution': count} for tag, count in pos_distribution.items()]

        logger.debug("inflectional_variants")
        logger.info([
                item for item in root_words if item["original_word"].lower() != item["root_word"]
            ])
        
        return {
            'prefix_usage': dict(prefix_analysis),
            'suffix_usage': dict(suffix_analysis),
            'morphological_complexity': len(prefix_analysis) + len(suffix_analysis),
            'root_word_variations': root_words,
            'pos_tag_distribution': pos_distribution_list,
            'inflectional_variants': [
                item for item in root_words if item["original_word"].lower() != item["root_word"]
            ]
        }

    
    def _analyze_syntactic_structures(self, sentences: list, pos_tags: list):
        sentence_structures = []
        clause_patterns = []

        # Define a simple chunk grammar for NP, VP, PP
        grammar = r"""
            NP: {<DT|PRP\$>?<JJ.*>*<NN.*>+}          # Noun phrase
            VP: {<VB.*><NP|PP|CLAUSE>+$}             # Verb phrase
            PP: {<IN><NP>}                           # Prepositional phrase
            CLAUSE: {<NP><VP>}                       # Simple clause
        """

        chunker = RegexpParser(grammar)
        for sentence in sentences:
            sent_tokens = word_tokenize(sentence)
            sent_pos = pos_tag(sent_tokens)
            structure = [tag for _, tag in sent_pos]
            sentence_structures.append(structure)

            # Identify clause boundaries (very simplified)
            cluase_indicators = ['WDT', 'WP', 'WRB', 'IN', 'CC']
            clauses = []
            current_clause = []
            for word, tag in sent_pos:
                if "." not in tag:
                    if tag in cluase_indicators and current_clause:
                        clauses.append(current_clause)
                        current_clause = [tag]
                    else:
                        current_clause.append(tag)
            if current_clause:
                clauses.append(current_clause)
            clause_patterns.extend(clauses)
        

        # --- Extract syntactic chunks ---
        pattern_counter = Counter()
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            tagged = pos_tag(tokens)
            tree = chunker.parse(tagged)

            # Traverse chunk tree for syntactic patterns
            pattern_seq = []
            for subtree in tree:
                if isinstance(subtree, nltk.Tree):
                    pattern_seq.append(subtree.label()) # NP, VP, PP, CLAUSE
                else:
                    pattern_seq.append(subtree[1]) # fallback POS

            if pattern_seq:
                pattern_counter["_".join(pattern_seq)] += 1
        
        # --- POS trigram fallback ---
        for structure in sentence_structures:
            if "." not in structure:
                for i in range(len(structure) - 2):
                    trigram = "_".join(structure[i:i+3])
                    pattern_counter[trigram] += 1
        
        # --- Complexity metrics ---
        avg_sentence_length = sum(len(str(s).split()) for s in sentences) / max(len(sentences), 1)
        logger.debug("clause_analysis")
        logger.info(clause_patterns)


        # Modify common_syntactic_patterns to include keys and values
        most_common = pattern_counter.most_common(10)
        common_syntactic_patterns = []
        for pattern, count in most_common:
            common_syntactic_patterns.append({"pattern": pattern, "count": count})

        logger.debug("common_syntactic_patterns")
        logger.info(common_syntactic_patterns)

        return {
            'sentence_structures': sentence_structures[:10],  # First 10 for brevity
            'average_sentence_length': round(avg_sentence_length, 2),
            'clause_analysis': clause_patterns[:20],  # First 20 clauses
            'syntactic_complexity_score': round(avg_sentence_length / 10, 2),
            'common_syntactic_patterns': common_syntactic_patterns,
            'sentence_types': self._classify_sentence_types(sentences)
        }
    
    
    def _classify_sentence_types(self, sentences):
        types = {'declarative': 0, 'interrogative': 0, 'exclamatory': 0, 'imperative': 0}
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence.endswith('?'):
                types['interrogative'] += 1
            elif sentence.endswith('!'):
                types['exclamatory'] += 1
            elif sentence and sentence[0].isupper() and not any(word in sentence.lower() for word in ['please', 'you should', 'i think']):
                # Simple heuristic for imperative
                words = sentence.split()
                if len(words) > 0 and words[0].lower() in ['go', 'come', 'do', 'don', 'stop', 'start', 'take', 'give', 'make', 'let', 'help']:
                    types['imperative'] += 1
                else:
                    types['declarative'] += 1
            else:
                types['declarative'] += 1
        
        return types
    
    def _analyze_semantic_elements(self, text, words, pos_tags):
        # --- Named entity recognition ---
        entities = ne_chunk(pos_tags)
        named_entities = []
        for chunk in entities:
            if hasattr(chunk, 'label'):
                entity_text = ' '.join([token for token, pos in chunk.leaves()])
                named_entities.append({
                    "entity_text": entity_text,
                    "entity_type": chunk.label()
                })

        # --- Semantic field analysis using WordNet ---
        semantic_fields = {
            'emotion': [],
            'temporal': [],
            'spatial': [],
            'social': [],
            'cognition': [],
            'motion': [],
        }

        def get_semantic_field(word):
            synsets = wn.synsets(word)
            fields = set()
            for syn in synsets:
                # Get hypernyms (general categories)
                for hyper in syn.hypernyms():
                    name = hyper.lemma_names()[0].lower()
                    # Assign fields heuristically
                    if name in ["emotion", "feeling", "state"]:
                        fields.add("emotion")
                    elif name in ["time", "day", "period", "event"]:
                        fields.add("temporal")
                    elif name in ["location", "place", "area", "region", "position"]:
                        fields.add("spatial")
                    elif name in ["person", "group", "society", "organization"]:
                        fields.add("social")
                    elif name in ["cognition", "idea", "knowledge", "thought"]:
                        fields.add("cognition")
                    elif name in ["motion", "movement", "act", "process"]:
                        fields.add("motion")
            return fields

        for word in words:
            fields = get_semantic_field(word)
            for f in fields:
                semantic_fields[f].append(word)

        # --- Concept density analysis ---
        nouns = [word for word, pos in pos_tags if pos.startswith('NN')]
        verbs = [word for word, pos in pos_tags if pos.startswith('VB')]
        adjectives = [word for word, pos in pos_tags if pos.startswith('JJ')]

        # --- Word frequency and semantic weight ---
        word_freq = Counter(words)
        high_freq_words = dict(word_freq.most_common(20))
        high_freq_words_list = [{'word': word, 'frequency': freq} for word, freq in high_freq_words.items()]

        logger.debug("named_entities")
        logger.info(named_entities)
        logger.debug("semantic_fields")
        logger.info(semantic_fields)
        logger.debug("high_frequency_concepts")
        logger.info(high_freq_words_list)
        return {
            'named_entities': named_entities,
            'semantic_fields': {k: v for k, v in semantic_fields.items() if v},  # filter empty
            'concept_density': {
                'noun_ratio': len(nouns) / max(len(words), 1),
                'verb_ratio': len(verbs) / max(len(words), 1),
                'adjective_ratio': len(adjectives) / max(len(words), 1)
            },
            'high_frequency_concepts': high_freq_words_list,
            'semantic_richness': len(set(words)) / max(len(words), 1),
            'content_words': [word for word in words if word not in self.stop_words][:30]
        }    
    
    def _analyze_lexical_diversity(self, words, tokens):
        if not words:
            return {}
            
        total_words = len(words)
        unique_words = len(set(words))
        
        # Type-Token Ratio
        ttr = unique_words / total_words if total_words > 0 else 0
        
        # Hapax legomena (words that appear only once)
        word_freq = Counter(words)
        hapax = sum(1 for count in word_freq.values() if count == 1)
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / total_words if total_words > 0 else 0
        
        # Vocabulary sophistication
        long_words = [word for word in words if len(word) > 6]
        sophisticated_vocab_ratio = len(long_words) / total_words if total_words > 0 else 0
        
        # Lexical density (content words vs function words)
        content_words = [word for word in words if word not in self.stop_words]
        lexical_density = len(content_words) / total_words if total_words > 0 else 0

        most_frequent_words = [{'word': word, 'frequency': freq} for word, freq in dict(Counter(words).most_common(15)).items()]

        return {
            'type_token_ratio': round(ttr, 3),
            'total_words': total_words,
            'unique_words': unique_words,
            'hapax_legomena': hapax,
            'average_word_length': round(avg_word_length, 2),
            'vocabulary_sophistication': round(sophisticated_vocab_ratio, 3),
            'lexical_density': round(lexical_density, 3),
            'most_frequent_words': most_frequent_words
        }
    
    def _analyze_register(self, text, words, sentences):
        # Formality indicators
        formal_words = ['therefore', 'furthermore', 'moreover', 'consequently', 'nevertheless', 'however', 'thus', 'hence', 'regarding', 'concerning', 'utilize', 'demonstrate', 'indicate', 'establish', 'constitute']
        informal_words = ['yeah', 'okay', 'cool', 'awesome', 'stuff', 'things', 'gonna', 'wanna', 'kinda', 'sorta', 'really', 'pretty', 'super', 'totally']
        
        formal_count = sum(1 for word in words if word in formal_words)
        informal_count = sum(1 for word in words if word in informal_words)
        
        # Technical/specialized vocabulary
        technical_indicators = ['data', 'analysis', 'methodology', 'hypothesis', 'correlation', 'variable', 'parameter', 'algorithm', 'implementation', 'optimization', 'configuration', 'specification']
        technical_count = sum(1 for word in words if word in technical_indicators)
        
        # Sentence complexity as register indicator
        avg_sentence_length = sum(len(sent.split()) for sent in sentences) / max(len(sentences), 1)
        
        # Contractions (informal indicator)
        contractions = ["n't", "'ll", "'re", "'ve", "'d", "'s", "'m"]
        contraction_count = sum(text.count(contraction) for contraction in contractions)
        
        # Passive voice detection (formal indicator)
        passive_indicators = ['was', 'were', 'been', 'being']
        passive_count = sum(1 for word in words if word in passive_indicators)
        
        # Calculate register score (-1 = very informal, 1 = very formal)
        formality_score = (formal_count - informal_count + technical_count - contraction_count) / max(len(words), 1)
        
        register_classification = 'neutral'
        if formality_score > 0.05:
            register_classification = 'formal'
        elif formality_score < -0.05:
            register_classification = 'informal'
        
        return {
            'formality_score': round(formality_score, 3),
            'register_classification': register_classification,
            'formal_language_indicators': formal_count,
            'informal_language_indicators': informal_count,
            'technical_vocabulary_usage': technical_count,
            'contraction_usage': contraction_count,
            'average_sentence_complexity': round(avg_sentence_length, 2),
            'passive_voice_indicators': passive_count,
            'register_features': {
                'academic': technical_count > 0 and formal_count > informal_count,
                'conversational': informal_count > 0 or contraction_count > 0,
                'technical': technical_count > len(words) * 0.02,
                'literary': avg_sentence_length > 15 and formal_count > 0
            }
        }
    
if __name__ == "__main__":
    lang = LinguisticsTextInterpreter()
    
    news_article: dict = {
        "source": "None",
        "author": "Stephanie Mencimer",
        "title": "They Survived Katrina and Started to Rebuild. Now Trump’s Cuts May Flood Them Out Again.",
        "description": "After Hurricane Katrina and its related flooding inundated much of New Orleans in 2005, residents of a large Vietnamese community in the Algiers neighborhood were some of the first to return and rebuild. But 20 years later, their neighborhood is still prone t…",
        "url": "https://www.motherjones.com/politics/2025/08/they-survived-katrina-and-started-to-rebuild-now-trumps-cuts-may-flood-them-out-again/",
        "urlToImage": "https://www.motherjones.com/wp-content/uploads/2025/08/20250828_new-orleans-flooding_2000.png?w=1200&h=630&crop=1",
        "publishedAt": "2025-08-28T10:00:00Z",
        "content": "Mother Jones; Justin Sullivan/Getty; Aaron Schwartz/Pool/Cnp/Zuma Get your news from a source thats not owned and controlled by oligarchs. Sign up for the free Mother Jones Daily. After Hurricane K… [+6817 chars]"
    }
    lang.interpret_text(news_article)
    
