import requests
import spacy
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64
from typing import List, Dict, Any

class NewsCypherGraph:
    def __init__(self, base_url: str = "http://10.42.0.1:5000"):
        self.base_url = base_url
        self.endpoint = f"{self.base_url}/news_search_by_query?query="
        # Load English NLP model (make sure to install: python -m spacy download en_core_web_sm)
        self.nlp = spacy.load("en_core_web_sm")

    def fetch_news(self, query: str) -> List[Dict[str, Any]]:
        """Fetch news articles from the API endpoint."""
        url = self.endpoint + query
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def analyze_article(self, text: str) -> Dict[str, List[str]]:
        """Extract actions, decisions, and facts from an article using NLP."""
        doc = self.nlp(text)
        actions, decisions, facts = [], [], []

        for sent in doc.sents:
            # Action: look for verbs
            verbs = [token.lemma_ for token in sent if token.pos_ == "VERB"]
            if verbs:
                actions.extend(verbs)

            # Decision: look for modal verbs (will, shall, must, may) or "decide/approve/announce"
            if any(token.lemma_ in ["decide", "approve", "announce", "agree", "resolve"] or token.tag_ == "MD" for token in sent):
                decisions.append(sent.text)

            # Fact: simple declarative statements without conditionals/questions
            if sent[-1].text not in ["?", "!"] and not any(token.dep_ == "advcl" for token in sent):
                facts.append(sent.text)

        return {
            "actions": list(set(actions)),   # unique verbs
            "decisions": decisions,
            "facts": facts
        }
    
    

    def process_query(self, query: str) -> Dict[str, List[str]]:
        """Fetch articles, analyze them, and return structured output."""
        articles = self.fetch_news(query)
        all_actions, all_decisions, all_facts = [], [], []

        for article in articles:
            combined_text = " ".join([
                article.get("title", ""),
                article.get("description", ""),
                article.get("content", "")
            ])
            result = self.analyze_article(combined_text)
            all_actions.extend(result["actions"])
            all_decisions.extend(result["decisions"])
            all_facts.extend(result["facts"])

        return {
            "actions": list(set(all_actions)),
            "decisions": list(set(all_decisions)),
            "facts": list(set(all_facts))
        }
    
    
    def _plot_intelligent_graph(self, category: str, triples: List[Dict[str, str]]) -> str:
        """Draw graph with small node labels, return base64 PNG."""
        G = nx.DiGraph()

        for t in triples:
            subj, rel, obj = t["subject"], t["relation"], t["object"]
            G.add_node(subj)
            G.add_node(obj)
            G.add_edge(subj, obj, label=rel)

        pos = nx.spring_layout(G, seed=42)

        plt.figure(figsize=(18, 6), dpi=300)
        nx.draw(
            G, pos,
            with_labels=True,
            node_size=1200,
            node_color="#FFDD99",
            font_size=8,
            font_weight="bold",
            edge_color="#333"
        )
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

        plt.title(f"{category} Graph (Cypher-style)", fontsize=12)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def _plot_graph(self, category: str, items: List[str]) -> str:
        """Create a Neo4j-style flat graph and return base64 image string."""
        G = nx.Graph()

        # Add central node
        G.add_node(category)

        # Connect each item to central node
        for item in items:
            G.add_node(item)
            G.add_edge(category, item)

        
        # Draw graph
        plt.figure(figsize=(18, 6), dpi=300) # Increased size and DPI
        pos = nx.spring_layout(G, seed=42) # stable layout
        nx.draw(
            G, pos,
            with_labels=True,
            node_size=4000, # Increased node size
            node_color="#07AD97",
            font_size=12, # Increased font size
            font_weight="bold",
            edge_color="#555"
        )
        plt.title(f"{category} Graph", fontsize=16) # Increased title font size

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)

        # Encode as base64
        return base64.b64encode(buf.read()).decode("utf-8")
    
    def analyze_sentences(self, sentences: List[str]) -> List[Dict[str, str]]:
        """
        Convert sentences into Cypher-like triples: (subject)-[:REL]->(object).
        """
        triples = []
        for sent in sentences:
            doc = self.nlp(sent)
            subj, verb, obj = None, None, None
            for token in doc:
                if "subj" in token.dep_:
                    subj = token.text
                if token.pos_ == "VERB":
                    verb = token.lemma_
                if "obj" in token.dep_:
                    obj = token.text
            if subj and verb and obj:
                triples.append({"subject": subj, "relation": verb, "object": obj})
        return triples
    
    def generate_graphs(self, query: str) -> Dict[str, List[str]]:
        """Return base64 encoded graph images for actions, decisions, and facts."""
        data = self.process_query(query)
        # data_v1 = self.process_query_v1(query)
        graphs = {}
        for category in ["actions"]:
            graphs[category] = [self._plot_graph(category, data[category])]

        for category in ["decisions", "facts"]:
            triples = self.analyze_sentences(data[category])
            graphs[category] = [self._plot_intelligent_graph(category, triples)]
        
        results = {
            "actions": {
                "content": [", ".join(data["actions"])],
                "plot": graphs["actions"][0]
            },
            "decisions": {
                "content": data["decisions"],
                "plot": graphs["decisions"][0]
            },
            "facts": {
                "content": data["facts"],
                "plot": graphs["facts"][0]
            }
        }
        return results