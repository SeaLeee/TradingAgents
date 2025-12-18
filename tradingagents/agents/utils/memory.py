import chromadb
from chromadb.config import Settings
from openai import OpenAI
import os
import hashlib
import numpy as np


# Global cache for sentence transformer model (load once)
_sentence_transformer_model = None


def get_sentence_transformer():
    """Lazy load sentence transformer model."""
    global _sentence_transformer_model
    if _sentence_transformer_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Use a lightweight but effective model
            _sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            return None
    return _sentence_transformer_model


class FinancialSituationMemory:
    def __init__(self, name, config):
        self.config = config
        self.llm_provider = config.get("llm_provider", "openai").lower()
        
        if self.llm_provider == "google":
            # Use Google's embedding
            self.embedding = "models/text-embedding-004"
            self.client = None  # Will use google-genai directly
        elif self.llm_provider == "openrouter":
            # OpenRouter - use local sentence-transformers
            self.embedding = None
            self.client = None
            self._use_local_embedding = True
        elif self.llm_provider == "deepseek":
            # DeepSeek - use local sentence-transformers
            self.embedding = None
            self.client = None
            self._use_local_embedding = True
        elif config["backend_url"] == "http://localhost:11434/v1":
            self.embedding = "nomic-embed-text"
            self.client = OpenAI(base_url=config["backend_url"])
        else:
            self.embedding = "text-embedding-3-small"
            self.client = OpenAI(base_url=config["backend_url"])
        
        self.chroma_client = chromadb.Client(Settings(allow_reset=True))
        # Use get_or_create_collection to avoid "already exists" error
        self.situation_collection = self.chroma_client.get_or_create_collection(name=name)

    def _hash_embedding(self, text, dim=384):
        """Generate a simple hash-based embedding for text.
        This is a fallback when no embedding API is available.
        Uses SHA256 hash expanded to create a pseudo-embedding vector.
        """
        # Create a deterministic embedding from text hash
        text_bytes = text.encode('utf-8')
        hash_bytes = hashlib.sha256(text_bytes).digest()
        
        # Expand hash to desired dimension
        np.random.seed(int.from_bytes(hash_bytes[:4], 'big'))
        embedding = np.random.randn(dim).astype(np.float32)
        
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()

    def _local_embedding(self, text):
        """Use local sentence-transformers for embedding."""
        model = get_sentence_transformer()
        if model is not None:
            embedding = model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        else:
            # Fallback to hash-based if sentence-transformers not installed
            return self._hash_embedding(text)

    def get_embedding(self, text):
        """Get embedding for a text based on the LLM provider"""
        
        # Use local embedding for providers without embedding API
        if getattr(self, '_use_local_embedding', False):
            return self._local_embedding(text)
        
        if self.llm_provider == "google" or self.client is None:
            # Use Google's embedding API
            try:
                from google import genai
                
                api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set")
                
                client = genai.Client(api_key=api_key)
                response = client.models.embed_content(
                    model=self.embedding,
                    contents=text
                )
                return response.embeddings[0].values
            except ImportError:
                raise ImportError("google-genai package is required for Google embeddings. Install with: pip install google-genai")
        else:
            # Use OpenAI-compatible embedding
            response = self.client.embeddings.create(
                model=self.embedding, input=text
            )
            return response.data[0].embedding

    def add_situations(self, situations_and_advice):
        """Add financial situations and their corresponding advice. Parameter is a list of tuples (situation, rec)"""

        situations = []
        advice = []
        ids = []
        embeddings = []

        offset = self.situation_collection.count()

        for i, (situation, recommendation) in enumerate(situations_and_advice):
            situations.append(situation)
            advice.append(recommendation)
            ids.append(str(offset + i))
            embeddings.append(self.get_embedding(situation))

        self.situation_collection.add(
            documents=situations,
            metadatas=[{"recommendation": rec} for rec in advice],
            embeddings=embeddings,
            ids=ids,
        )

    def get_memories(self, current_situation, n_matches=1):
        """Find matching recommendations using OpenAI embeddings"""
        query_embedding = self.get_embedding(current_situation)

        results = self.situation_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_matches,
            include=["metadatas", "documents", "distances"],
        )

        matched_results = []
        for i in range(len(results["documents"][0])):
            matched_results.append(
                {
                    "matched_situation": results["documents"][0][i],
                    "recommendation": results["metadatas"][0][i]["recommendation"],
                    "similarity_score": 1 - results["distances"][0][i],
                }
            )

        return matched_results


if __name__ == "__main__":
    # Example usage
    matcher = FinancialSituationMemory()

    # Example data
    example_data = [
        (
            "High inflation rate with rising interest rates and declining consumer spending",
            "Consider defensive sectors like consumer staples and utilities. Review fixed-income portfolio duration.",
        ),
        (
            "Tech sector showing high volatility with increasing institutional selling pressure",
            "Reduce exposure to high-growth tech stocks. Look for value opportunities in established tech companies with strong cash flows.",
        ),
        (
            "Strong dollar affecting emerging markets with increasing forex volatility",
            "Hedge currency exposure in international positions. Consider reducing allocation to emerging market debt.",
        ),
        (
            "Market showing signs of sector rotation with rising yields",
            "Rebalance portfolio to maintain target allocations. Consider increasing exposure to sectors benefiting from higher rates.",
        ),
    ]

    # Add the example situations and recommendations
    matcher.add_situations(example_data)

    # Example query
    current_situation = """
    Market showing increased volatility in tech sector, with institutional investors 
    reducing positions and rising interest rates affecting growth stock valuations
    """

    try:
        recommendations = matcher.get_memories(current_situation, n_matches=2)

        for i, rec in enumerate(recommendations, 1):
            print(f"\nMatch {i}:")
            print(f"Similarity Score: {rec['similarity_score']:.2f}")
            print(f"Matched Situation: {rec['matched_situation']}")
            print(f"Recommendation: {rec['recommendation']}")

    except Exception as e:
        print(f"Error during recommendation: {str(e)}")
