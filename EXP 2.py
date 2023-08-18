import math

class ProbabilisticModel:
    def __init__(self, documents):
        self.documents = documents
        self.document_lengths = {doc_id: len(doc.split()) for doc_id, doc in documents.items()}
        self.avg_doc_length = sum(self.document_lengths.values()) / len(self.documents)
        self.k1 = 1.5
        self.b = 0.75

    def calculate_score(self, query, doc_id):
        score = 0
        for term in query:
            if term in self.documents[doc_id]:
                tf = self.documents[doc_id].count(term)
                doc_length = self.document_lengths[doc_id]
                idf = math.log((len(self.documents) - sum(1 for doc in self.documents.values() if term in doc) + 0.5) / (sum(1 for doc in self.documents.values() if term in doc) + 0.5) + 1)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
                score += idf * (numerator / denominator)
        return score

    def rank_documents(self, query):
        scores = {}
        for doc_id in self.documents:
            score = self.calculate_score(query, doc_id)
            scores[doc_id] = score
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked_docs

# Example documents
documents = {
    "doc1": "information retrieval techniques",
    "doc2": "probabilistic models in IR",
    "doc3": "vector space model for information retrieval",
    "doc4": "ranking algorithms in IR",
}

# Example query
query = ["information", "retrieval"]

# Create a probabilistic information retrieval model
model = ProbabilisticModel(documents)

# Rank documents based on the query
ranked_documents = model.rank_documents(query)

# Display the ranked documents
for rank, (doc_id, score) in enumerate(ranked_documents, start=1):
    print(f"Rank {rank}: Document {doc_id} (Score: {score:.4f})")
