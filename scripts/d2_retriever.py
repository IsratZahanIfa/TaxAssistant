import warnings
from pathlib import Path
from typing import List, Optional, Set
import torch
from FlagEmbedding import FlagReranker
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

warnings.filterwarnings("ignore")


class D2Retriever:

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-m3",
        cross_encoder_model: str = "BAAI/bge-reranker-v2-m3",
        llm_model: str = "gemma3:12b",  # Ollama gemma3:12b Model
    ):
        self.embedding_model = embedding_model
        self.cross_encoder_model = cross_encoder_model
        self.cross_encoder = None

        print(f"Loading HyDE LLM ({llm_model})...")
        self.llm = Ollama(model=llm_model)

        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent

        # Updated: taxdatabase folder path
        self.law_db_dir = project_root / "taxdatabase"

        if not self.law_db_dir.exists():
            raise FileNotFoundError(f"Law DB directory not found at: {self.law_db_dir}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            encode_kwargs={"normalize_embeddings": True, "device": device},
        )

        # Loading FAISS with index_name='tax_data'
        try:
            self.law_db_vector_store = FAISS.load_local(
                str(self.law_db_dir),
                self.embeddings,
                index_name="tax_data",
                allow_dangerous_deserialization=True,
            )
        except Exception as e:
            # Fallback for default index name if custom name fails
            print(f"Failed loading with index_name='tax_data', trying default: {e}")
            self.law_db_vector_store = FAISS.load_local(
                str(self.law_db_dir),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            
        print("✓ Law Vector Database loaded successfully.")

    def generate_hypothetical_document(self, query: str) -> str:
        """HyDE using Ollama gemma3:12b"""
        prompt = (
            f"Please write a passage that directly answers the following legal query. "
            f"Write it as if it were an official law text or a legal explanation.\n\n"
            f"Query: {query}\n\nPassage:"
        )
        try:
            hypothetical_doc = self.llm.invoke(prompt)
            return hypothetical_doc.strip()
        except Exception as e:
            print(f"HyDE LLM generation failed, falling back to original query: {e}")
            return query

    def _rerank_docs(self, query: str, docs: List) -> List:
        if not docs:
            return []

        if self.cross_encoder is None:
            self.cross_encoder = FlagReranker(self.cross_encoder_model)

        sentence_pairs = [[query, doc.page_content] for doc in docs]
        scores = self.cross_encoder.compute_score(sentence_pairs)

        for idx, doc in enumerate(docs):
            doc.metadata["cross_encoder_score"] = float(scores[idx])

        return sorted(
            docs,
            key=lambda d: d.metadata.get("cross_encoder_score", float("-inf")),
            reverse=True,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_hyde: bool = True,
        use_reranker: bool = True,
        top_k_after_rerank: Optional[int] = None,
    ) -> List[List[str]]:
        search_text = self.generate_hypothetical_document(query) if use_hyde else query

        results = self.law_db_vector_store.similarity_search(query=search_text, k=top_k)

        docs = [doc for doc in results]
        if use_reranker and docs:
            try:
                docs = self._rerank_docs(query, docs)
            except Exception as e:
                print(f"Reranker error, fallback to vector order: {e}")

        final_results: List[List[str]] = []
        seen_groups: Set[tuple] = set()
        output_limit = top_k_after_rerank if (use_reranker and top_k_after_rerank) else top_k

        for doc in docs:
            if len(final_results) >= output_limit:
                break

            law_numbers = doc.metadata.get("law_number", [])
            if not isinstance(law_numbers, list):
                law_numbers = [law_numbers]

            norm_group = tuple(sorted(law_numbers))
            if norm_group not in seen_groups:
                seen_groups.add(norm_group)
                final_results.append(list(law_numbers))

        return final_results[:output_limit]