import os
import logging
from typing import Dict, Any, List

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

from retriever import ClinicalRetriever
from prompts import get_prompt_for_query

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"


class ClinicalRAGChain:
    """Orchestrates the retrieval-augmented generation pipeline."""

    def __init__(self, k: int = 4):
        logger.info("Initializing ClinicalRAGChain components")
        self.retriever = ClinicalRetriever()
        self.k = k

        logger.info(f"Establishing connection to LLM: {GROQ_MODEL}")
        self.llm = ChatGroq(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0.1,
            max_tokens=1024,
        )
        self.output_parser = StrOutputParser()
        logger.info("LLM connection established")

    def _retrieve_and_format(self, question: str) -> str:
        """Fetch relative documents and inject into context window formatting."""
        chunks = self.retriever.mmr_search(question, k=self.k)
        return self.retriever.format_context_for_llm(chunks)

    def ask(self, question: str) -> Dict[str, Any]:
        """Execute a full RAG inference cycle for a given query."""
        context = self._retrieve_and_format(question)
        prompt = get_prompt_for_query(question)
        
        chain = prompt | self.llm | self.output_parser
        
        try:
            answer = chain.invoke({
                "context": context,
                "question": question
            })
            
            return {
                "question": question,
                "context": context,
                "answer": answer,
                "model": GROQ_MODEL,
                "chunks_retrieved": self.k
            }
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}", exc_info=True)
            raise

    def ask_with_history(self, question: str, chat_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Execute RAG inference utilizing recent chat history for context."""
        history_str = ""
        for msg in chat_history[-4:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content']}\n"

        enriched_question = f"Conversation History:\n{history_str}\nCurrent Query: {question}"
        return self.ask(enriched_question)
