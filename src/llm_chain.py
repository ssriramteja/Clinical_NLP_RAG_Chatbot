import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from retriever import ClinicalRetriever
from prompts import get_prompt_for_query, CLINICAL_QA_PROMPT

load_dotenv()

# ── Groq LLM Config ───────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = "llama-3.3-70b-versatile"


class ClinicalRAGChain:
    """
    Full RAG chain:
      Question → Retrieve Context → Build Prompt → Groq LLM → Answer

    Uses LangChain's LCEL (LangChain Expression Language) pipe syntax:
      chain = retriever | prompt | llm | output_parser
    """

    def __init__(self, k: int = 4):
        """
        k: number of chunks to retrieve from FAISS per query
        """
        # Step 2 retriever
        self.retriever = ClinicalRetriever()
        self.k         = k

        # Groq LLM (FREE, fast)
        print(f"⏳ Connecting to Groq ({GROQ_MODEL})...")
        self.llm = ChatGroq(
            model       = GROQ_MODEL,
            api_key     = GROQ_API_KEY,
            temperature = 0.1,     # low temp = factual, consistent answers
            max_tokens  = 1024,    # enough for detailed clinical summaries
        )

        # Output parser — converts LLM message object → plain string
        self.output_parser = StrOutputParser()
        print("✅ Groq LLM connected!\n")


    def _retrieve_and_format(self, question: str) -> str:
        """
        Retrieves top-K chunks using MMR search and
        formats them into a context string for the prompt.
        """
        chunks  = self.retriever.mmr_search(question, k=self.k)
        context = self.retriever.format_context_for_llm(chunks)
        return context


    def ask(self, question: str, verbose: bool = False) -> dict:
        """
        Main method — takes a question, runs full RAG pipeline,
        returns answer + metadata.

        Returns:
          {
            "question": str,
            "context":  str,   ← retrieved chunks (for transparency)
            "answer":   str,   ← Groq's grounded answer
            "model":    str,
            "chunks_retrieved": int
          }
        """
        # 1. Retrieve context
        context = self._retrieve_and_format(question)

        # 2. Auto-select best prompt template
        prompt = get_prompt_for_query(question)

        # 3. Build LCEL chain: prompt → llm → parse output
        chain = prompt | self.llm | self.output_parser

        if verbose:
            print(f"\n📥 Question: {question}")
            print(f"\n📄 Retrieved Context:\n{'-'*40}\n{context}\n{'-'*40}")

        # 4. Invoke chain with context + question
        answer = chain.invoke({
            "context":  context,
            "question": question
        })

        return {
            "question":         question,
            "context":          context,
            "answer":           answer,
            "model":            GROQ_MODEL,
            "chunks_retrieved": self.k
        }


    def ask_with_history(
        self,
        question: str,
        chat_history: list[dict]
    ) -> dict:
        """
        Conversational version — appends previous Q&A to the context
        so the LLM maintains conversation continuity.

        chat_history format:
          [{"role": "user",      "content": "..."},
           {"role": "assistant", "content": "..."}]
        """
        # Build history string
        history_str = ""
        for msg in chat_history[-4:]:   # last 4 messages to stay within token limit
            role    = "User" if msg["role"] == "user" else "ClinicalBot"
            history_str += f"{role}: {msg['content']}\n"

        # Append history to question for better context-aware retrieval
        enriched_question = f"{history_str}Current Question: {question}"

        return self.ask(enriched_question, verbose=False)


# ── Test / Demo ───────────────────────────────────────────────────────────────
def main():
    chain = ClinicalRAGChain(k=4)

    # ── Test 1: General Clinical Q&A ─────────────────────────────────────────
    print("=" * 60)
    print("TEST 1: General Clinical Q&A")
    print("=" * 60)
    result = chain.ask(
        "What are the symptoms of the patient with chest pain?",
        verbose=True
    )
    print(f"\n🤖 ANSWER:\n{result['answer']}")

    # ── Test 2: Risk Triage ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 2: Risk Assessment / Triage")
    print("=" * 60)
    result = chain.ask(
        "Which patients are at highest risk and need urgent attention?",
        verbose=False
    )
    print(f"\n🤖 ANSWER:\n{result['answer']}")

    # ── Test 3: Patient Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 3: Patient Summary")
    print("=" * 60)
    result = chain.ask(
        "Summarize patient P010",
        verbose=False
    )
    print(f"\n🤖 ANSWER:\n{result['answer']}")

    # ── Test 4: Treatment Plan ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 4: Treatment / Medication Query")
    print("=" * 60)
    result = chain.ask(
        "What medications were prescribed to diabetic patients?",
        verbose=False
    )
    print(f"\n🤖 ANSWER:\n{result['answer']}")

    # ── Test 5: Conversational ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 5: Conversational (with chat history)")
    print("=" * 60)
    history = [
        {"role": "user",      "content": "Tell me about stroke patients"},
        {"role": "assistant", "content": "Patient P010 presents with acute ischemic stroke..."}
    ]
    result = chain.ask_with_history(
        "What treatment was given to that patient?",
        chat_history=history
    )
    print(f"\n🤖 ANSWER:\n{result['answer']}")

    print("\n✅ Step 3 Complete! Run step 4 next: app/main.py (FastAPI)")


if __name__ == "__main__":
    main()
