"""PJI Recommendation Engine.

Takes clinical snapshot data, uses RAG pipeline to retrieve relevant
medical guidelines, and generates structured PJI treatment recommendations.
"""

import json
import logging
from typing import Any, Optional

from langchain_core.prompts import PromptTemplate

from .engine import AdaptiveRAG
from .llm_config import get_groq_llm

logger = logging.getLogger(__name__)


RECOMMENDATION_SYSTEM_PROMPT = """Ban la mot he thong AI ho tro quyet dinh lam sang (Clinical Decision Support)
chuyen ve Nhiem trung Khop gia (Periprosthetic Joint Infection - PJI).

Nhiem vu cua ban:
1. Phan tich du lieu lam sang cua benh nhan
2. Ap dung tieu chuan ICM 2018 de chan doan PJI
3. De xuat phac do dieu tri bao gom: chan doan, khang sinh toan than, khang sinh tai cho, va phau thuat
4. Cung cap canh bao ve di ung, tuong tac thuoc, va nguy co

Du lieu lam sang benh nhan:
{snapshot_data}

Tai lieu tham khao lien quan:
{rag_context}

QUAN TRONG:
- Tra loi bang tieng Viet
- Phan tich dua tren du lieu thuc te cua benh nhan
- Ap dung dung tieu chuan ICM 2018
- Xem xet tat ca ket qua xet nghiem, nuoi cay, khang sinh do
- Canh bao ve di ung thuoc cua benh nhan
- Xem xet tuong tac thuoc voi benh nen

Hay tra ve ket qua theo CHINH XAC dinh dang JSON sau (KHONG them bat ky text nao khac ngoai JSON):
{{
  "assessment_json": {{
    "primary_diagnosis": "...",
    "infection_classification": "ACUTE|CHRONIC",
    "icm_score": <so diem>,
    "icm_interpretation": "INFECTED|INCONCLUSIVE|NOT_INFECTED",
    "identified_organism": {{
      "name": "...",
      "resistance_profile": "...",
      "biofilm_forming": true|false
    }}
  }},
  "explanation_json": {{
    "reasoning_summary": "...",
    "infection_classification_reasoning": "...",
    "treatment_rationale": "..."
  }},
  "warnings_json": [
    {{
      "type": "ALLERGY_ALERT|DRUG_INTERACTION|RENAL_CHECK|GLYCEMIC_ALERT",
      "severity": "HIGH|MEDIUM|LOW",
      "message": "..."
    }}
  ],
  "items": [
    {{
      "client_item_key": "diagnostic_test_1",
      "category": "DIAGNOSTIC_TEST",
      "title": "...",
      "priority_order": 1,
      "is_primary": true,
      "item_json": {{}}
    }},
    {{
      "client_item_key": "systemic_abx_1",
      "category": "SYSTEMIC_ANTIBIOTIC",
      "title": "...",
      "priority_order": 2,
      "is_primary": true,
      "item_json": {{}}
    }},
    {{
      "client_item_key": "local_abx_1",
      "category": "LOCAL_ANTIBIOTIC",
      "title": "...",
      "priority_order": 3,
      "is_primary": false,
      "item_json": {{}}
    }},
    {{
      "client_item_key": "surgery_1",
      "category": "SURGERY_PROCEDURE",
      "title": "...",
      "priority_order": 4,
      "is_primary": true,
      "item_json": {{}}
    }}
  ],
  "citations": [
    {{
      "client_item_key": "diagnostic_test_1",
      "source_type": "GUIDELINE|META_ANALYSIS|JOURNAL_ARTICLE|CONSENSUS_STATEMENT|SYSTEMATIC_REVIEW",
      "source_title": "...",
      "source_uri": "...",
      "snippet": "...",
      "relevance_score": 0.95,
      "cited_for": "..."
    }}
  ]
}}
"""

CHAT_SYSTEM_PROMPT = """Ban la mot tro ly AI chuyen gia ve Nhiem trung Khop gia (PJI).
Ban dang trao doi voi bac si ve mot ca benh cu the.

{episode_context}

{recommendation_context}

Lich su hoi thoai:
{chat_history}

QUAN TRONG:
- Tra loi bang tieng Viet
- Dua tren du lieu lam sang thuc te
- Trich dan tai lieu y khoa khi can
- Giai thich ro rang va chuyen nghiep

Cau hoi cua bac si: {question}

Hay tra loi bang tieng Viet, dinh dang Markdown:"""


class PJIRecommendationEngine:
    """Engine for generating PJI treatment recommendations using RAG."""

    def __init__(self, rag_system: AdaptiveRAG):
        self.rag_system = rag_system
        self.llm = get_groq_llm(
            model_name="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.1,
        )
        self.model_name = "rag-llm"
        self.model_version = "v1"

        self.recommendation_prompt = PromptTemplate(
            template=RECOMMENDATION_SYSTEM_PROMPT,
            input_variables=["snapshot_data", "rag_context"],
        )

        self.chat_prompt = PromptTemplate(
            template=CHAT_SYSTEM_PROMPT,
            input_variables=[
                "episode_context",
                "recommendation_context",
                "chat_history",
                "question",
            ],
        )

    def generate_recommendation(
        self,
        snapshot_data: dict[str, Any],
        options: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Generate PJI treatment recommendation from clinical snapshot.

        Args:
            snapshot_data: Normalized clinical data JSON
            options: Generation options (language, include_citations, top_k)

        Returns:
            dict with assessment_json, explanation_json, warnings_json,
            items, and citations
        """
        options = options or {}

        # Build query from snapshot for RAG retrieval
        query = self._build_rag_query(snapshot_data)

        # Retrieve relevant medical guidelines using RAG
        docs, category = self.rag_system.retriever.get_relevant_documents(
            query=query,
            user_context=self._extract_patient_context(snapshot_data),
            use_web_search=True,
        )

        rag_context = "\n\n".join(
            f"[{doc.metadata.get('source', 'Unknown')}]: {doc.page_content}"
            for doc in docs
        )

        # Generate recommendation using LLM
        snapshot_str = json.dumps(snapshot_data, ensure_ascii=False, indent=2)

        chain = self.recommendation_prompt | self.llm
        result = chain.invoke({
            "snapshot_data": snapshot_str,
            "rag_context": rag_context if rag_context else "Khong tim thay tai lieu tham khao cu the.",
        })

        # Parse LLM response as JSON
        try:
            response_text = result.content.strip()
            # Remove markdown code block markers if present
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                # Remove first and last lines (``` markers)
                lines = [l for l in lines if not l.strip().startswith("```")]
                response_text = "\n".join(lines)

            parsed = json.loads(response_text)
            return parsed
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Raw response: {result.content[:500]}")
            # Return a minimal valid structure
            return self._build_fallback_response(result.content, docs)

    def chat(
        self,
        question: str,
        episode_summary: Optional[dict] = None,
        recommendation_context: Optional[dict] = None,
        chat_history: Optional[list[dict]] = None,
    ) -> dict[str, Any]:
        """Chat with AI about PJI clinical decisions.

        Args:
            question: Doctor's question
            episode_summary: Snapshot data for context
            recommendation_context: Current recommendation items
            chat_history: Previous messages in the conversation

        Returns:
            dict with answer, tokens_used, references
        """
        # Build context strings
        episode_ctx = ""
        if episode_summary:
            episode_ctx = f"Du lieu lam sang benh nhan:\n{json.dumps(episode_summary, ensure_ascii=False, indent=2)}"

        rec_ctx = ""
        if recommendation_context:
            rec_ctx = f"Ket qua de xuat AI hien tai:\n{json.dumps(recommendation_context, ensure_ascii=False, indent=2)}"

        history_str = ""
        if chat_history:
            history_parts = []
            for msg in chat_history[-10:]:  # Last 10 messages
                role = "Bac si" if msg.get("role") == "user" else "AI"
                history_parts.append(f"{role}: {msg.get('content', '')}")
            history_str = "\n".join(history_parts)

        # Also do RAG retrieval for the question
        docs, _ = self.rag_system.retriever.get_relevant_documents(
            query=question,
            user_context=episode_ctx[:500] if episode_ctx else None,
            use_web_search=False,
        )

        # Append RAG results to episode context
        if docs:
            rag_refs = "\n\nTai lieu tham khao:\n" + "\n".join(
                f"- [{d.metadata.get('source', 'N/A')}]: {d.page_content[:200]}"
                for d in docs[:3]
            )
            episode_ctx += rag_refs

        chain = self.chat_prompt | self.llm
        result = chain.invoke({
            "episode_context": episode_ctx or "Khong co du lieu lam sang.",
            "recommendation_context": rec_ctx or "Khong co de xuat AI.",
            "chat_history": history_str or "Day la cau hoi dau tien.",
            "question": question,
        })

        references = []
        if docs:
            references = [
                {
                    "source": d.metadata.get("source", "Unknown"),
                    "type": d.metadata.get("type", "document"),
                    "snippet": d.page_content[:300],
                }
                for d in docs[:5]
            ]

        return {
            "answer": result.content,
            "tokens_used": None,  # Groq doesn't easily expose this
            "references": references if references else None,
        }

    def _build_rag_query(self, snapshot_data: dict) -> str:
        """Build a search query from snapshot data for RAG retrieval."""
        parts = ["PJI periprosthetic joint infection diagnosis treatment"]

        # Extract key clinical info
        clinical = snapshot_data.get("clinical_records", {})
        infection = clinical.get("infection_assessment", {})

        if infection.get("prosthesis_joint"):
            parts.append(f"joint: {infection['prosthesis_joint']}")
        if infection.get("suspected_infection_type"):
            parts.append(f"type: {infection['suspected_infection_type']}")

        # Culture results
        cultures = snapshot_data.get("culture_results", {}).get("items", [])
        for c in cultures:
            if c.get("organism_name"):
                parts.append(f"organism: {c['organism_name']}")

        # Lab markers
        lab = snapshot_data.get("lab_results", {}).get("latest", {})
        inflammatory = lab.get("inflammatory_markers_blood", {})
        if inflammatory.get("alpha_defensin"):
            parts.append(f"alpha-defensin: {inflammatory['alpha_defensin']}")

        return " ".join(parts)

    def _extract_patient_context(self, snapshot_data: dict) -> str:
        """Extract patient context string for contextual RAG strategy."""
        parts = []

        demographics = snapshot_data.get("patient_demographics", {})
        if demographics.get("gender"):
            parts.append(f"Gender: {demographics['gender']}")

        med_history = snapshot_data.get("medical_history", {})
        if med_history.get("medical_history"):
            parts.append(f"History: {med_history['medical_history'][:200]}")

        allergies = med_history.get("allergies", {})
        if allergies.get("is_allergy"):
            parts.append(f"Allergy: {allergies.get('allergy_note', 'Yes')}")

        return "; ".join(parts) if parts else None

    def _build_fallback_response(self, raw_text: str, docs: list) -> dict:
        """Build a minimal valid response when JSON parsing fails."""
        return {
            "assessment_json": {
                "primary_diagnosis": "PJI - Can danh gia them",
                "note": "AI response could not be parsed as structured JSON",
                "raw_text": raw_text[:1000],
            },
            "explanation_json": {
                "reasoning_summary": raw_text[:500],
            },
            "warnings_json": [],
            "items": [
                {
                    "client_item_key": "diagnostic_test_1",
                    "category": "DIAGNOSTIC_TEST",
                    "title": "Danh gia chan doan PJI",
                    "priority_order": 1,
                    "is_primary": True,
                    "item_json": {"raw_analysis": raw_text[:2000]},
                }
            ],
            "citations": [
                {
                    "client_item_key": "diagnostic_test_1",
                    "source_type": "GUIDELINE",
                    "source_title": d.metadata.get("source", "Unknown"),
                    "source_uri": d.metadata.get("source", ""),
                    "snippet": d.page_content[:300],
                    "relevance_score": 0.8,
                    "cited_for": "Tai lieu tham khao RAG",
                }
                for d in docs[:3]
            ],
        }
