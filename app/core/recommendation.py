"""PJI Recommendation Engine.

Takes clinical snapshot data, uses the RAG pipeline to retrieve
relevant medical guidelines, and generates structured PJI treatment
recommendations matching the API contract format
(``ai_recommendation_items`` + ``ai_rag_citations``).
"""

import json
import logging
import uuid
from typing import Any, Optional

from langchain_core.prompts import PromptTemplate

from app.core.rag.retriever import AdaptiveRAG
from app.core.shared import SharedResources
from app.prompts import load_prompt

logger = logging.getLogger(__name__)


class PJIRecommendationEngine:
    """Generate PJI treatment recommendations using RAG."""

    def __init__(self, rag_system: AdaptiveRAG, resources: SharedResources) -> None:
        self.rag_system = rag_system
        self.llm = resources.llm
        self.model_name = "rag-llm"
        self.model_version = "v1"

        self.recommendation_prompt = PromptTemplate(
            template=load_prompt("recommendation_system"),
            input_variables=["snapshot_data", "rag_context"],
        )

        self.chat_prompt = PromptTemplate(
            template=load_prompt("chat_system"),
            input_variables=[
                "episode_context",
                "recommendation_context",
                "rag_context",
                "chat_history",
                "question",
            ],
        )

    # ------------------------------------------------------------------
    # Main recommendation
    # ------------------------------------------------------------------

    def generate_recommendation(
        self,
        snapshot_data: dict[str, Any],
        options: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Generate PJI treatment recommendation from clinical snapshot."""
        options = options or {}
        run_id = f"run-{uuid.uuid4()}"

        query = self._build_rag_query(snapshot_data)

        docs, _ = self.rag_system.retriever.get_relevant_documents(
            query=query,
            user_context=self._extract_patient_context(snapshot_data),
            use_web_search=True,
        )

        rag_context = "\n\n".join(
            f"[{doc.metadata.get('source', 'Unknown')}]: {doc.page_content}"
            for doc in docs
        )

        snapshot_str = json.dumps(snapshot_data, ensure_ascii=False, indent=2)

        chain = self.recommendation_prompt | self.llm
        result = chain.invoke({
            "snapshot_data": snapshot_str,
            "rag_context": rag_context or "Khong tim thay tai lieu tham khao cu the.",
        })

        try:
            response_text = result.content.strip()
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                lines = [ln for ln in lines if not ln.strip().startswith("```")]
                response_text = "\n".join(lines)
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            logger.error(
                "Failed to parse LLM response as JSON: %s",
                result.content[:500],
                exc_info=True,
            )
            parsed = self._build_fallback_response(result.content, docs)

        return self._format_output(parsed, run_id, docs, options)

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def chat(
        self,
        question: str,
        episode_summary: Optional[dict] = None,
        recommendation_context: Optional[dict] = None,
        chat_history: Optional[list[dict]] = None,
    ) -> dict[str, Any]:
        """Conversational Q&A about a PJI clinical case."""
        episode_ctx = ""
        if episode_summary:
            episode_ctx = (
                "Du lieu lam sang benh nhan:\n"
                + json.dumps(episode_summary, ensure_ascii=False, indent=2)
            )

        rec_ctx = ""
        if recommendation_context:
            rec_ctx = (
                "Ket qua de xuat AI hien tai:\n"
                + json.dumps(recommendation_context, ensure_ascii=False, indent=2)
            )

        history_str = ""
        if chat_history:
            parts = []
            for msg in chat_history[-10:]:
                role = "Bac si" if msg.get("role") == "user" else "AI"
                parts.append(f"{role}: {msg.get('content', '')}")
            history_str = "\n".join(parts)

        docs, _ = self.rag_system.retriever.get_relevant_documents(
            query=question,
            user_context=episode_ctx[:500] if episode_ctx else None,
            use_web_search=False,
        )

        rag_refs = ""
        if docs:
            rag_refs = "\n".join(
                f"- [{d.metadata.get('source', 'N/A')}]: {d.page_content[:200]}"
                for d in docs[:3]
            )

        chain = self.chat_prompt | self.llm
        result = chain.invoke({
            "episode_context": episode_ctx or "Khong co du lieu lam sang.",
            "recommendation_context": rec_ctx or "Khong co de xuat AI.",
            "rag_context": rag_refs or "Khong co tai lieu tham khao.",
            "chat_history": history_str or "Day la cau hoi dau tien.",
            "question": question,
        })

        references = None
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
            "tokens_used": None,
            "references": references,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_rag_query(snapshot_data: dict) -> str:
        """Build a search query from snapshot data for RAG retrieval."""
        parts = ["PJI periprosthetic joint infection diagnosis treatment"]

        clinical = snapshot_data.get("clinical_records", {})
        infection = clinical.get("infection_assessment", {})

        if infection.get("prosthesis_joint"):
            parts.append(f"joint: {infection['prosthesis_joint']}")
        if infection.get("suspected_infection_type"):
            parts.append(f"type: {infection['suspected_infection_type']}")

        cultures = snapshot_data.get("culture_results", {}).get("items", [])
        for c in cultures:
            org = c.get("organism_name") or c.get("name")
            if org:
                parts.append(f"organism: {org}")

        lab = snapshot_data.get("lab_results", {}).get("latest", {})
        inflammatory = lab.get("inflammatory_markers_blood", {})
        if inflammatory.get("alpha_defensin"):
            parts.append(f"alpha-defensin: {inflammatory['alpha_defensin']}")

        for c in cultures:
            sensitivities = c.get("sensitivities", [])
            resistant = [
                s["antibiotic_name"]
                for s in sensitivities
                if s.get("sensitivity_code") == "R"
            ]
            if resistant:
                parts.append(f"resistant to: {', '.join(resistant)}")

        return " ".join(parts)

    @staticmethod
    def _extract_patient_context(snapshot_data: dict) -> Optional[str]:
        """Extract patient context string for contextual RAG strategy."""
        parts: list[str] = []

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

    @staticmethod
    def _format_output(
        parsed: dict,
        run_id: str,
        docs: list,
        options: dict,
    ) -> dict[str, Any]:
        """Transform LLM output into the API contract format."""
        recommendation_items: list[dict] = []
        citations: list[dict] = []

        raw_items = parsed.get("recommendation_items", [])
        raw_citations = parsed.get("citations", [])

        category_to_item_id: dict[str, str] = {}
        for item in raw_items:
            item_id = f"item-{uuid.uuid4()}"
            cat = item.get("category", "UNKNOWN")
            category_to_item_id[cat] = item_id
            recommendation_items.append({
                "id": item_id,
                "category": cat,
                "title": item.get("title", ""),
                "item_json": item.get("item_json", {}),
            })

        include_citations = options.get("include_citations", True)
        if include_citations:
            for cit in raw_citations:
                cit_id = f"cit-{uuid.uuid4()}"
                item_cat = cit.get("item_category", "")
                linked_item_id = category_to_item_id.get(item_cat)
                citations.append({
                    "id": cit_id,
                    "run_id": run_id,
                    "item_id": linked_item_id,
                    "source_type": cit.get("source_type", "JOURNAL_ARTICLE"),
                    "source_title": cit.get("source_title", ""),
                    "source_uri": cit.get("source_uri"),
                    "snippet": cit.get("snippet", ""),
                    "relevance_score": cit.get("relevance_score", 0.8),
                    "cited_for": cit.get("cited_for"),
                })

            if len(citations) < 2 and docs:
                for doc in docs[:3]:
                    cit_id = f"cit-{uuid.uuid4()}"
                    citations.append({
                        "id": cit_id,
                        "run_id": run_id,
                        "item_id": None,
                        "source_type": "GUIDELINE",
                        "source_title": doc.metadata.get("source", "Unknown"),
                        "source_uri": doc.metadata.get("source", ""),
                        "snippet": doc.page_content[:300],
                        "relevance_score": 0.80,
                        "cited_for": "Tai lieu tham khao RAG",
                    })

        return {
            "run_id": run_id,
            "recommendation_items": recommendation_items,
            "citations": citations,
        }

    @staticmethod
    def _build_fallback_response(raw_text: str, docs: list) -> dict:
        """Build a minimal valid response when JSON parsing fails."""
        return {
            "recommendation_items": [
                {
                    "category": "DIAGNOSTIC_TEST",
                    "title": "Danh gia chan doan PJI",
                    "item_json": {
                        "scoring_system": {
                            "name": "ICM 2018 PJI Diagnostic Criteria",
                            "version": "2018",
                            "total_score": 0,
                            "interpretation": "INCONCLUSIVE",
                            "confidence_note": (
                                "AI response khong parse duoc JSON "
                                "— can danh gia lai"
                            ),
                        },
                        "ai_reasoning": {
                            "primary_diagnosis": "Can danh gia them",
                            "reasoning_summary": raw_text[:2000],
                            "warnings": [],
                        },
                    },
                },
            ],
            "citations": [
                {
                    "item_category": "DIAGNOSTIC_TEST",
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
