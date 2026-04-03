"""PJI Recommendation Engine.

Takes clinical snapshot data, uses RAG pipeline to retrieve relevant
medical guidelines, and generates structured PJI treatment recommendations
matching the API contract format (ai_recommendation_items + ai_rag_citations).
"""

import json
import logging
import uuid
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
3. De xuat phac do dieu tri bao gom: chan doan (DIAGNOSTIC_TEST), khang sinh toan than (SYSTEMIC_ANTIBIOTIC), khang sinh tai cho (LOCAL_ANTIBIOTIC), va phau thuat (SURGERY_PROCEDURE)
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
  "recommendation_items": [
    {{
      "category": "DIAGNOSTIC_TEST",
      "title": "Tieu de chan doan...",
      "item_json": {{
        "scoring_system": {{
          "name": "ICM 2018 PJI Diagnostic Criteria",
          "version": "2018 (Philadelphia Consensus)",
          "total_score": <so diem>,
          "interpretation": "INFECTED|INCONCLUSIVE|NOT_INFECTED",
          "confidence_note": "..."
        }},
        "major_criteria": {{
          "note": "Moi tieu chi chinh = Decision",
          "items": [
            {{
              "criterion": "...",
              "result": true|false,
              "result_detail": "...",
              "is_decisive": true|false
            }}
          ],
          "major_criteria_met": true|false,
          "major_criteria_conclusion": "..."
        }},
        "minor_criteria_scoring": {{
          "note": "Tinh diem tu cac tieu chi phu",
          "items": [
            {{
              "criterion": "...",
              "score_weight": <number>,
              "result": true|false|null,
              "result_detail": "...",
              "score_awarded": <number>
            }}
          ],
          "total_minor_score": <number>,
          "total_minor_score_note": "..."
        }},
        "ai_reasoning": {{
          "primary_diagnosis": "...",
          "infection_classification": "ACUTE|CHRONIC",
          "infection_classification_reasoning": "...",
          "identified_organism": {{
            "name": "...",
            "resistance_profile": "...",
            "resistance_detail": "...",
            "biofilm_forming": true|false,
            "virulence_note": "..."
          }},
          "reasoning_summary": "...",
          "warnings": [
            {{
              "type": "ALLERGY_ALERT|DRUG_INTERACTION|RENAL_CHECK|GLYCEMIC_ALERT",
              "severity": "HIGH|MEDIUM|LOW",
              "message": "..."
            }}
          ]
        }}
      }}
    }},
    {{
      "category": "LOCAL_ANTIBIOTIC",
      "title": "Phac do khang sinh tai cho - ...",
      "item_json": {{
        "regimen_name": "...",
        "indication": "...",
        "duration_days": <number>,
        "duration_note": "...",
        "delivery_info": {{
          "delivery_method": "CEMENT_SPACER|BEADS|OTHER",
          "spacer_type": "ARTICULATING|STATIC",
          "cement_brand_suggestion": "...",
          "mixing_ratio": "..."
        }},
        "antibiotics": [
          {{
            "antibiotic_name": "...",
            "dosage": "...",
            "frequency": "...",
            "route": "LOCAL_CEMENT",
            "sequence_order": 1,
            "role": "PRIMARY|SYNERGISTIC",
            "notes": "..."
          }}
        ],
        "monitoring": ["..."],
        "contraindications": ["..."],
        "notes": "..."
      }}
    }},
    {{
      "category": "SYSTEMIC_ANTIBIOTIC",
      "title": "Phac do khang sinh toan than - ...",
      "item_json": {{
        "regimen_name": "...",
        "indication": "...",
        "total_duration_weeks": <number>,
        "phases": [
          {{
            "phase_name": "Giai doan tan cong (Induction/IV phase)",
            "phase_order": 1,
            "duration_weeks": <number>,
            "duration_note": "...",
            "antibiotics": [
              {{
                "antibiotic_name": "...",
                "dosage": "...",
                "frequency": "...",
                "route": "IV|ORAL",
                "sequence_order": 1,
                "role": "PRIMARY|BIOFILM_AGENT|PRIMARY_ORAL",
                "notes": "..."
              }}
            ]
          }}
        ],
        "monitoring": ["..."],
        "contraindications": ["..."],
        "notes": "..."
      }}
    }},
    {{
      "category": "SURGERY_PROCEDURE",
      "title": "Phac do phau thuat - ...",
      "item_json": {{
        "surgery_strategy_type": "TWO_STAGE_REVISION|ONE_STAGE_REVISION|DAIR|OTHER",
        "strategy_rationale": "...",
        "priority_level": "HIGH|MEDIUM|LOW",
        "priority_note": "...",
        "stages": [
          {{
            "stage_order": 1,
            "stage_name": "...",
            "estimated_duration_minutes": <number>,
            "interval_from_stage1": null,
            "preconditions": []
          }}
        ],
        "estimated_total_treatment_time": "...",
        "risks_and_complications": ["..."],
        "notes": "..."
      }}
    }}
  ],
  "citations": [
    {{
      "item_category": "DIAGNOSTIC_TEST|LOCAL_ANTIBIOTIC|SYSTEMIC_ANTIBIOTIC|SURGERY_PROCEDURE",
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

Tai lieu tham khao:
{rag_context}

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
                "rag_context",
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
            snapshot_data: Normalized clinical data JSON (section_1_input format)
            options: Generation options (language, include_citations, top_k)

        Returns:
            dict with recommendation_items and citations matching API contract
        """
        options = options or {}
        run_id = f"run-{uuid.uuid4()}"

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
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                response_text = "\n".join(lines)

            parsed = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Raw response: {result.content[:500]}")
            parsed = self._build_fallback_response(result.content, docs)

        # Transform into final contract format
        return self._format_output(parsed, run_id, docs, options)

    def _format_output(
        self,
        parsed: dict,
        run_id: str,
        docs: list,
        options: dict,
    ) -> dict[str, Any]:
        """Transform LLM output into the API contract format.

        Produces:
        - recommendation_items: list of {category, title, item_json}
        - citations: list of {id, run_id, item_id, source_type, ...}
        """
        recommendation_items = []
        citations = []

        raw_items = parsed.get("recommendation_items", [])
        raw_citations = parsed.get("citations", [])

        # Build recommendation items with generated IDs
        category_to_item_id = {}
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

        # Build citations with IDs linked to items
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

            # Also add RAG document sources as citations if not enough
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

    def chat(
        self,
        question: str,
        episode_summary: Optional[dict] = None,
        recommendation_context: Optional[dict] = None,
        chat_history: Optional[list[dict]] = None,
    ) -> dict[str, Any]:
        """Chat with AI about PJI clinical decisions."""
        episode_ctx = ""
        if episode_summary:
            episode_ctx = f"Du lieu lam sang benh nhan:\n{json.dumps(episode_summary, ensure_ascii=False, indent=2)}"

        rec_ctx = ""
        if recommendation_context:
            rec_ctx = f"Ket qua de xuat AI hien tai:\n{json.dumps(recommendation_context, ensure_ascii=False, indent=2)}"

        history_str = ""
        if chat_history:
            history_parts = []
            for msg in chat_history[-10:]:
                role = "Bac si" if msg.get("role") == "user" else "AI"
                history_parts.append(f"{role}: {msg.get('content', '')}")
            history_str = "\n".join(history_parts)

        # RAG retrieval for the question
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
            "tokens_used": None,
            "references": references if references else None,
        }

    def _build_rag_query(self, snapshot_data: dict) -> str:
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

        # Add sensitivity info for antibiotic selection
        for c in cultures:
            sensitivities = c.get("sensitivities", [])
            resistant = [s["antibiotic_name"] for s in sensitivities if s.get("sensitivity_code") == "R"]
            if resistant:
                parts.append(f"resistant to: {', '.join(resistant)}")

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
                            "confidence_note": "AI response khong parse duoc JSON - can danh gia lai",
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
