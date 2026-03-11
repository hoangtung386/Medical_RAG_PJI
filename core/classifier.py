from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from .llm_config import get_groq_llm

VALID_CATEGORIES = ("Factual", "Analytical", "Opinion", "Contextual")


class CategoryOptions(BaseModel):
    category: str = Field(
        description=(
            "The category of the query, the options are: "
            "Factual, Analytical, Opinion, or Contextual"
        ),
        example="Factual",
    )


class QueryClassifier:

    def __init__(self):
        self.llm = get_groq_llm(
            model_name="llama-3.1-8b-instant",
            temperature=0.0,
        )
        self.parser = PydanticOutputParser(
            pydantic_object=CategoryOptions,
        )

        self.prompt = PromptTemplate(
            input_variables=["query"],
            template=(
                "You are an expert intent classifier for a "
                "medical AI system.\n"
                "The query may be in any language (including "
                "Vietnamese). Classify it into exactly one of "
                "these four categories: Factual, Analytical, "
                "Opinion, or Contextual.\n\n"
                "- Factual: The query asks for specific facts "
                "or verifiable information.\n"
                "- Analytical: The query asks for an "
                "explanation, comparison, or comprehensive "
                "analysis.\n"
                "- Opinion: The query asks for subjective "
                "viewpoints, recommendations, or varying "
                "perspectives.\n"
                "- Contextual: The query includes specific "
                "user context or personal details that must "
                "be considered.\n\n"
                "{format_instructions}\n"
                "Query: {query}\n"
            ),
            partial_variables={
                "format_instructions": (
                    self.parser.get_format_instructions()
                ),
            },
        )

        self.chain = self.prompt | self.llm | self.parser

    def classify(self, query: str) -> str:
        print(f"Classifying query intent for: '{query}'")
        try:
            result = self.chain.invoke({"query": query})
            category = result.category
            if category not in VALID_CATEGORIES:
                return "Factual"
            return category
        except Exception as e:
            print(
                "Error identifying category, defaulting to "
                f"Factual. Error: {e}"
            )
            return "Factual"
