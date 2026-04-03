"""Query intent classifier for adaptive retrieval."""

import logging

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from app.core.shared import SharedResources
from app.prompts import load_prompt

logger = logging.getLogger(__name__)

VALID_CATEGORIES = ("Factual", "Analytical", "Opinion", "Contextual")


class CategoryOptions(BaseModel):
    """Structured output for the classifier."""

    category: str = Field(
        description=(
            "The category of the query, the options are: "
            "Factual, Analytical, Opinion, or Contextual"
        ),
        examples=["Factual"],
    )


class QueryClassifier:
    """Classify a user query into one of four retrieval categories."""

    def __init__(self, resources: SharedResources) -> None:
        self.llm = resources.fast_llm
        self.parser = PydanticOutputParser(pydantic_object=CategoryOptions)

        self.prompt = PromptTemplate(
            input_variables=["query"],
            template=load_prompt("query_classifier"),
            partial_variables={
                "format_instructions": self.parser.get_format_instructions(),
            },
        )

        self.chain = self.prompt | self.llm | self.parser

    def classify(self, query: str) -> str:
        """Return one of ``VALID_CATEGORIES`` for *query*."""
        logger.info("Classifying query intent for: '%s'", query)
        try:
            result = self.chain.invoke({"query": query})
            category = result.category
            if category not in VALID_CATEGORIES:
                return "Factual"
            return category
        except Exception:
            logger.warning(
                "Classification failed, defaulting to Factual.",
                exc_info=True,
            )
            return "Factual"
