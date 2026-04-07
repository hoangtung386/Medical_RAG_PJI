"""RabbitMQ worker — consumes recommendation requests, publishes results.

Bridges the Spring Boot backend with the RAG recommendation engine
via async message processing over RabbitMQ.

Queue contract (aligned with Spring Boot ``RabbitMQConfig``):
  - Consumes from: ``pji.ai.recommendation.queue``
    (bound to exchange ``pji.ai.exchange`` with routing keys
     ``ai.recommendation.generate`` and ``ai.recommendation.refresh``)
  - Publishes to:  ``pji.ai.exchange`` / ``ai.recommendation.result``
    (routed to ``pji.ai.recommendation.result.queue``)
"""

import asyncio
import json
import logging
import time
from typing import Any

import aio_pika
from aio_pika.abc import AbstractIncomingMessage

from app.config import settings
from app.core.completeness import check_data_completeness
from app.core.recommendation import PJIRecommendationEngine

logger = logging.getLogger(__name__)


class RecommendationWorker:
    """Async RabbitMQ consumer for AI recommendation processing."""

    def __init__(self, engine: PJIRecommendationEngine) -> None:
        self.engine = engine
        self._connection: aio_pika.abc.AbstractRobustConnection | None = None
        self._channel: aio_pika.abc.AbstractChannel | None = None
        self._consumer_tag: str | None = None

    async def start(self) -> None:
        """Connect to RabbitMQ and start consuming."""
        url = settings.rabbitmq_url
        if not url:
            logger.warning(
                "RABBITMQ_URL not set — RabbitMQ worker disabled. "
                "Recommendations will only work via HTTP."
            )
            return

        try:
            self._connection = await aio_pika.connect_robust(url)
            self._channel = await self._connection.channel()
            await self._channel.set_qos(
                prefetch_count=settings.rabbitmq_prefetch_count,
            )

            queue = await self._channel.declare_queue(
                settings.rabbitmq_recommendation_queue,
                durable=True,
                passive=True,
            )

            self._consumer_tag = await queue.consume(self._on_message)
            logger.info(
                "RabbitMQ worker started — consuming from '%s'",
                settings.rabbitmq_recommendation_queue,
            )
        except Exception:
            logger.error("Failed to start RabbitMQ worker", exc_info=True)
            await self.stop()

    async def stop(self) -> None:
        """Gracefully shut down the worker."""
        if self._channel and not self._channel.is_closed:
            await self._channel.close()
        if self._connection and not self._connection.is_closed:
            await self._connection.close()
        logger.info("RabbitMQ worker stopped.")

    # ------------------------------------------------------------------
    # Message handler
    # ------------------------------------------------------------------

    async def _on_message(self, message: AbstractIncomingMessage) -> None:
        """Process a single recommendation request message."""
        async with message.process(requeue=False):
            body: dict[str, Any] = {}
            try:
                body = json.loads(message.body.decode())
                request_id = body.get("requestId", "unknown")
                run_id = body.get("runId")
                logger.info(
                    "Processing recommendation: requestId=%s, runId=%s",
                    request_id,
                    run_id,
                )

                result = await self._process(body)
                await self._publish_result(result)

                logger.info(
                    "Published result: requestId=%s, status=%s",
                    result.get("request_id"),
                    result.get("status"),
                )
            except Exception as exc:
                logger.error(
                    "Failed to process message: %s", exc, exc_info=True,
                )
                error_result = self._build_error_result(body, str(exc))
                try:
                    await self._publish_result(error_result)
                except Exception:
                    logger.error(
                        "Failed to publish error result", exc_info=True,
                    )

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    async def _process(self, body: dict[str, Any]) -> dict[str, Any]:
        """Run the recommendation pipeline and build a result message."""
        request_id = body.get("requestId", "")
        run_id = body.get("runId")
        snapshot_data = body.get("snapshotDataJson", {})
        options = body.get("options", {
            "language": "vi",
            "include_citations": True,
            "top_k": 5,
        })

        start = time.time()

        # Completeness check (deterministic, fast)
        completeness = check_data_completeness(snapshot_data)

        # RAG recommendation (CPU/IO-bound — run in thread)
        result = await asyncio.to_thread(
            self.engine.generate_recommendation,
            snapshot_data=snapshot_data,
            options=options,
        )

        latency_ms = int((time.time() - start) * 1000)

        items = self._map_items(result.get("recommendation_items", []))
        citations = self._map_citations(result.get("citations", []))

        return {
            "request_id": request_id,
            "run_id": run_id,
            "status": "SUCCESS",
            "model": {
                "name": self.engine.model_name,
                "version": self.engine.model_version,
            },
            "latency_ms": latency_ms,
            "assessment_json": result.get("assessment_json"),
            "explanation_json": result.get("explanation_json"),
            "warnings_json": result.get("warnings_json"),
            "items": items,
            "citations": citations,
            "data_completeness": {
                "is_complete": completeness.get("is_complete", False),
                "missing_items": completeness.get("missing_items", []),
                "completeness_score": completeness.get(
                    "completeness_score", "0%",
                ),
                "impact_note": completeness.get("impact_note", ""),
            },
            "error_message": None,
        }

    # ------------------------------------------------------------------
    # Mapping helpers — align RAG output to Spring Boot contract
    # ------------------------------------------------------------------

    @staticmethod
    def _map_items(
        raw_items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Map recommendation items to the Spring Boot DTO format.

        Spring Boot ``RabbitMQRecommendationResultMessage.ItemDTO`` expects:
        client_item_key, category, title, priority_order, is_primary, item_json
        """
        mapped: list[dict[str, Any]] = []
        for idx, item in enumerate(raw_items):
            mapped.append({
                "client_item_key": item.get("id", f"item-{idx}"),
                "category": item.get("category", "DIAGNOSTIC_TEST"),
                "title": item.get("title", ""),
                "priority_order": idx + 1,
                "is_primary": idx == 0,
                "item_json": item.get("item_json", {}),
            })
        return mapped

    @staticmethod
    def _map_citations(
        raw_citations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Map citations to the Spring Boot DTO format.

        Spring Boot ``RabbitMQRecommendationResultMessage.CitationDTO`` expects:
        client_item_key, source_type, source_title, source_uri,
        snippet, relevance_score, cited_for
        """
        mapped: list[dict[str, Any]] = []
        for cit in raw_citations:
            mapped.append({
                "client_item_key": cit.get("item_id"),
                "source_type": cit.get("source_type", "GUIDELINE"),
                "source_title": cit.get("source_title", ""),
                "source_uri": cit.get("source_uri"),
                "snippet": cit.get("snippet", ""),
                "relevance_score": cit.get("relevance_score", 0.8),
                "cited_for": cit.get("cited_for"),
            })
        return mapped

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    async def _publish_result(self, result: dict[str, Any]) -> None:
        """Publish a result message to the result exchange/routing key."""
        if not self._channel or self._channel.is_closed:
            logger.error("Cannot publish — channel is closed")
            return

        exchange = await self._channel.get_exchange(
            settings.rabbitmq_exchange,
        )

        await exchange.publish(
            aio_pika.Message(
                body=json.dumps(result, ensure_ascii=False).encode(),
                content_type="application/json",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            ),
            routing_key=settings.rabbitmq_recommendation_result_routing_key,
        )

    @staticmethod
    def _build_error_result(
        body: dict[str, Any],
        error_message: str,
    ) -> dict[str, Any]:
        """Build a FAILED result message for error reporting."""
        return {
            "request_id": body.get("requestId", ""),
            "run_id": body.get("runId"),
            "status": "FAILED",
            "model": None,
            "latency_ms": 0,
            "assessment_json": None,
            "explanation_json": None,
            "warnings_json": None,
            "items": [],
            "citations": [],
            "data_completeness": None,
            "error_message": error_message[:2000],
        }
