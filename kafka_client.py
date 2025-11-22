# app/kafka_client.py

import os
from typing import Any, Dict, Optional
import json

try:
    from kafka import KafkaProducer  # type: ignore
except ImportError:
    KafkaProducer = None  # type: ignore

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "")
KAFKA_ENABLED = bool(KAFKA_BROKER and KafkaProducer is not None)

# Use Any here to avoid Pylance complaining about variable-as-type
_producer: Optional[Any] = None


def get_producer() -> Optional[Any]:
    global _producer
    if not KAFKA_ENABLED:
        return None
    if _producer is None:
        try:
            _producer = KafkaProducer(
                bootstrap_servers=[KAFKA_BROKER],
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            print("[Kafka] Producer initialized")
        except Exception as e:
            print("[Kafka] Init failed:", e)
            return None
    return _producer


def publish(topic: str, payload: Dict[str, Any]) -> None:
    producer = get_producer()
    if not producer:
        return
    try:
        producer.send(topic, value=payload)
    except Exception as e:
        print(f"[Kafka] Publish error ({topic}):", e)