"""Topics API – CRUD for knowledge domains."""

import re
import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from backend.database.db import (
    get_all_topics, get_topic, create_topic,
    update_topic, delete_topic,
)
from backend.rag.vector_store import collection_info

router = APIRouter(prefix="/api/topics", tags=["topics"])


class TopicCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    description: str = ""
    color: str = "#7c3aed"


class TopicUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    color: Optional[str] = None


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", name.lower().strip())[:40]


@router.get("")
def list_topics():
    topics = get_all_topics()
    for t in topics:
        info = collection_info(t["id"])
        t.update(info)
    return topics


@router.post("", status_code=201)
def create(body: TopicCreate):
    topic_id = _slug(body.name) or str(uuid.uuid4())[:8]
    if get_topic(topic_id):
        topic_id = f"{topic_id}_{uuid.uuid4().hex[:6]}"
    return create_topic(topic_id, body.name, body.description, body.color)


@router.get("/{topic_id}")
def get(topic_id: str):
    t = get_topic(topic_id)
    if not t:
        raise HTTPException(404, "Topic not found")
    info = collection_info(topic_id)
    t.update(info)
    return t


@router.patch("/{topic_id}")
def update(topic_id: str, body: TopicUpdate):
    if not get_topic(topic_id):
        raise HTTPException(404, "Topic not found")
    updates = {k: v for k, v in body.dict().items() if v is not None}
    return update_topic(topic_id, **updates)


@router.delete("/{topic_id}", status_code=204)
def remove(topic_id: str):
    if topic_id == "general":
        raise HTTPException(400, "Cannot delete the default topic")
    if not get_topic(topic_id):
        raise HTTPException(404, "Topic not found")
    delete_topic(topic_id)
