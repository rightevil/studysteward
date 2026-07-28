from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class Summary:
    title: str
    keywords: list[str]
    summary: str

@dataclass
class Chunk:
    doc_id: int
    content: str
    chunk_index: int
    embedding_id: str
    doc_title: str = ""

class AIBackend(ABC):
    @abstractmethod
    def summarize(self, text: str) -> Summary:
        ...

    @abstractmethod
    def ask(self, question: str, context: list[Chunk]) -> str:
        ...

    @abstractmethod
    def suggest_tags(self, text: str) -> list[str]:
        ...
