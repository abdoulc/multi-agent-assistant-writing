from typing import List, Dict, Any
from pydantic import BaseModel, ConfigDict, Field
from pydantic import model_validator
from pydantic import field_validator


class ResearchSource(BaseModel):
    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    location: str = Field(min_length=1)
    excerpt: str = Field(min_length=1)

class AgentState(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    topic: str
    research_notes: List[str] = Field(default_factory=list)
    article: str = ""
    feedback: str = ""
    approved: bool = False
    rounds: int = 0
    sources: list[ResearchSource] = Field(default_factory=list)


    def apply(self, update: Dict[str, Any]) -> None:
        for key in update:
            if key not in type(self).model_fields:
                raise ValueError(f"Invalid state attribute: {key}")

        candidate = type(self).model_validate({**self.model_dump(), **update})
        for key in update:
            setattr(self, key, getattr(candidate, key))

class EditorDecision(BaseModel):
    approved: bool
    feedback: str = ""

    @model_validator(mode="after")
    def check_consistency(self) -> "EditorDecision":
        if self.approved and self.feedback.strip():
            raise ValueError("An approved article must not require changes.")

        if not self.approved and not self.feedback.strip():
            raise ValueError("A rejected article must include feedback.")

        return self

class SearchQuery(BaseModel):
    query: str = Field(min_length=1)

    @field_validator("query")
    @classmethod
    def reject_blank_query(cls, value: str) -> str:
        query = value.strip()
        if not query:
            raise ValueError("Search query must not be blank.")
        return query