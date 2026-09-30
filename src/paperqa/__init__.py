"""Focused scientific research agent."""

from paperqa.research.config import AgentConfig, EvidencePolicy
from paperqa.research.service import ResearchService

__all__ = ["AgentConfig", "EvidencePolicy", "ResearchService"]

try:
    from paperqa.version import __version__
except ImportError:  # source archives without generated package metadata
    __version__ = "0+unknown"
