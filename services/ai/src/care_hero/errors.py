"""Local pipeline exceptions with concise messages."""


class CareHeroError(Exception):
    """Base error for expected local pipeline failures."""


class ConfigurationError(CareHeroError):
    """Configuration is missing or invalid."""


class MediaReadError(CareHeroError):
    """An image or video cannot be opened or decoded."""


class UnsupportedMediaError(CareHeroError):
    """The supplied extension is not allow-listed."""


class DependencyError(CareHeroError):
    """A required local dependency cannot be initialized."""


class FaceMeshProcessingError(CareHeroError):
    """A face mesh frame could not be processed."""


class ScoringError(CareHeroError):
    """Scenario rules are invalid or cannot be evaluated."""


class LLMConfigurationError(CareHeroError):
    """LLM credentials or configuration are missing."""


class LLMResponseError(CareHeroError):
    """The LLM call failed or did not satisfy the output contract."""


class SpeechTranscriptionError(CareHeroError):
    """Local speech transcription could not be initialized or completed."""
