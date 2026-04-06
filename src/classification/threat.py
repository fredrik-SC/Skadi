"""Threat level assignment based on signal classification.

Maps classified signal types to threat levels using a configurable
keyword-based lookup table loaded from threat_levels.yaml.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Valid threat levels
THREAT_LEVELS = {"CRITICAL", "HIGH", "MEDIUM", "LOW", "INFORMATIONAL"}


class ThreatMapper:
    """Assign threat levels to classified signals.

    Uses keyword-based matching against signal name and description.
    Rules are evaluated in order; the first matching rule wins.
    Unmatched signals receive the configured default threat level.

    Args:
        config_path: Path to threat_levels.yaml. If None, uses defaults.
    """

    def __init__(self, config_path: Path | None = None) -> None:
        self._default_level = "MEDIUM"
        self._rules: list[dict[str, Any]] = []

        if config_path is not None:
            self.load(config_path)

    def load(self, config_path: Path) -> None:
        """Load threat level rules from a YAML file.

        Args:
            config_path: Path to the threat_levels.yaml file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Threat config not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self._default_level = (data or {}).get("default_threat_level", "MEDIUM")
        self._rules = (data or {}).get("rules", [])

        logger.info(
            "Loaded %d threat rules from %s (default: %s)",
            len(self._rules), config_path, self._default_level,
        )

    @property
    def default_level(self) -> str:
        """The default threat level for unmatched signals."""
        return self._default_level

    def assess(
        self,
        signal_name: str | None = None,
        description: str | None = None,
    ) -> str:
        """Assign a threat level to a classified signal.

        Searches signal name and description for keywords defined in
        the threat rules. Rules are evaluated in priority order;
        the first match wins.

        Args:
            signal_name: The Artemis signal name (e.g. "STANAG 4285").
            description: The Artemis signal description text.

        Returns:
            Threat level string: CRITICAL, HIGH, MEDIUM, LOW, or INFORMATIONAL.
        """
        # Build searchable text (lowercase for case-insensitive matching)
        text = ""
        if signal_name:
            text += signal_name.lower() + " "
        if description:
            text += description.lower()

        if not text.strip():
            return self._default_level

        for rule in self._rules:
            threat_level = rule.get("threat_level", "MEDIUM")
            keywords = rule.get("keywords", [])

            for keyword in keywords:
                if keyword.lower() in text:
                    logger.debug(
                        "Threat match: '%s' → %s (keyword: '%s')",
                        signal_name or "unknown", threat_level, keyword,
                    )
                    return threat_level

        return self._default_level
