from dataclasses import dataclass
from typing import Literal


RiskLevel = Literal["low", "moderate", "high"]


@dataclass
class UserProfile:
    """
    Simple user profile capturing a name and risk preference.

    risk_level:
        - "low":      conservative / risk-averse
        - "moderate": balanced
        - "high":     aggressive / risk-seeking
    """
    name: str
    risk_level: RiskLevel   # 'low', 'moderate', 'high'

    @classmethod
    def from_type(cls, user_type: str) -> "UserProfile":
        """
        Factory for convenient construction from a simple string.

        Examples:
            UserProfile.from_type("conservative") -> ("Conservative", "low")
            UserProfile.from_type("moderate")     -> ("Moderate", "moderate")
            UserProfile.from_type("aggressive")   -> ("Aggressive", "high")
        """
        t = (user_type or "").strip().lower()

        if t in ("conservative", "low", "risk_low", "risk-averse", "risk_averse"):
            return cls(name="Conservative", risk_level="low")

        if t in ("aggressive", "high", "risk_high", "risk-seeking", "risk_seeking"):
            return cls(name="Aggressive", risk_level="high")

        # default: moderate
        return cls(name="Moderate", risk_level="moderate")


# Optional handy constants (keeps your old style)
CONSERVATIVE = UserProfile("Conservative", "low")
MODERATE = UserProfile("Moderate", "moderate")
AGGRESSIVE = UserProfile("Aggressive", "high")
