"""
Circuit Breaker Pattern for Fault Tolerance

Implements the circuit breaker pattern to prevent cascading failures
and enable graceful degradation when external services are unavailable.

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Too many failures, requests are blocked (fail fast)
- HALF_OPEN: Testing if service recovered, limited requests allowed
"""

from __future__ import annotations

import time
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Callable, Any, Optional, TypeVar, Dict
from functools import wraps
from threading import RLock

from src.errors import CircuitOpenError

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ============================================================================
# Data Structures
# ============================================================================


class CircuitState(str, Enum):
    """Circuit breaker state."""
    CLOSED = "closed"          # Normal operation
    OPEN = "open"              # Blocking requests (fail fast)
    HALF_OPEN = "half_open"    # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5         # Failures before opening
    recovery_timeout_seconds: float = 60.0  # Time before trying half-open
    success_threshold: int = 2         # Successes in half-open before closing
    name: str = "circuit_breaker"


@dataclass
class CircuitBreakerMetrics:
    """Metrics for monitoring circuit breaker."""
    total_requests: int = 0
    total_failures: int = 0
    total_successes: int = 0
    consecutive_failures: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    state_changes: int = 0
    time_opened: Optional[float] = None


# ============================================================================
# Circuit Breaker Implementation
# ============================================================================


class CircuitBreaker:
    """
    Circuit breaker for handling failures in external service calls.

    Protects against cascading failures by stopping requests to a failing service
    and allowing it time to recover.
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        """Initialize circuit breaker."""
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self._lock = RLock()
        self._half_open_successes = 0
        self._state_changed_at = time.time()

    def call(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute a function through the circuit breaker.

        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitOpenError: If circuit is open
            DependencyError: If function call fails
        """
        with self._lock:
            # Check if circuit should transition states
            self._check_state_transition()

            # Reject if open
            if self.state == CircuitState.OPEN:
                raise CircuitOpenError(
                    f"Circuit breaker '{self.config.name}' is OPEN - service unavailable",
                    service=self.config.name,
                )

            # Limit concurrent requests in half-open state
            if (
                self.state == CircuitState.HALF_OPEN
                and self._half_open_successes >= self.config.success_threshold
            ):
                raise CircuitOpenError(
                    f"Circuit breaker '{self.config.name}' is testing recovery - "
                    f"max concurrent requests exceeded",
                    service=self.config.name,
                )

        # Execute function
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(e)
            raise

    def _check_state_transition(self) -> None:
        """Check if state should transition."""
        now = time.time()

        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed
            if self.metrics.time_opened is None:
                self.metrics.time_opened = now

            elapsed = now - self.metrics.time_opened
            if elapsed >= self.config.recovery_timeout_seconds:
                self._transition_to(CircuitState.HALF_OPEN)
                logger.info(
                    f"Circuit breaker '{self.config.name}' transitioning to HALF_OPEN "
                    f"after {elapsed:.1f}s timeout"
                )

    def _on_success(self) -> None:
        """Handle successful call."""
        with self._lock:
            self.metrics.total_successes += 1
            self.metrics.total_requests += 1
            self.metrics.consecutive_failures = 0
            self.metrics.last_success_time = time.time()

            if self.state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
                    logger.info(
                        f"Circuit breaker '{self.config.name}' transitioning to CLOSED "
                        f"after {self._half_open_successes} successes"
                    )

    def _on_failure(self, error: Exception) -> None:
        """Handle failed call."""
        with self._lock:
            self.metrics.total_failures += 1
            self.metrics.total_requests += 1
            self.metrics.consecutive_failures += 1
            self.metrics.last_failure_time = time.time()

            logger.warning(
                f"Circuit breaker '{self.config.name}' failure "
                f"({self.metrics.consecutive_failures}/{self.config.failure_threshold}): {error}"
            )

            # Transition to open if threshold reached
            if self.metrics.consecutive_failures >= self.config.failure_threshold:
                self._transition_to(CircuitState.OPEN)
                logger.error(
                    f"Circuit breaker '{self.config.name}' transitioning to OPEN - "
                    f"{self.metrics.consecutive_failures} consecutive failures"
                )

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to new state."""
        if self.state != new_state:
            old_state = self.state
            self.state = new_state
            self.metrics.state_changes += 1
            self.metrics.time_opened = time.time() if new_state == CircuitState.OPEN else None
            self._half_open_successes = 0

            logger.info(
                f"Circuit breaker '{self.config.name}' transitioned from "
                f"{old_state.value} to {new_state.value}"
            )

    def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self.metrics.consecutive_failures = 0
            self._half_open_successes = 0
            logger.info(f"Circuit breaker '{self.config.name}' manually reset")

    def get_state(self) -> CircuitState:
        """Get current state."""
        with self._lock:
            self._check_state_transition()
            return self.state

    def get_metrics(self) -> CircuitBreakerMetrics:
        """Get current metrics."""
        with self._lock:
            return CircuitBreakerMetrics(
                total_requests=self.metrics.total_requests,
                total_failures=self.metrics.total_failures,
                total_successes=self.metrics.total_successes,
                consecutive_failures=self.metrics.consecutive_failures,
                last_failure_time=self.metrics.last_failure_time,
                last_success_time=self.metrics.last_success_time,
                state_changes=self.metrics.state_changes,
                time_opened=self.metrics.time_opened,
            )

    def get_status(self) -> Dict[str, Any]:
        """Get human-readable status."""
        with self._lock:
            self._check_state_transition()
            metrics = self.get_metrics()

            success_rate = (
                (metrics.total_successes / metrics.total_requests * 100)
                if metrics.total_requests > 0
                else 0
            )

            return {
                "name": self.config.name,
                "state": self.state.value,
                "total_requests": metrics.total_requests,
                "total_failures": metrics.total_failures,
                "total_successes": metrics.total_successes,
                "success_rate": f"{success_rate:.1f}%",
                "consecutive_failures": metrics.consecutive_failures,
                "failure_threshold": self.config.failure_threshold,
                "last_failure": metrics.last_failure_time,
                "last_success": metrics.last_success_time,
                "state_changes": metrics.state_changes,
            }


# ============================================================================
# Decorator for Easy Integration
# ============================================================================


def circuit_breaker(
    name: str = "default",
    failure_threshold: int = 5,
    recovery_timeout_seconds: float = 60.0,
    success_threshold: int = 2,
):
    """
    Decorator to wrap a function with circuit breaker protection.

    Args:
        name: Name of the circuit breaker
        failure_threshold: Failures before opening
        recovery_timeout_seconds: Time before testing recovery
        success_threshold: Successes in half-open before closing

    Returns:
        Decorated function with circuit breaker protection

    Example:
        @circuit_breaker(name="llm_service", failure_threshold=3)
        def call_llm(prompt):
            return llm.generate(prompt)
    """
    config = CircuitBreakerConfig(
        name=name,
        failure_threshold=failure_threshold,
        recovery_timeout_seconds=recovery_timeout_seconds,
        success_threshold=success_threshold,
    )
    breaker = CircuitBreaker(config)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return breaker.call(func, *args, **kwargs)

        # Expose breaker for manual management
        wrapper._circuit_breaker = breaker  # type: ignore

        return wrapper

    return decorator


# ============================================================================
# Global Circuit Breaker Registry
# ============================================================================


class CircuitBreakerRegistry:
    """Global registry for managing multiple circuit breakers."""

    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = RLock()

    def register(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Register a new circuit breaker."""
        with self._lock:
            if name in self._breakers:
                logger.warning(f"Circuit breaker '{name}' already registered")
                return self._breakers[name]

            config = config or CircuitBreakerConfig(name=name)
            breaker = CircuitBreaker(config)
            self._breakers[name] = breaker
            logger.info(f"Registered circuit breaker '{name}'")
            return breaker

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        with self._lock:
            return self._breakers.get(name)

    def get_or_create(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create circuit breaker."""
        breaker = self.get(name)
        if breaker is None:
            breaker = self.register(name, config)
        return breaker

    def list_all(self) -> Dict[str, Dict[str, Any]]:
        """List all circuit breakers and their status."""
        with self._lock:
            return {name: breaker.get_status() for name, breaker in self._breakers.items()}

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
            logger.info("Reset all circuit breakers")


# Global registry instance
_registry = CircuitBreakerRegistry()


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """Get or create a circuit breaker from the global registry."""
    return _registry.get_or_create(name, config)


def get_all_circuit_breakers() -> Dict[str, Dict[str, Any]]:
    """Get status of all registered circuit breakers."""
    return _registry.list_all()


def reset_all_circuit_breakers() -> None:
    """Reset all circuit breakers."""
    _registry.reset_all()
