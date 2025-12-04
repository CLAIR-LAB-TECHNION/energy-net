from typing import Any, Dict, Optional


class State:
    """
    Generic container for entity state.

    Stores named state attributes in a dictionary so subclasses can
    extend behavior without modifying this base class.
    """

    def __init__(self, initial_attributes: Optional[Dict[str, Any]] = None):
        """Create a new State with optional initial attributes."""
        self._attributes: Dict[str, Any] = (
            dict(initial_attributes) if initial_attributes else {}
        )

    def set_attribute(self, name: str, value: Any) -> None:
        """Add or update a state attribute."""
        self._attributes[name] = value

    def get_attribute(self, name: str) -> Any:
        """Return the value of a state attribute, or None if missing."""
        return self._attributes.get(name)

    def get_all_attributes(self) -> Dict[str, Any]:
        """Return a shallow copy of all state attributes."""
        return self._attributes.copy()

    def remove_attribute(self, name: str) -> None:
        """Remove a state attribute if it exists."""
        self._attributes.pop(name, None)

    def clear_attributes(self) -> None:
        """Remove all state attributes."""
        self._attributes.clear()

    def __contains__(self, name: str) -> bool:
        """Return True if an attribute exists."""
        return name in self._attributes

    def __len__(self) -> int:
        """Return the number of stored attributes."""
        return len(self._attributes)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._attributes})"


class Action:
    """
    Generic container for entity actions.

    Stores action parameters or flags, mirroring the design of `State`
    for consistent handling in downstream components.
    """

    def __init__(self, initial_actions: Optional[Dict[str, Any]] = None):
        """Create an Action container with optional initial actions."""
        self._actions: Dict[str, Any] = (
            dict(initial_actions) if initial_actions else {}
        )

    def set_action(self, name: str, value: Any) -> None:
        """Add or update an action parameter."""
        self._actions[name] = value

    def get_action(self, name: str) -> Any:
        """Return an action value, or None if missing."""
        return self._actions.get(name)

    def get_all_actions(self) -> Dict[str, Any]:
        """Return a shallow copy of all actions."""
        return self._actions.copy()

    def remove_action(self, name: str) -> None:
        """Remove an action parameter if present."""
        self._actions.pop(name, None)

    def clear_actions(self) -> None:
        """Remove all action parameters."""
        self._actions.clear()

    def __contains__(self, name: str) -> bool:
        """Return True if an action exists."""
        return name in self._actions

    def __len__(self) -> int:
        """Return the number of stored actions."""
        return len(self._actions)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._actions})"
