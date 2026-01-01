# components/grid_entity.py

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from energy_net.common.utils import setup_logger
from energy_net.foundation.model import State, Action


class GridEntity(ABC):
    """
    Abstract Base Class for all grid entities such as Battery, ProductionUnit, ConsumptionUnit,
    and composite entities.

    This class defines the interface that all grid entities must implement, ensuring consistency
    across different components within the smart grid simulation.
    """

    def __init__(self, log_file: str):
        """
        Initializes the GridEntity with specified dynamics and sets up logging.

        Args:
            log_file (str): Path to the log file for the grid entity.
        """
        self.logger = setup_logger(self.__class__.__name__, log_file)

    @abstractmethod
    def update(self, state: State, action: Optional[Action] = None) -> None:
        """
        Updates the grid entity based on state and optional action.

        This method must be implemented by all subclasses, defining how the entity responds to a given action.

        Args:
            state (State): State object containing time and other state information.
            action (Optional[Action]): The action to perform. None if no action is taken.
        """
        pass

    def reset(self) -> None:
        """
        Resets the grid entity to its initial state.

        Subclasses can override this method to define specific reset behaviors.
        """
        self.logger.info(f"Resetting {self.__class__.__name__} to initial state.")
        # Default implementation does nothing. Subclasses should override as needed.
        pass


class ElementaryGridEntity(GridEntity):
    """
    Represents a basic grid entity such as Battery, ProductionUnit, and ConsumptionUnit.

    This class defines the interface that all grid entities must implement, ensuring consistency
    across different components within the smart grid simulation.
    """

    def __init__(self, dynamics: Any, log_file: str):
        """
        Initializes the GridEntity with specified dynamics and sets up logging.

        Args:
            dynamics (Any): The dynamics model associated with the grid entity.
            log_file (str): Path to the log file for the grid entity.
        """
        super().__init__(log_file)
        self.dynamics = dynamics
        self.logger.info(f"Initialized {self.__class__.__name__} with dynamics: {self.dynamics}")

    @abstractmethod
    def get_state(self) -> State:
        """
        Retrieves the current state of the grid entity.

        This method must be implemented by all subclasses, providing a way to access the
        entity's current state (e.g., energy level for a Battery).

        Returns:
            State: The current state of the entity.
        """
        pass

    @abstractmethod
    def perform_action(self, action: Action) -> None:
        """
        Performs an action on the grid entity.

        Args:
            action (Action): Action object specifying the operation to perform.
        """
        pass


class CompositeGridEntity(GridEntity):
    """
    Represents a composite grid entity composed of multiple sub-entities.
    Manages actions and updates across all sub-entities and aggregates their states.
    """

    def __init__(self, sub_entities: List[GridEntity], log_file: str,
                 entity_names: Optional[Dict[int, str]] = None):
        """
        Initializes the CompositeGridEntity with specified sub-entities and sets up logging.

        Args:
            sub_entities (List[GridEntity]): A list of sub-entities composing this composite entity.
            log_file (str): Path to the log file for the composite grid entity.
            entity_names (Optional[Dict[int, str]]): Optional mapping of sub-entity indices to custom names.
                If provided, uses custom names for specified entities. Indices not in this dict
                will use auto-generated names (ClassName_N). If None, all entities use auto-generated names.

        Example:
            # Use custom names for first two entities, auto-generate for the rest
            composite = CompositeGridEntity(
                [battery, solar_panel, inverter],
                "composite.log",
                entity_names={0: "main_battery", 1: "rooftop_solar"}
            )
            # Result: sub_entities = {"main_battery": battery, "rooftop_solar": solar_panel, "Inverter_0": inverter}
        """
        super().__init__(log_file)
        self.sub_entities: Dict[str, GridEntity] = {}
        self._initialize_sub_entities(sub_entities, entity_names)
        self.logger.info(f"CompositeGridEntity initialized with {len(self.sub_entities)} sub-entities.")

        # Initialize internal state
        self._state = State({'time': 0.0})

    def _initialize_sub_entities(self, sub_entities: List[GridEntity],
                                 entity_names: Optional[Dict[int, str]] = None) -> None:
        """
        Assigns unique identifiers to each sub-entity and stores them in a dictionary.

        Uses custom names from entity_names dict when provided, otherwise auto-generates
        identifiers based on class name and count.

        Args:
            sub_entities (List[GridEntity]): The list of sub-entities to be managed.
            entity_names (Optional[Dict[int, str]]): Optional mapping of indices to custom names.

        Raises:
            ValueError: If a custom name is duplicated or conflicts with an auto-generated name.
        """
        entity_names = entity_names or {}
        class_counters = {}
        used_names = set()

        # Validate custom names for duplicates
        custom_name_values = list(entity_names.values())
        if len(custom_name_values) != len(set(custom_name_values)):
            raise ValueError("Duplicate custom names detected in entity_names")

        for idx, entity in enumerate(sub_entities):
            # Check if custom name is provided for this index
            if idx in entity_names:
                identifier = entity_names[idx]
                if identifier in used_names:
                    raise ValueError(
                        f"Custom name '{identifier}' conflicts with an existing entity name"
                    )
            else:
                # Auto-generate identifier
                class_name = entity.__class__.__name__
                if class_name not in class_counters:
                    class_counters[class_name] = 0

                # Ensure auto-generated name doesn't conflict with custom names
                while True:
                    identifier = f"{class_name}_{class_counters[class_name]}"
                    class_counters[class_name] += 1
                    if identifier not in entity_names.values():
                        break

            used_names.add(identifier)
            self.sub_entities[identifier] = entity
            self.logger.debug(f"Sub-entity added with ID '{identifier}': {entity}")

    def update(self, state: State, actions: Optional[Dict[str, Action]] = None) -> None:
        """
        Updates all sub-entities based on the provided state and per-entity Action objects.

        Args:
            state (State): State object containing time and other state information.
            actions (Optional[Dict[str, Action]]): Dictionary mapping sub-entity IDs
                                                   to their respective Action objects.
        """
        # Type validation
        if not isinstance(state, State):
            raise TypeError(
                f"CompositeGridEntity.update requires a State object. "
                f"Received: {type(state)}"
            )

        if actions is not None and not isinstance(actions, dict):
            raise TypeError(
                f"CompositeGridEntity.update requires a Dict[str, Action] or None. "
                f"Received: {type(actions)}"
            )

        if actions is not None:
            for key, value in actions.items():
                if not isinstance(key, str):
                    raise TypeError("All action dictionary keys must be strings (entity IDs).")
                if not isinstance(value, Action):
                    raise TypeError(
                        f"All values in actions dict must be Action objects. "
                        f"Key '{key}' has type {type(value)}"
                    )

        # Extract time from state
        time_value = state.get_attribute('time')
        if time_value is None:
            raise ValueError("State object must contain a 'time' attribute.")

        # Update internal state
        self._state.set_attribute('time', time_value)
        self.logger.debug(
            f"Updating CompositeGridEntity at time: {time_value} with actions: {actions}"
        )

        # Update each sub-entity
        for identifier, entity in self.sub_entities.items():
            entity_action = actions.get(identifier) if actions is not None else None

            if entity_action is not None:
                self.logger.info(
                    f"Updating sub-entity '{identifier}' with Action: {entity_action}"
                )
                entity.update(state, entity_action)
            else:
                # Explicitly no action for this entity
                entity.update(state)

    def reset(self) -> None:
        """
        Resets all sub-entities to their initial states.
        """
        super().reset()
        self.logger.info("Resetting all sub-entities to their initial states.")
        for identifier, entity in self.sub_entities.items():
            self.logger.info(f"Resetting sub-entity '{identifier}'.")
            entity.reset()

        # Reset internal state
        self._state.clear_attributes()
        self._state.set_attribute('time', 0.0)

        self.logger.info("All sub-entities have been reset.")

    def get_sub_entity(self, identifier: str) -> Optional[GridEntity]:
        """
        Retrieves a sub-entity by its identifier.

        Args:
            identifier (str): The unique identifier of the sub-entity.

        Returns:
            Optional[GridEntity]: The requested sub-entity or None if not found.
        """
        return self.sub_entities.get(identifier)