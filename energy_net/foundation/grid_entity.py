# components/grid_entity.py

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from original.utils.logger import setup_logger
from energy_net.foundation.model import State, Action


class GridEntity(ABC):
    """
    Abstract Base Class for all grid entities such as Battery, ProductionUnit, ConsumptionUnit,
    and composite entities.

    This class defines the interface that all grid entities must implement, ensuring consistency
    across different components within the smart grid simulation.

    Supports both legacy (float for time) and new (State) interfaces.
    """

    def __init__(self, log_file: str):
        """
        Initializes the GridEntity with specified dynamics and sets up logging.

        Args:
            log_file (str): Path to the log file for the grid entity.
        """
        self.logger = setup_logger(self.__class__.__name__, log_file)

    @abstractmethod
    def update(self, state: Union[float, State], action: Union[float, Action, Dict[str, Any], None] = None) -> None:
        """
        Updates the grid entity based on state and optional action.

        Supports both legacy and new interfaces:
        - Legacy: update(0.5, action=5.0)  # float is interpreted as time
        - New: update(state_obj, action_obj)

        This method must be implemented by all subclasses, defining how the entity responds to a given action.

        Args:
            state: State object containing time and other state information OR
                   float (legacy) representing time as fraction of day (0 to 1).
            action: The action to perform. Can be:
                   - float (legacy): single action value
                   - Dict (legacy): actions for composite entities
                   - Action object (new): action with named parameters
                   - None: no action
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

    Supports both legacy (float) and new (State/Action) interfaces.
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
    def get_state(self) -> Union[float, State]:
        """
        Retrieves the current state of the grid entity.

        This method must be implemented by all subclasses, providing a way to access the
        entity's current state (e.g., energy level for a Battery).

        Returns:
            float or State: The current state of the entity.
                           Legacy implementations return float.
                           New implementations can return State objects.
        """
        pass

    def get_full_state(self) -> State:
        """
        Retrieves the full state as a State object.

        Subclasses should override this to return their complete state.
        Default implementation creates a basic State from get_state().

        Returns:
            State: Complete state information.
        """
        state_value = self.get_state()
        if isinstance(state_value, State):
            return state_value
        else:
            # Wrap float in a State object
            return State({'value': state_value})

    @abstractmethod
    def perform_action(self, action: Union[float, Action]) -> None:
        """
        Performs an action on the grid entity.

        Args:
            action: Either a float (legacy) or Action object (new interface).
        """
        pass


class CompositeGridEntity(GridEntity):
    """
    Represents a composite grid entity composed of multiple sub-entities.
    Manages actions and updates across all sub-entities and aggregates their states.

    Supports both legacy (dict) and new (State/Action) interfaces.
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
    def update(self, state: Union[float, State], actions: Optional[Union[Dict[str, Any], Action]] = None) -> None:
        """
        Updates all sub-entities based on the provided state and actions.

        Supports both legacy and new interfaces:
        - Legacy: update(0.5, actions={'Battery_0': 5.0})  # float is time
        - New: update(state_obj, action_obj)

        Args:
            state: State object containing time and other state information OR
                   float (legacy) representing time as fraction of day (0 to 1).
            actions: Can be:
                    - Dict[str, float] (legacy): Maps sub-entity IDs to action values
                    - Action object (new): Contains actions for sub-entities
                    - None: No actions
        """
        # Handle both legacy (float) and new (State) interfaces
        if isinstance(state, State):
            time_value = state.get_attribute('time')
            if time_value is None:
                self.logger.warning("State object missing 'time' attribute, using 0.0")
                time_value = 0.0
            state_obj = state
        else:
            # Legacy: state parameter is actually just time as a float
            time_value = state
            state_obj = State({'time': time_value})

        # Update internal state
        self._state.set_attribute('time', time_value)

        self.logger.debug(f"Updating CompositeGridEntity at time: {time_value} with actions: {actions}")

        # Handle both legacy (dict) and new (Action) interfaces for actions
        if isinstance(actions, Action):
            # New interface: extract actions from Action object
            # Actions should be stored with sub-entity identifiers as keys
            actions_dict = {}
            for identifier in self.sub_entities.keys():
                action_value = actions.get_action(identifier)
                if action_value is not None:
                    actions_dict[identifier] = action_value
        elif isinstance(actions, dict):
            # Legacy interface: actions is already a dict
            actions_dict = actions
        else:
            # No actions provided
            actions_dict = {}

        # Update each sub-entity
        for identifier, entity in self.sub_entities.items():
            # EXAMPLE IDENTIFIER: "Battery_0", "ProductionUnit_1", etc.
            action = actions_dict.get(identifier) if actions_dict else None
            if action is not None:
                self.logger.info(f"Updating sub-entity '{identifier}' with action: {action}")
                entity.update(state_obj, action)
            else:
                entity.update(state_obj)

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieves the current states of all sub-entities.

        Returns:
            Dict[str, Any]: A dictionary mapping sub-entity identifiers to their current states.
        """
        states = {}
        for identifier, entity in self.sub_entities.items():
            state = entity.get_state()
            states[identifier] = state
            self.logger.debug(f"State of '{identifier}': {state}")
        return states

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