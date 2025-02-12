# components/grid_entity.py

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging

from energy_net.utils.logger import setup_logger  # Import the logger setup utility


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
    def perform_action(self, action: float) -> None:
        """
        Performs an action (e.g., charging or discharging) on the grid entity.

        This method must be implemented by all subclasses, defining how the entity responds
        to a given action.

        Args:
            action (float): The action to perform. The meaning of the action depends on the entity.
                            For example, positive values might indicate charging, while negative
                            values indicate discharging for a Battery.
        """
        pass

    @abstractmethod
    def get_state(self) -> float:
        """
        Retrieves the current state of the grid entity.

        This method must be implemented by all subclasses, providing a way to access the
        entity's current state (e.g., energy level for a Battery).

        Returns:
            float: The current state of the entity.
        """
        pass

    @abstractmethod
    def update(self, time: float, action: float = 0.0) -> None:
        """
        Updates the state of the grid entity based on the current time and action.

        This method must be implemented by all subclasses, defining how the entity's state
        evolves over time and in response to actions.

        Args:
            time (float): The current time as a fraction of the day (0 to 1).
            action (float, optional): The action to perform (default is 0.0).
                                      The meaning of the action depends on the entity.
        """
        pass


class CompositeGridEntity(GridEntity):
    """
    Represents a composite grid entity composed of multiple sub-entities.

    Manages actions and updates across all sub-entities and aggregates their states.
    """

    def __init__(self, sub_entities: List[GridEntity], log_file: str):
        """
        Initializes the CompositeGridEntity with specified sub-entities and sets up logging.

        Args:
            sub_entities (List[GridEntity]): A list of sub-entities composing this composite entity.
            log_file (str): Path to the log file for the composite grid entity.
        """
        super().__init__(log_file)
        self.sub_entities: Dict[str, GridEntity] = {}
        self._initialize_sub_entities(sub_entities)
        self.logger.info(f"CompositeGridEntity initialized with {len(self.sub_entities)} sub-entities.")

    def _initialize_sub_entities(self, sub_entities: List[GridEntity]) -> None:
        """
        Assigns unique identifiers to each sub-entity and stores them in a dictionary.

        Args:
            sub_entities (List[GridEntity]): The list of sub-entities to be managed.
        """
        for idx, entity in enumerate(sub_entities):
            identifier = f"{entity.__class__.__name__}_{idx}"
            self.sub_entities[identifier] = entity
            self.logger.debug(f"Sub-entity added with ID '{identifier}': {entity}")

    def perform_action(self, actions: Dict[str, float]) -> None:
        """
        Performs actions on all sub-entities.

        Args:
            actions (Dict[str, float]): A dictionary mapping sub-entity identifiers to actions.
        """
        self.logger.debug(f"Performing actions: {actions}")
        for identifier, action in actions.items():
            entity = self.sub_entities.get(identifier)
            if entity:
                self.logger.info(f"Performing action on '{identifier}': {action}")
                entity.perform_action(action)
            else:
                self.logger.warning(f"No sub-entity found with ID '{identifier}'.")

    def get_state(self) -> Dict[str, float]:
        """
        Retrieves the current states of all sub-entities.

        Returns:
            Dict[str, float]: A dictionary mapping sub-entity identifiers to their current states.
        """
        states = {}
        for identifier, entity in self.sub_entities.items():
            state = entity.get_state()
            states[identifier] = state
            self.logger.debug(f"State of '{identifier}': {state}")
        return states

    def update(self, time: float, actions: Dict[str, float] = {}) -> None:
        """
        Updates all sub-entities based on the current time and their respective actions.

        Args:
            time (float): The current time as a fraction of the day (0 to 1).
            actions (Dict[str, float], optional): A dictionary of actions for each sub-entity.
        """
        self.logger.debug(f"Updating CompositeGridEntity with time: {time} and actions: {actions}")
        for identifier, entity in self.sub_entities.items():
            action = actions.get(identifier, 0.0)
            self.logger.info(f"Updating '{identifier}' with action: {action}")
            entity.update(time, action)
        self.logger.info("All sub-entities have been updated.")

    def reset(self) -> None:
        """
        Resets all sub-entities to their initial states.
        """
        super().reset()
        self.logger.info("Resetting all sub-entities to their initial states.")
        for identifier, entity in self.sub_entities.items():
            self.logger.info(f"Resetting sub-entity '{identifier}'.")
            entity.reset()
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
