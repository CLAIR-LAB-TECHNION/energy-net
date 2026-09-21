# test_multi_agent_coordinator.py
"""
Unit tests for the MultiAgentCoordinator class in multi_agent_env.py
"""

import pytest
from unittest.mock import Mock, MagicMock
from stable_baselines3 import PPO, A2C
from energy_net.gym_envs.agent_config import AgentConfig
from energy_net.gym_envs.multi_agent_env import MultiAgentCoordinator


class TestMultiAgentCoordinatorInitialization:
    """Test MultiAgentCoordinator initialization and validation."""
    
    def test_valid_initialization_default_sequence(self):
        """Test initialization with default training sequence."""
        mock_env1 = Mock()
        mock_env2 = Mock()
        
        agents = [
            AgentConfig("agent1", "ISO", mock_env1, PPO),
            AgentConfig("agent2", "PCS", mock_env2, A2C)
        ]
        
        coordinator = MultiAgentCoordinator(agents, verbose=0, auto_initialize=False)
        
        assert len(coordinator.agents) == 2
        assert coordinator.training_sequence == [0, 1]
        assert coordinator._iteration == 0
    
    def test_valid_initialization_custom_sequence(self):
        """Test initialization with custom training sequence."""
        mock_env1 = Mock()
        mock_env2 = Mock()
        mock_env3 = Mock()
        
        agents = [
            AgentConfig("iso", "ISO", mock_env1, PPO),
            AgentConfig("pcs1", "PCS", mock_env2, PPO),
            AgentConfig("pcs2", "PCS", mock_env3, PPO)
        ]
        
        # Custom sequence: ISO, PCS1, PCS2, PCS1, ISO
        custom_seq = [0, 1, 2, 1, 0]
        coordinator = MultiAgentCoordinator(agents, training_sequence=custom_seq, auto_initialize=False)
        
        assert coordinator.training_sequence == custom_seq
    
    def test_empty_agents_list(self):
        """Test that empty agents list raises ValueError."""
        with pytest.raises(ValueError, match="agents list cannot be empty"):
            MultiAgentCoordinator([])
    
    def test_duplicate_agent_ids(self):
        """Test that duplicate agent IDs raise ValueError."""
        mock_env1 = Mock()
        mock_env2 = Mock()
        
        agents = [
            AgentConfig("same_id", "ISO", mock_env1, PPO),
            AgentConfig("same_id", "PCS", mock_env2, PPO)
        ]
        
        with pytest.raises(ValueError, match="All agent_id values must be unique"):
            MultiAgentCoordinator(agents)
    
    def test_empty_training_sequence(self):
        """Test that empty training sequence raises ValueError."""
        mock_env = Mock()
        agents = [AgentConfig("agent1", "ISO", mock_env, PPO)]
        
        with pytest.raises(ValueError, match="training_sequence cannot be empty"):
            MultiAgentCoordinator(agents, training_sequence=[])
    
    def test_invalid_sequence_index(self):
        """Test that out-of-range sequence index raises ValueError."""
        mock_env1 = Mock()
        mock_env2 = Mock()
        
        agents = [
            AgentConfig("agent1", "ISO", mock_env1, PPO),
            AgentConfig("agent2", "PCS", mock_env2, PPO)
        ]
        
        # Index 2 is out of range (only 0 and 1 are valid)
        with pytest.raises(ValueError, match="training_sequence index 2 out of range"):
            MultiAgentCoordinator(agents, training_sequence=[0, 1, 2])
    
    def test_negative_sequence_index(self):
        """Test that negative sequence index raises ValueError."""
        mock_env = Mock()
        agents = [AgentConfig("agent1", "ISO", mock_env, PPO)]
        
        with pytest.raises(ValueError, match="training_sequence index -1 out of range"):
            MultiAgentCoordinator(agents, training_sequence=[-1])
    
    def test_non_integer_sequence_value(self):
        """Test that non-integer sequence values raise ValueError."""
        mock_env = Mock()
        agents = [AgentConfig("agent1", "ISO", mock_env, PPO)]
        
        with pytest.raises(ValueError, match="training_sequence must contain integers"):
            MultiAgentCoordinator(agents, training_sequence=["0"])


class TestMultiAgentCoordinatorAgentAccess:
    """Test agent access methods."""
    
    def test_get_agent_by_id_found(self):
        """Test getting agent by ID when it exists."""
        mock_env1 = Mock()
        mock_env2 = Mock()
        
        agents = [
            AgentConfig("iso_agent", "ISO", mock_env1, PPO),
            AgentConfig("pcs_agent", "PCS", mock_env2, PPO)
        ]
        
        coordinator = MultiAgentCoordinator(agents, auto_initialize=False)
        agent = coordinator.get_agent_by_id("pcs_agent")
        
        assert agent is not None
        assert agent.agent_id == "pcs_agent"
        assert agent.agent_type == "PCS"
    
    def test_get_agent_by_id_not_found(self):
        """Test getting agent by ID when it doesn't exist."""
        mock_env = Mock()
        agents = [AgentConfig("agent1", "ISO", mock_env, PPO)]
        
        coordinator = MultiAgentCoordinator(agents, auto_initialize=False)
        agent = coordinator.get_agent_by_id("nonexistent")
        
        assert agent is None
    
    def test_get_agent_by_index(self):
        """Test getting agent by index."""
        mock_env1 = Mock()
        mock_env2 = Mock()
        
        agents = [
            AgentConfig("agent1", "ISO", mock_env1, PPO),
            AgentConfig("agent2", "PCS", mock_env2, PPO)
        ]
        
        coordinator = MultiAgentCoordinator(agents, auto_initialize=False)
        agent = coordinator.get_agent_by_index(1)
        
        assert agent.agent_id == "agent2"
    
    def test_get_agent_by_index_out_of_range(self):
        """Test that out-of-range index raises IndexError."""
        mock_env = Mock()
        agents = [AgentConfig("agent1", "ISO", mock_env, PPO)]
        
        coordinator = MultiAgentCoordinator(agents, auto_initialize=False)
        
        with pytest.raises(IndexError):
            coordinator.get_agent_by_index(5)


class TestMultiAgentCoordinatorTraining:
    """Test training methods."""
    
    def test_train_single_cycle(self):
        """Test executing a single training cycle."""
        mock_env1 = Mock()
        mock_env2 = Mock()
        
        agents = [
            AgentConfig("agent1", "ISO", mock_env1, PPO, timesteps_per_turn=10),
            AgentConfig("agent2", "PCS", mock_env2, PPO, timesteps_per_turn=20)
        ]
        
        coordinator = MultiAgentCoordinator(agents, verbose=0, auto_initialize=False)
        
        # Mock the train methods
        agents[0].model = Mock()
        agents[1].model = Mock()
        
        metrics = coordinator.train_single_cycle()
        
        assert metrics["iteration"] == 1
        assert len(metrics["agents_trained"]) == 2
        assert metrics["agents_trained"][0]["agent_id"] == "agent1"
        assert metrics["agents_trained"][1]["agent_id"] == "agent2"
        assert coordinator._iteration == 1
    
    def test_train_custom_sequence(self):
        """Test training with custom sequence."""
        mock_env1 = Mock()
        mock_env2 = Mock()
        
        agents = [
            AgentConfig("agent1", "ISO", mock_env1, PPO),
            AgentConfig("agent2", "PCS", mock_env2, PPO)
        ]
        
        # Train agent2 twice
        coordinator = MultiAgentCoordinator(
            agents,
            training_sequence=[1, 0, 1],
            verbose=0,
            auto_initialize=False
        )
        
        agents[0].model = Mock()
        agents[1].model = Mock()
        
        metrics = coordinator.train_single_cycle()
        
        assert len(metrics["agents_trained"]) == 3
        assert metrics["agents_trained"][0]["agent_id"] == "agent2"
        assert metrics["agents_trained"][1]["agent_id"] == "agent1"
        assert metrics["agents_trained"][2]["agent_id"] == "agent2"
    
    def test_train_multiple_iterations(self):
        """Test training for multiple iterations."""
        mock_env = Mock()
        agents = [AgentConfig("agent1", "ISO", mock_env, PPO)]
        
        coordinator = MultiAgentCoordinator(agents, verbose=0, auto_initialize=False)
        agents[0].model = Mock()
        
        history = coordinator.train(total_iterations=3)
        
        assert coordinator._iteration == 3
        assert len(history["iteration"]) == 3
    
    def test_train_with_callback(self):
        """Test training with callback function."""
        mock_env = Mock()
        agents = [AgentConfig("agent1", "ISO", mock_env, PPO)]
        
        coordinator = MultiAgentCoordinator(agents, verbose=0, auto_initialize=False)
        agents[0].model = Mock()
        
        callback_results = []
        def callback(metrics):
            callback_results.append(metrics["iteration"])
        
        coordinator.train(total_iterations=2, callback=callback)
        
        assert callback_results == [1, 2]
    
    def test_train_invalid_iterations(self):
        """Test that invalid total_iterations raises ValueError."""
        mock_env = Mock()
        agents = [AgentConfig("agent1", "ISO", mock_env, PPO)]
        
        coordinator = MultiAgentCoordinator(agents, verbose=0, auto_initialize=False)
        
        with pytest.raises(ValueError, match="total_iterations must be >= 1"):
            coordinator.train(total_iterations=0)


class TestMultiAgentCoordinatorSummary:
    """Test summary and representation methods."""
    
    def test_get_summary(self):
        """Test getting coordinator summary."""
        mock_env1 = Mock()
        mock_env2 = Mock()
        
        agents = [
            AgentConfig("iso", "ISO", mock_env1, PPO, timesteps_per_turn=7),
            AgentConfig("pcs", "PCS", mock_env2, A2C, timesteps_per_turn=48)
        ]
        
        coordinator = MultiAgentCoordinator(
            agents,
            training_sequence=[0, 1, 0],
            verbose=0,
            auto_initialize=False
        )
        
        summary = coordinator.get_summary()
        
        assert summary["num_agents"] == 2
        assert len(summary["agents"]) == 2
        assert summary["agents"][0]["id"] == "iso"
        assert summary["agents"][0]["algorithm"] == "PPO"
        assert summary["agents"][1]["id"] == "pcs"
        assert summary["agents"][1]["algorithm"] == "A2C"
        assert summary["training_sequence"] == [0, 1, 0]
        assert summary["sequence_names"] == ["iso", "pcs", "iso"]
        assert summary["iterations_completed"] == 0
    
    def test_repr(self):
        """Test string representation of coordinator."""
        mock_env1 = Mock()
        mock_env2 = Mock()
        
        agents = [
            AgentConfig("agent1", "ISO", mock_env1, PPO),
            AgentConfig("agent2", "PCS", mock_env2, PPO)
        ]
        
        coordinator = MultiAgentCoordinator(agents, verbose=0, auto_initialize=False)
        
        repr_str = repr(coordinator)
        assert "agents=2" in repr_str
        assert "sequence_length=2" in repr_str
        assert "iterations=0" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])