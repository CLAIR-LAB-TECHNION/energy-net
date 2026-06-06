# test_multi_agent_config.py
"""
Unit tests for the AgentConfig class in multi_agent_env.py
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from stable_baselines3 import PPO, A2C
from energy_net.gym_envs.agent_config import AgentConfig


class TestAgentConfigValidation:
    """Test validation in AgentConfig.__post_init__"""
    
    def test_valid_iso_config(self):
        """Test creating a valid ISO agent configuration."""
        mock_env = Mock()
        config = AgentConfig(
            agent_id="test_iso",
            agent_type="ISO",
            env=mock_env,
            algo_class=PPO,
            policy="MlpPolicy",
            timesteps_per_turn=7
        )
        assert config.agent_id == "test_iso"
        assert config.agent_type == "ISO"
        assert config.timesteps_per_turn == 7
        assert config.model is None
    
    def test_valid_pcs_config(self):
        """Test creating a valid PCS agent configuration."""
        mock_env = Mock()
        config = AgentConfig(
            agent_id="test_pcs",
            agent_type="PCS",
            env=mock_env,
            algo_class=A2C,
            policy="MlpPolicy",
            timesteps_per_turn=48
        )
        assert config.agent_id == "test_pcs"
        assert config.agent_type == "PCS"
        assert config.algo_class == A2C
    
    def test_invalid_agent_type(self):
        """Test that invalid agent_type raises ValueError."""
        mock_env = Mock()
        with pytest.raises(ValueError, match="agent_type must be 'ISO' or 'PCS'"):
            AgentConfig(
                agent_id="test",
                agent_type="INVALID",
                env=mock_env,
                algo_class=PPO
            )
    
    def test_empty_agent_id(self):
        """Test that empty agent_id raises ValueError."""
        mock_env = Mock()
        with pytest.raises(ValueError, match="agent_id cannot be empty"):
            AgentConfig(
                agent_id="",
                agent_type="ISO",
                env=mock_env,
                algo_class=PPO
            )
    
    def test_invalid_timesteps_per_turn(self):
        """Test that timesteps_per_turn < 1 raises ValueError."""
        mock_env = Mock()
        with pytest.raises(ValueError, match="timesteps_per_turn must be >= 1"):
            AgentConfig(
                agent_id="test",
                agent_type="ISO",
                env=mock_env,
                algo_class=PPO,
                timesteps_per_turn=0
            )
    
    def test_invalid_algo_kwargs_type(self):
        """Test that non-dict algo_kwargs raises ValueError."""
        mock_env = Mock()
        with pytest.raises(ValueError, match="algo_kwargs must be a dictionary"):
            AgentConfig(
                agent_id="test",
                agent_type="ISO",
                env=mock_env,
                algo_class=PPO,
                algo_kwargs="not_a_dict"
            )
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        mock_env = Mock()
        config = AgentConfig(
            agent_id="test",
            agent_type="ISO",
            env=mock_env,
            algo_class=PPO
        )
        assert config.policy == "MlpPolicy"
        assert config.timesteps_per_turn == 48
        assert config.algo_kwargs == {}
        assert config.model is None


class TestAgentConfigModelManagement:
    """Test model initialization and management methods."""
    
    def test_initialize_model(self):
        """Test model initialization."""
        mock_env = Mock()
        mock_env.observation_space = Mock()
        mock_env.action_space = Mock()
        
        # Create a mock algorithm class
        mock_algo_class = Mock()
        mock_model = Mock()
        mock_algo_class.return_value = mock_model
        
        config = AgentConfig(
            agent_id="test",
            agent_type="ISO",
            env=mock_env,
            algo_class=mock_algo_class,
            algo_kwargs={"n_steps": 7}
        )
        
        model = config.initialize_model(verbose=1)
        
        # Verify model was created with correct parameters
        mock_algo_class.assert_called_once_with(
            "MlpPolicy",
            mock_env,
            verbose=1,
            n_steps=7
        )
        assert config.model == mock_model
        assert model == mock_model
    
    def test_initialize_model_idempotent(self):
        """Test that initialize_model returns existing model if already initialized."""
        mock_env = Mock()
        mock_model = Mock()
        
        config = AgentConfig(
            agent_id="test",
            agent_type="ISO",
            env=mock_env,
            algo_class=PPO
        )
        config.model = mock_model
        
        # Should return existing model without creating new one
        result = config.initialize_model()
        assert result == mock_model
    
    def test_train_without_initialization(self):
        """Test that training without model initialization raises error."""
        mock_env = Mock()
        config = AgentConfig(
            agent_id="test",
            agent_type="ISO",
            env=mock_env,
            algo_class=PPO
        )
        
        with pytest.raises(RuntimeError, match="model not initialized"):
            config.train()
    
    def test_train_with_model(self):
        """Test training an initialized model."""
        mock_env = Mock()
        mock_model = Mock()
        
        config = AgentConfig(
            agent_id="test_agent",
            agent_type="ISO",
            env=mock_env,
            algo_class=PPO,
            timesteps_per_turn=10
        )
        config.model = mock_model
        
        result = config.train(reset_num_timesteps=True)
        
        # Verify learn was called with correct parameters
        mock_model.learn.assert_called_once_with(
            total_timesteps=10,
            reset_num_timesteps=True
        )
        
        # Verify return value
        assert result["agent_id"] == "test_agent"
        assert result["agent_type"] == "ISO"
        assert result["timesteps_trained"] == 10
    
    def test_predict_without_initialization(self):
        """Test that predict without model initialization raises error."""
        mock_env = Mock()
        config = AgentConfig(
            agent_id="test",
            agent_type="ISO",
            env=mock_env,
            algo_class=PPO
        )
        
        with pytest.raises(RuntimeError, match="model not initialized"):
            config.predict(np.array([1, 2, 3]))
    
    def test_predict_with_model(self):
        """Test prediction with initialized model."""
        mock_env = Mock()
        mock_model = Mock()
        mock_model.predict.return_value = (np.array([0.5]), None)
        
        config = AgentConfig(
            agent_id="test",
            agent_type="ISO",
            env=mock_env,
            algo_class=PPO
        )
        config.model = mock_model
        
        obs = np.array([1, 2, 3])
        action, state = config.predict(obs, deterministic=True)
        
        mock_model.predict.assert_called_once_with(obs, deterministic=True)
        assert np.array_equal(action, np.array([0.5]))


class TestAgentConfigRepresentation:
    """Test string representation of AgentConfig."""
    
    def test_repr_uninitialized(self):
        """Test __repr__ for uninitialized model."""
        mock_env = Mock()
        config = AgentConfig(
            agent_id="test_iso",
            agent_type="ISO",
            env=mock_env,
            algo_class=PPO,
            timesteps_per_turn=7
        )
        
        repr_str = repr(config)
        assert "test_iso" in repr_str
        assert "ISO" in repr_str
        assert "PPO" in repr_str
        assert "timesteps=7" in repr_str
        assert "not initialized" in repr_str
    
    def test_repr_initialized(self):
        """Test __repr__ for initialized model."""
        mock_env = Mock()
        mock_model = Mock()
        
        config = AgentConfig(
            agent_id="test_pcs",
            agent_type="PCS",
            env=mock_env,
            algo_class=A2C,
            timesteps_per_turn=48
        )
        config.model = mock_model
        
        repr_str = repr(config)
        assert "test_pcs" in repr_str
        assert "PCS" in repr_str
        assert "A2C" in repr_str
        assert "initialized" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])