# Training the ISO Agent with Predefined PCS Actions

This guide explains how to train the ISO agent in an environment where the PCS agent's actions are completely predefined. This feature allows you to have full control over the PCS agent's behavior while training the ISO agent.

## Overview

The implementation allows you to:

1. Pass in a sequence of PCS actions of length T at the beginning of training
2. Have the environment automatically apply the next action from this sequence on each call to `env.step()`
3. Train the ISO agent as usual, with the agent learning to respond to this fixed sequence of PCS actions

## Generating PCS Action Sequences

Before training, you need to create a sequence of PCS actions. You can use the provided `create_pcs_action_sequence.py` script to generate action sequences in several ways:

### 1. Random actions

Generate a sequence of random PCS actions:

```bash
python create_pcs_action_sequence.py \
  --output-file pcs_actions/random_actions.npy \
  --sequence-length 1000 \
  --method pattern \
  --seed 42
```

### 2. From a trained policy

Generate actions by running a previously trained PCS policy:

```bash
python create_pcs_action_sequence.py \
  --output-file pcs_actions/policy_actions.npy \
  --sequence-length 1000 \
  --method from_policy \
  --policy-path models/ppo_pcs_best.zip \
  --policy-type ppo \
  --demand-pattern SINUSOIDAL \
  --pricing-policy ONLINE \
  --cost-type CONSTANT
```

### 3. Using predefined patterns

Generate actions following a pattern like charge-discharge cycles:

```bash
python create_pcs_action_sequence.py \
  --output-file pcs_actions/pattern_actions.npy \
  --sequence-length 1000 \
  --method pattern \
  --pattern-type charge_discharge_cycle \
  --cycle-length 48
```

Or price-responsive behavior:

```bash
python create_pcs_action_sequence.py \
  --output-file pcs_actions/price_responsive_actions.npy \
  --sequence-length 1000 \
  --method pattern \
  --pattern-type price_responsive \
  --cycle-length 48
```

## Training the ISO Agent

To train the ISO agent with a predefined sequence of PCS actions, use the `--pcs-action-file` parameter:

```bash
python train_iso_recurrent.py \
  --demand-pattern SINUSOIDAL \
  --pricing-policy CONSTANT \
  --cost-type CONSTANT \
  --use-dispatch \
  --iterations 10 \
  --algorithm td3 \
  --timesteps 4800 \
  --pcs-action-file pcs_actions/pattern_actions.npy
```

The training script will:

1. Load the PCS action sequence from the specified file
2. Initialize the appropriate wrapper that consumes actions from this sequence
3. Train the ISO agent as usual

If the episode length exceeds the sequence length, the wrapper will cycle through the sequence by using modulo indexing.

## Implementation Details

The implementation consists of the following components:

1. `PreDefinedPCSWrapper`: A new wrapper class that extends `ISOEnvWrapper` and uses a predefined sequence of PCS actions
2. Updates to `make_iso_env` function to support the new wrapper
3. Updates to `train_iso_recurrent.py` to accept the PCS action sequence file
4. A new script `create_pcs_action_sequence.py` to generate action sequences

The `PreDefinedPCSWrapper`:
- Takes a numpy array of PCS actions and cycles through them
- Validates the actions to ensure they're within the valid range
- Provides detailed logging for debugging
- Handles edge cases like missing or invalid action sequences

## Advanced Usage

### 1. Combining with trained PCS policy

You can first train a PCS agent, then generate a sequence of actions from it, and finally use that sequence to train the ISO agent. This allows you to train the agents sequentially while ensuring consistent PCS behavior:

```bash
# First train a PCS agent
python train_pcs.py ...

# Generate an action sequence from the trained policy
python create_pcs_action_sequence.py --method from_policy --policy-path models/pcs_best.zip ...

# Train the ISO agent with the fixed sequence
python train_iso_recurrent.py --pcs-action-file pcs_actions/from_trained_policy.npy ...
```

### 2. Testing different PCS strategies

You can generate multiple action sequences representing different PCS strategies and evaluate how the ISO agent learns to respond to each:

```bash
# Generate sequences for different strategies
python create_pcs_action_sequence.py --method pattern --pattern-type charge_discharge_cycle ...
python create_pcs_action_sequence.py --method pattern --pattern-type price_responsive ...

# Train ISO agents for each strategy
python train_iso_recurrent.py --pcs-action-file pcs_actions/strategy1.npy ...
python train_iso_recurrent.py --pcs-action-file pcs_actions/strategy2.npy ...

# Compare the results
```

### 3. Realistic market situations

For more realistic simulations, you can create action sequences that represent specific market behaviors or historical patterns:

```bash
# Generate a custom sequence (you might need to modify the script for specific patterns)
python create_pcs_action_sequence.py --method pattern --pattern-type custom ...

# Train the ISO agent with this realistic scenario
python train_iso_recurrent.py --pcs-action-file pcs_actions/market_scenario.npy ...
```

## Troubleshooting

- **Issue**: PCS actions don't match expectations during training
  - **Solution**: Check that your actions are in the expected range (-1 to 1) and have the correct shape

- **Issue**: Training performance degrades when using predefined actions
  - **Solution**: Try modifying the sequence to provide a better learning signal to the ISO agent. Very random or extreme actions can make learning difficult.

- **Issue**: Agent doesn't seem to adapt to the PCS actions
  - **Solution**: Increase the training time (iterations or timesteps) as learning to respond to a fixed pattern may take longer

## Future Enhancements

Possible improvements to this feature:

1. Support for multiple PCS agents with different action sequences
2. Ability to modify the sequence during training based on certain criteria
3. Tools to visualize the PCS action sequence and ISO responses together
4. Integration with economic models to generate more realistic sequences 