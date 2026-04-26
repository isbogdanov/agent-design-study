# Learning Metrics System

This system tracks detailed performance metrics during continual learning sessions for easy analysis and plotting.

## Features

- **Automatic JSON logging** of attempts, steps, rewards, and reflections
- **Real-time metrics** updated during each step of learning
- **Session summaries** with success rates and performance statistics  
- **Plotting utilities** for visualizing learning progress
- **Exportable data** for external analysis tools

## Files

- `learning_metrics.py` - Core metrics tracking class
- `plot_learning_metrics.py` - Plotting and analysis utility
- `learning_metrics.json` - Auto-generated metrics file (in session directory)

## Usage

### Automatic Logging (Built-in)

When you run continual learning mode, metrics are automatically tracked:

```bash
python run_cyborg_coordinator.py --learning --agent_to_improve ActionChooser
```

Metrics are saved to: `logs/runs/learning/learning_session_YYYYMMDD_HHMMSS/learning_metrics.json`

### Viewing Learning Progress

#### Quick Summary
```bash
python utils/plot_learning_metrics.py logs/runs/learning/learning_session_*/learning_metrics.json --no-plot
```

#### Full Analysis with Plots
```bash
python utils/plot_learning_metrics.py logs/runs/learning/learning_session_*/learning_metrics.json
```

#### Save Plots to Specific Directory
```bash
python utils/plot_learning_metrics.py path/to/learning_metrics.json --output-dir plots/
```

## Metrics Structure

### JSON Format
```json
{
  "session_info": {
    "session_start_time": "2025-08-27T23:24:13.719000",
    "total_attempts": 3,
    "successful_attempts": 1,
    "failed_attempts": 2
  },
  "attempts": {
    "1": {
      "attempt_number": 1,
      "steps_completed": 3,
      "total_reward": -0.1,
      "step_rewards": [
        {"step": 1, "reward": 0.0, "cumulative_reward": 0.0},
        {"step": 2, "reward": 0.0, "cumulative_reward": 0.0},
        {"step": 3, "reward": -0.1, "cumulative_reward": -0.1}
      ],
      "actions_taken": [
        {"step": 1, "action": "Monitor", "timestamp": "..."},
        {"step": 2, "action": "Monitor", "timestamp": "..."},
        {"step": 3, "action": "Monitor", "timestamp": "..."}
      ],
      "reflection_triggered": true,
      "reflection_step": 3,
      "reflection_reason": "reward_-0.1_below_threshold_-0.05",
      "attempt_status": "failed"
    }
  }
}
```

### Key Metrics Tracked

- **Per Attempt**: Steps completed, total reward, average reward, success/failure
- **Per Step**: Individual rewards, actions taken, timestamps
- **Reflections**: When triggered, at what step, and why
- **Session Summary**: Success rates, best/worst performance, trends

## Plotting Features

The plotting utility creates 4 comprehensive visualizations:

1. **Total Reward per Attempt** - Bar chart showing overall performance
2. **Steps Completed per Attempt** - How long each attempt lasted  
3. **Learning Efficiency** - Average reward per step over time
4. **Step-by-Step Progress** - Detailed view of recent attempts

### Plot Features:
- ✅ **Color coding**: Green=successful, Red=failed attempts
- 🔵 **Reflection markers**: Shows when/where reflection was triggered
- 📈 **Trend lines**: Track learning efficiency over attempts
- 🎯 **Success highlighting**: Emphasizes breakthrough moments

## Example Output

```
============================================================
LEARNING SESSION SUMMARY
============================================================
Session Start: 2025-08-27T23:24:13.719000
Total Attempts: 5
Successful Attempts: 2
Failed Attempts: 3
Success Rate: 40.0%
Best Reward: 15.2
Worst Reward: -2.1
Average Reward: 3.8
Average Steps per Attempt: 7.2
Reflections Triggered: 3

PER-ATTEMPT BREAKDOWN:
------------------------------------------------------------
Attempt  1: ✗ Steps= 3 Reward= -0.10 (R@3)
Attempt  2: ✗ Steps= 5 Reward= -1.20 (R@5)  
Attempt  3: ✓ Steps=10 Reward= 12.50
Attempt  4: ✗ Steps= 4 Reward= -2.10 (R@4)
Attempt  5: ✓ Steps=10 Reward= 15.20
```

## Integration Points

### In `run_cyborg_coordinator.py`:
- Metrics tracker automatically created for learning sessions
- Step rewards logged after each environment step
- Reflection events logged when triggered
- Session summary printed at completion

### In `ReflectionCoordinator`:
- Receives metrics tracker for reflection logging
- Can access historical performance data for better reflection

### Extensibility:
- Easy to add new metrics by extending `LearningMetricsTracker`
- JSON format allows integration with external analysis tools
- Plotting utility can be customized for specific visualizations

## Dependencies

- **Core functionality**: Python standard library only
- **Plotting**: `matplotlib` (optional, install with `pip install matplotlib`)




