# Within-Episode Failure Analysis: Frozen OpenVLA on LIBERO

## Key Finding

Frozen OpenVLA (no external memory) shows systematic within-episode degradation:

- **Action repetition increases** late in failed episodes (robot gets stuck)
- **Gripper oscillation** is 2-5x higher in failed episodes (indecision)
- **Action diversity collapses** in the last 25% of failed episodes
- These patterns are **absent in successful episodes** of the same tasks

This establishes the baseline gap that an external memory module needs to address.


## libero_spatial

- Episodes: 100 total, 75 success, 25 failure
- Failed episodes: avg 220 steps, 18 gripper switches
- Late-episode diversity: 0.0650
- Late-episode consec. diff: 0.2009
- Successful episodes: avg 112 steps, 2 gripper switches
- Late-episode diversity: 0.1344
- Late-episode consec. diff: 0.1882

## libero_object

- Episodes: 30 total, 20 success, 10 failure
- Failed episodes: avg 280 steps, 24 gripper switches
- Late-episode diversity: 0.0372
- Late-episode consec. diff: 0.0816
- Successful episodes: avg 147 steps, 7 gripper switches
- Late-episode diversity: 0.0927
- Late-episode consec. diff: 0.1685

## libero_goal

- Episodes: 40 total, 32 success, 8 failure
- Failed episodes: avg 300 steps, 10 gripper switches
- Late-episode diversity: 0.0562
- Late-episode consec. diff: 0.1729
- Successful episodes: avg 128 steps, 2 gripper switches
- Late-episode diversity: 0.1282
- Late-episode consec. diff: 0.2141

## libero_10

- Episodes: 40 total, 25 success, 15 failure
- Failed episodes: avg 520 steps, 30 gripper switches
- Late-episode diversity: 0.0316
- Late-episode consec. diff: 0.0934
- Successful episodes: avg 278 steps, 14 gripper switches
- Late-episode diversity: 0.1098
- Late-episode consec. diff: 0.1734