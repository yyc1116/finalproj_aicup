# AGENTS.md

## Project Overview

This repository is for a table-tennis sequential prediction competition.

The raw data is stroke-level data. Each row represents one stroke in a rally.

The task is to predict, for each rally sequence:

1. the next stroke `actionId`
2. the next stroke `pointId`
3. the rally result `serverGetPoint`

## Data Semantics

The following values are valid classes and must not be treated as missing values:

- `actionId = 0` means none / other / unrecognized stroke type.
- `pointId = 0` means no valid 3x3 landing-grid point, such as net, out, or direct out.

`pointId` is defined from the receiver player's dominant-hand perspective, not from absolute left/right table coordinates.

`serverGetPoint` is binary:

- `1`: server gets the point
- `0`: server does not get the point

## Data Leakage Rules

Never use future strokes to construct features.

For a training sample that predicts stroke `t + 1`, features may only use strokes up to and including stroke `t`.

Never use the following target columns as model features:

- `target_actionId`
- `target_pointId`
- `target_serverGetPoint`
- `serverGetPoint`

Exception: `serverGetPoint` may only be used as the label for the win model. It must not be used as an input feature.

Do not use random row split for validation.

Use group-based validation split:

1. Prefer grouping by `match`.
2. If `match` is unavailable or unsuitable, group by `rally_uid`.

Train and validation sets must have no overlapping `rally_uid`.

If splitting by `match`, train and validation sets must have no overlapping `match`.

## Feature Engineering Rules

The feature builder should convert stroke-level data into tabular prefix-prediction samples.

For training:

- Sort by `rally_uid` and `strikeNumber`.
- For each rally prefix ending at stroke `t`, use the last `k` known strokes as lag features.
- Predict stroke `t + 1`.
- Drop the last stroke of each rally for next-stroke prediction because it has no next stroke.

For testing:

- Produce exactly one row per `rally_uid`.
- Use the known test sequence for that `rally_uid`, especially the last `k` strokes, to predict the next stroke and rally outcome.

Lag features should use names such as:

- `prev1_actionId`
- `prev1_pointId`
- `prev1_spinId`
- `prev2_actionId`
- `prev2_pointId`
- `prev2_spinId`

Missing lag values caused by short prefixes should be filled with `0`.

## Submission Format

The final submission CSV must have exactly these columns in this order:

```csv
rally_uid,actionId,pointId,serverGetPoint
```

## Required Validation Checks

When implementing or modifying the feature pipeline, add checks for:

- no overlapping rally_uid between train and validation
- no overlapping match between train and validation when match-based split is used
- no target columns used as features
- feature_max_strikeNumber < target_strikeNumber for every next-stroke training sample
- actionId = 0 and pointId = 0 remain valid classes
- scoreSelf and scoreOther are inspected for possible leakage