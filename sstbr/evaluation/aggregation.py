"""Public construction of 10-second predictions and direct references."""
from __future__ import annotations

import pandas as pd


PREDICTION_KEYS = ["subject_id", "source_segment_id", "local_window_id"]
EVALUATION_KEYS = ["subject_id", "source_segment_id", "evaluation_window_id"]


def aggregate_consecutive_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Average consecutive local prediction pairs within each source segment."""
    required = {*PREDICTION_KEYS, "pred_hr"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
    frame = predictions.copy()
    frame["local_window_id"] = pd.to_numeric(frame.local_window_id, errors="raise").astype(int)
    if frame.duplicated(PREDICTION_KEYS).any():
        raise ValueError("Duplicate local prediction key.")
    if (frame.local_window_id < 0).any():
        raise ValueError("local_window_id must be non-negative.")
    frame["evaluation_window_id"] = frame.local_window_id // 2
    group_keys = [column for column in ["fold", *EVALUATION_KEYS] if column in frame.columns]
    # Final evaluation scale: average each adjacent pair of local estimates
    # into one 10-s prediction; PPG references are attached independently.
    result = frame.groupby(group_keys, sort=False, as_index=False).agg(
        pred_hr=("pred_hr", "mean"),
        source_window_count=("local_window_id", "size"),
        source_window_ids=("local_window_id", lambda values: ",".join(map(str, sorted(values)))),
    )
    if not result.source_window_count.eq(2).all():
        raise ValueError("Every 10-second prediction requires exactly two consecutive local predictions.")
    expected_pairs = result.source_window_ids.str.split(",").map(lambda pair: int(pair[1]) == int(pair[0]) + 1 and int(pair[0]) % 2 == 0)
    if not expected_pairs.all():
        raise ValueError("Local prediction IDs must form pairs (0,1), (2,3), ...")
    return result


def attach_direct_references(aggregated: pd.DataFrame, direct_labels: pd.DataFrame) -> pd.DataFrame:
    """Attach independently estimated 10-second PPG references by explicit keys."""
    required = {*EVALUATION_KEYS, "hr_bpm"}
    missing = required - set(direct_labels.columns)
    if missing:
        raise ValueError(f"Direct-label table is missing columns: {sorted(missing)}")
    if direct_labels.duplicated(EVALUATION_KEYS).any():
        raise ValueError("Duplicate direct-reference key.")
    references = direct_labels[[*EVALUATION_KEYS, "hr_bpm"]].rename(columns={"hr_bpm": "gt_hr"})
    output = aggregated.merge(references, on=EVALUATION_KEYS, how="left", validate="many_to_one", indicator=True)
    if not output._merge.eq("both").all():
        raise ValueError("A 10-second prediction has no matching direct PPG reference.")
    output = output.drop(columns="_merge")
    output["error"] = output.pred_hr - output.gt_hr
    return output
