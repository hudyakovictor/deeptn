"""
Config validation for DEEPUTIN pipeline.

Usage:
    from deeputin.shared.config_validation import validate_config
    
    config = load_yaml("pipeline.yaml")
    errors = validate_config(config)
    if errors:
        for err in errors:
            log.error(err)
"""

from __future__ import annotations

from typing import Any, Dict, List


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate pipeline configuration.
    
    Returns list of error messages (empty if valid).
    """
    errors = []
    
    if not isinstance(config, dict):
        return ["Config must be a dictionary"]
    
    # Validate stages
    if "stages" in config:
        stages = config["stages"]
        if not isinstance(stages, list):
            errors.append("'stages' must be a list")
        else:
            valid_stages = {"s1", "s2", "s3", "s4", "s5", "s6"}
            for stage in stages:
                if stage not in valid_stages:
                    errors.append(f"Invalid stage '{stage}', must be one of {valid_stages}")
    
    # Validate S1 config
    if "s1" in config:
        s1 = config["s1"]
        if not isinstance(s1, dict):
            errors.append("'s1' config must be a dictionary")
        else:
            # device
            if "device" in s1:
                device = s1["device"]
                if device not in ("auto", "cpu", "cuda"):
                    errors.append(f"s1.device must be 'auto', 'cpu', or 'cuda', got '{device}'")
            
            # backbone
            if "backbone" in s1:
                backbone = s1["backbone"]
                if backbone not in ("resnet50", "mobilenet"):
                    errors.append(f"s1.backbone must be 'resnet50' or 'mobilenet', got '{backbone}'")
            
            # neutral_expression
            if "neutral_expression" in s1:
                if not isinstance(s1["neutral_expression"], bool):
                    errors.append("s1.neutral_expression must be boolean")
            
            # identity_only
            if "identity_only" in s1:
                if not isinstance(s1["identity_only"], bool):
                    errors.append("s1.identity_only must be boolean")
    
    # Validate S2 config
    if "s2" in config:
        s2 = config["s2"]
        if not isinstance(s2, dict):
            errors.append("'s2' config must be a dictionary")
        else:
            # geometry_evidence_table
            if "geometry_evidence_table" in s2:
                if not isinstance(s2["geometry_evidence_table"], str):
                    errors.append("s2.geometry_evidence_table must be a string path")
            
            # texture_leaderboard
            if "texture_leaderboard" in s2:
                if not isinstance(s2["texture_leaderboard"], str):
                    errors.append("s2.texture_leaderboard must be a string path")
    
    # Validate S3 config
    if "s3" in config:
        s3 = config["s3"]
        if not isinstance(s3, dict):
            errors.append("'s3' config must be a dictionary")
        else:
            # min_calibration_pairs
            if "min_calibration_pairs" in s3:
                val = s3["min_calibration_pairs"]
                if not isinstance(val, int) or val < 1:
                    errors.append("s3.min_calibration_pairs must be a positive integer")
    
    # Validate S4 config
    if "s4" in config:
        s4 = config["s4"]
        if not isinstance(s4, dict):
            errors.append("'s4' config must be a dictionary")
        else:
            # anchor_every_n
            if "anchor_every_n" in s4:
                val = s4["anchor_every_n"]
                if not isinstance(val, int) or val < 1:
                    errors.append("s4.anchor_every_n must be a positive integer")
            
            # max_anchors
            if "max_anchors" in s4:
                val = s4["max_anchors"]
                if not isinstance(val, int) or val < 1:
                    errors.append("s4.max_anchors must be a positive integer")
            
            # comparison_window
            if "comparison_window" in s4:
                val = s4["comparison_window"]
                if not isinstance(val, int) or val < 1:
                    errors.append("s4.comparison_window must be a positive integer")
    
    # Validate S5 config
    if "s5" in config:
        s5 = config["s5"]
        if not isinstance(s5, dict):
            errors.append("'s5' config must be a dictionary")
    
    # Validate S6 config
    if "s6" in config:
        s6 = config["s6"]
        if not isinstance(s6, dict):
            errors.append("'s6' config must be a dictionary")
    
    return errors


def validate_paths(config: Dict[str, Any]) -> List[str]:
    """
    Validate that paths in config exist.
    
    Returns list of error messages.
    """
    from pathlib import Path
    
    errors = []
    
    # Check geometry_evidence_table
    if "s2" in config and "geometry_evidence_table" in config["s2"]:
        path = Path(config["s2"]["geometry_evidence_table"])
        if not path.exists():
            errors.append(f"s2.geometry_evidence_table not found: {path}")
    
    # Check texture_leaderboard
    if "s2" in config and "texture_leaderboard" in config["s2"]:
        path = Path(config["s2"]["texture_leaderboard"])
        if not path.exists():
            errors.append(f"s2.texture_leaderboard not found: {path}")
    
    return errors
