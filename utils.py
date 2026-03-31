"""
utils.py
Common utilities for Data Generation, Feature Engineering, and Inference for DairyVision AI.
"""
import math

def calculate_thi(temperature: float, humidity: float) -> float:
    """
    Calculate the Temperature-Humidity Index (THI).
    Formula: THI = 0.8 * T + (RH/100) * (T - 14.4) + 46.4
    where T is temperature in Celsius and RH is relative humidity in %.
    """
    return 0.8 * temperature + (humidity / 100.0) * (temperature - 14.4) + 46.4

def calculate_lactation_multiplier(dim: int) -> float:
    """
    Calculate lactation multiplier based on a simplified Wood's Curve.
    Peak milk yield typically occurs around 60-90 days in milk (DIM).
    """
    if dim <= 0:
        return 0.1
    # Wood's curve approximation structure: a * t^b * e^(-ct)
    # Normalized to return a multiplier around 0.5 to 1.3
    # Peak at approx 75 days
    multiplier = (dim ** 0.15) * math.exp(-0.002 * dim)
    # Scale to make peak roughly 1.3 and end (~305 days) roughly 0.6
    scaled_multiplier = multiplier * 0.72
    return max(0.1, min(scaled_multiplier, 1.5))

def calculate_heat_stress_penalty(thi: float) -> float:
    """
    Calculate milk yield penalty due to heat stress based on THI.
    Cows start experiencing heat stress around THI 68.
    """
    if thi < 68:
        return 1.0
    elif thi < 72:
        return 0.95  # Mild stress
    elif thi < 80:
        return 0.85  # Moderate stress
    elif thi < 90:
        return 0.70  # Severe stress
    else:
        return 0.50  # Dead/Critical stress
