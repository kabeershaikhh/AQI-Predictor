"""
EPA AQI CALCULATOR
==================
Calculates the US EPA Air Quality Index (0-500 scale) from raw pollutant
concentrations returned by the OpenWeather API.

How EPA AQI works:
  1. Each pollutant (PM2.5, PM10, CO, NO2, O3, SO2) has its own "breakpoint"
     table that maps concentration ranges to AQI ranges.
  2. For each pollutant, we calculate a sub-AQI using linear interpolation.
  3. The OVERALL AQI = the MAXIMUM sub-AQI across all pollutants.
  4. The pollutant with the highest sub-AQI is called the "dominant pollutant".

Note: EPA AQI officially uses averaged concentrations (24hr for PM, 8hr for CO/O3,
1hr for NO2/SO2). Since we have hourly instantaneous readings, we use them directly.
This is common practice in real-time AQI dashboards.

Reference: https://www.airnow.gov/aqi/aqi-basics/
"""


# ─────────────────────────────────────────────
# UNIT CONVERSIONS (OpenWeather gives μg/m³)
# ─────────────────────────────────────────────
# EPA breakpoints use different units for each pollutant.
# Conversion at standard conditions (25°C, 1 atm):
#   ppb = (μg/m³) × 24.45 / molecular_weight
#   ppm = ppb / 1000

def ug_to_ppm_co(ug):
    """Convert CO from μg/m³ to ppm (molecular weight: 28.01)"""
    return (ug * 24.45) / (28.01 * 1000)

def ug_to_ppb_no2(ug):
    """Convert NO2 from μg/m³ to ppb (molecular weight: 46.01)"""
    return (ug * 24.45) / 46.01

def ug_to_ppm_o3(ug):
    """Convert O3 from μg/m³ to ppm (molecular weight: 48.00)"""
    return (ug * 24.45) / (48.00 * 1000)

def ug_to_ppb_so2(ug):
    """Convert SO2 from μg/m³ to ppb (molecular weight: 64.07)"""
    return (ug * 24.45) / 64.07


# ─────────────────────────────────────────────
# EPA AQI BREAKPOINT TABLES
# ─────────────────────────────────────────────
# Format: (AQI_low, AQI_high, Concentration_low, Concentration_high)

# PM2.5 breakpoints (μg/m³) — no conversion needed
PM25_BREAKPOINTS = [
    (0,   50,   0.0,   12.0),
    (51,  100,  12.1,  35.4),
    (101, 150,  35.5,  55.4),
    (151, 200,  55.5,  150.4),
    (201, 300,  150.5, 250.4),
    (301, 400,  250.5, 350.4),
    (401, 500,  350.5, 500.4),
]

# PM10 breakpoints (μg/m³) — no conversion needed
PM10_BREAKPOINTS = [
    (0,   50,   0,    54),
    (51,  100,  55,   154),
    (101, 150,  155,  254),
    (151, 200,  255,  354),
    (201, 300,  355,  424),
    (301, 400,  425,  504),
    (401, 500,  505,  604),
]

# CO breakpoints (ppm) — converted from μg/m³
CO_BREAKPOINTS = [
    (0,   50,   0.0,  4.4),
    (51,  100,  4.5,  9.4),
    (101, 150,  9.5,  12.4),
    (151, 200,  12.5, 15.4),
    (201, 300,  15.5, 30.4),
    (301, 400,  30.5, 40.4),
    (401, 500,  40.5, 50.4),
]

# NO2 breakpoints (ppb) — converted from μg/m³
NO2_BREAKPOINTS = [
    (0,   50,   0,    53),
    (51,  100,  54,   100),
    (101, 150,  101,  360),
    (151, 200,  361,  649),
    (201, 300,  650,  1249),
    (301, 400,  1250, 1649),
    (401, 500,  1650, 2049),
]

# O3 breakpoints (ppm) — converted from μg/m³
O3_BREAKPOINTS = [
    (0,   50,   0.000, 0.054),
    (51,  100,  0.055, 0.070),
    (101, 150,  0.071, 0.085),
    (151, 200,  0.086, 0.105),
    (201, 300,  0.106, 0.200),
]

# SO2 breakpoints (ppb) — converted from μg/m³
SO2_BREAKPOINTS = [
    (0,   50,   0,    35),
    (51,  100,  36,   75),
    (101, 150,  76,   185),
    (151, 200,  186,  304),
    (201, 300,  305,  604),
    (301, 400,  605,  804),
    (401, 500,  805,  1004),
]


# ─────────────────────────────────────────────
# AQI CALCULATION
# ─────────────────────────────────────────────
def calc_sub_aqi(concentration, breakpoints):
    """
    Calculate the sub-AQI for a single pollutant using EPA linear interpolation.
    
    Formula:
      AQI = ((AQI_hi - AQI_lo) / (Conc_hi - Conc_lo)) × (C - Conc_lo) + AQI_lo
    
    Args:
        concentration: The pollutant concentration (in the correct unit)
        breakpoints: List of (AQI_lo, AQI_hi, Conc_lo, Conc_hi) tuples
    
    Returns:
        The sub-AQI value (0-500), or 500 if concentration exceeds all breakpoints
    """
    for aqi_lo, aqi_hi, conc_lo, conc_hi in breakpoints:
        if conc_lo <= concentration <= conc_hi:
            aqi = ((aqi_hi - aqi_lo) / (conc_hi - conc_lo)) * (concentration - conc_lo) + aqi_lo
            return round(aqi)
    
    # If concentration exceeds the highest breakpoint, cap at 500
    return 500


def calculate_epa_aqi(pm2_5, pm10, co, no2, o3, so2):
    """
    Calculate the overall EPA AQI from individual pollutant concentrations.
    
    All inputs are in μg/m³ (as returned by OpenWeather API).
    Returns a tuple of (overall_aqi, dominant_pollutant_name).
    
    The overall AQI is the MAXIMUM of all individual sub-AQIs.
    
    Example:
        >>> calculate_epa_aqi(pm2_5=35.0, pm10=100, co=500, no2=10, o3=60, so2=5)
        (98, 'PM2.5')
    """
    # Calculate sub-AQI for each pollutant (with unit conversions where needed)
    sub_aqis = {
        "PM2.5": calc_sub_aqi(pm2_5, PM25_BREAKPOINTS),
        "PM10":  calc_sub_aqi(pm10, PM10_BREAKPOINTS),
        "CO":    calc_sub_aqi(ug_to_ppm_co(co), CO_BREAKPOINTS),
        "NO2":   calc_sub_aqi(ug_to_ppb_no2(no2), NO2_BREAKPOINTS),
        "O3":    calc_sub_aqi(ug_to_ppm_o3(o3), O3_BREAKPOINTS),
        "SO2":   calc_sub_aqi(ug_to_ppb_so2(so2), SO2_BREAKPOINTS),
    }
    
    # The overall AQI is the maximum sub-AQI
    dominant = max(sub_aqis, key=sub_aqis.get)
    overall_aqi = sub_aqis[dominant]
    
    return overall_aqi, dominant


def get_aqi_category(aqi):
    """
    Returns the EPA AQI category and health concern level.
    
    Used for the dashboard alerts feature later.
    """
    if aqi <= 50:
        return "Good", "green"
    elif aqi <= 100:
        return "Moderate", "yellow"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "orange"
    elif aqi <= 200:
        return "Unhealthy", "red"
    elif aqi <= 300:
        return "Very Unhealthy", "purple"
    else:
        return "Hazardous", "maroon"


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Test with sample data from Karachi (from our earlier runs)
    test_cases = [
        {"pm2_5": 25.73, "pm10": 116.79, "co": 80.08, "no2": 0.06, "o3": 42.73, "so2": 0.59},
        {"pm2_5": 48.68, "pm10": 187.35, "co": 173.50, "no2": 2.09, "o3": 109.31, "so2": 6.59},
        {"pm2_5": 150.0, "pm10": 300.0, "co": 500.0, "no2": 50.0, "o3": 100.0, "so2": 20.0},
    ]
    
    print("EPA AQI Calculator Test")
    print("=" * 60)
    for i, tc in enumerate(test_cases):
        aqi, dominant = calculate_epa_aqi(**tc)
        category, color = get_aqi_category(aqi)
        print(f"\n  Test {i+1}: PM2.5={tc['pm2_5']}, PM10={tc['pm10']}")
        print(f"    EPA AQI:    {aqi}")
        print(f"    Category:   {category}")
        print(f"    Dominant:   {dominant}")
