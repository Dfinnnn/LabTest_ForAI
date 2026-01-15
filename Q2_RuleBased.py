import streamlit as st
import json
from typing import List, Dict, Any, Tuple

# Page Configuration
st.set_page_config(page_title="Rule-Based AC Controller", layout="centered")
st.title("Question 2: Smart AC Controller")
st.markdown("### Rule-Based Logic Engine (Based on Table 1)")

# --- 1. INPUT SECTION ---
st.header("1. Environmental Conditions")
col1, col2 = st.columns(2)

with col1:
    # Inputs matching the scenario variables
    temperature = st.number_input("Temperature (°C)", min_value=16.0, max_value=40.0, value=22.0, step=0.5)
    humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=46)
    windows_open = st.checkbox("Are Windows Open?", value=False)

with col2:
    occupancy = st.selectbox("Occupancy Status", ["OCCUPIED", "EMPTY"])
    time_of_day = st.selectbox("Time of Day", ["MORNING", "AFTERNOON", "EVENING", "NIGHT"], index=3)

# --- 2. RULE ENGINE (The Brain) ---
def get_ac_decision(temp, hum, occ, time, windows):
    # We evaluate rules strictly by PRIORITY (Highest to Lowest)
    
    # ---------------------------------------------------------
    # Rule 1: Windows open -> turn AC off (Priority 100)
    # Condition: windows_open == True
    # ---------------------------------------------------------
    if windows:
        return "OFF", "LOW", "-", "Windows are open (Priority 100)"
    
    # ---------------------------------------------------------
    # Rule 2: No one home -> eco mode (Priority 90)
    # Condition: occupancy == EMPTY AND temperature >= 24
    # ---------------------------------------------------------
    if occ == "EMPTY" and temp >= 24:
        return "ECO", "LOW", "27°C", "Home empty; save energy (Priority 90)"
    
    # ---------------------------------------------------------
    # Rule 7: Too cold -> turn off (Priority 85)
    # Condition: temperature <= 22
    # ---------------------------------------------------------
    if temp <= 22:
        return "OFF", "LOW", "-", "Already cold (Priority 85)"

    # ---------------------------------------------------------
    # Rule 3: Hot & humid (occupied) -> cool strong (Priority 80)
    # Condition: occupancy == OCCUPIED AND temp >= 30 AND hum >= 70
    # ---------------------------------------------------------
    if occ == "OCCUPIED" and temp >= 30 and hum >= 70:
        return "COOL", "HIGH", "23°C", "Hot and humid (Priority 80)"
        
    # ---------------------------------------------------------
    # Rule 6: Night (occupied) -> sleep mode (Priority 75)
    # Condition: occupancy == OCCUPIED AND time == NIGHT AND temp >= 26
    # ---------------------------------------------------------
    if occ == "OCCUPIED" and time == "NIGHT" and temp >= 26:
        return "SLEEP", "LOW", "26°C", "Night comfort (Priority 75)"
        
    # ---------------------------------------------------------
    # Rule 4: Hot (occupied) -> cool (Priority 70)
    # Condition: occupancy == OCCUPIED AND temp >= 28
    # ---------------------------------------------------------
    if occ == "OCCUPIED" and temp >= 28:
        return "COOL", "MEDIUM", "24°C", "Temperature high (Priority 70)"
        
    # ---------------------------------------------------------
    # Rule 5: Slightly warm (occupied) -> gentle cool (Priority 60)
    # Condition: occupancy == OCCUPIED AND 26 <= temp < 28
    # ---------------------------------------------------------
    if occ == "OCCUPIED" and 26 <= temp < 28:
        return "COOL", "LOW", "25°C", "Slightly warm (Priority 60)"
        
    # Default State (No rule matched)
    return "IDLE", "LOW", "-", "No specific rule matched (Comfort zone)"

# --- 3. EXECUTION & DISPLAY ---
if st.button("Determine Settings", type="primary"):
    mode, fan, setpoint, reason = get_ac_decision(temperature, humidity, occupancy, time_of_day, windows_open)
    
    st.divider()
    st.subheader("Decision Result")
    
    # Visual Cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("AC Mode", mode)
    with c2:
        st.metric("Fan Speed", fan)
    with c3:
        st.metric("Setpoint", setpoint)
        
    st.success(f"**Reason:** {reason}")
    
    # Debug info to prove conditions to examiner
    with st.expander("See matched conditions"):
        st.write(f"Temp: {temperature}°C | Humidity: {humidity}%")
        st.write(f"Occupancy: {occupancy} | Time: {time_of_day}")
        st.write(f"Windows: {windows_open}")