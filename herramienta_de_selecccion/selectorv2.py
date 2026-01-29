from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import rasterio
import numpy as np
import pandas as pd
import streamlit as st
import geopandas as gpd
import json
from shapely.geometry import Point
import folium
from streamlit_folium import st_folium

# ---------------------------
# Configuration & Paths
# ---------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "solar_data"
SUBTEL_DIR = BASE_DIR / "Subtel"
BOARD_FILE = BASE_DIR / "boards.json"

# ---------------------------
# Domain models
# ---------------------------
@dataclass
class PowerProfile:
    sleep_uA: float                 
    sample_mA: float                
    sample_ms: int                  
    tx_mA: float                    
    tx_s: float                     
    tx_interval_min: int            
    system_overhead_mW: float = 1.0 

@dataclass
class Board:
    key: str
    name: str
    sensors: List[str]
    comms: List[str]                
    solar_ok: bool                  
    non_solar_ok: bool              
    battery_chemistries: List[str]  
    max_panel_Wp: float             
    notes: str = ""
    power: PowerProfile = field(default_factory=lambda: PowerProfile(0,0,0,0,0,0))
    github: str = ""

# ---------------------------
# Board registry (load/save JSON)
# ---------------------------
def to_bool(value) -> bool:
    """Helper to safely parse booleans from JSON."""
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)

def load_boards() -> list[Board]:
    """Load boards from the external JSON file."""
    if not BOARD_FILE.exists():
        return []
        
    try:
        raw = json.loads(BOARD_FILE.read_text(encoding='utf-8'))
        boards = []
        for b in raw:
            boards.append(Board(
                key=b["key"], name=b["name"],
                sensors=b["sensors"], comms=b["comms"],
                solar_ok=to_bool(b["solar_ok"]),
                non_solar_ok=to_bool(b["non_solar_ok"]),
                battery_chemistries=b["battery_chemistries"],
                max_panel_Wp=b["max_panel_Wp"],
                notes=b.get("notes",""),
                power=PowerProfile(**b["power"]),
                github=b.get("github", "")
            ))
        return boards
    except Exception as e:
        st.error(f"Error reading boards.json: {e}")
        return []

def save_boards(boards: list[Board]):
    raw = []
    for b in boards:
        raw.append({
            "key": b.key, "name": b.name,
            "sensors": b.sensors, "comms": b.comms,
            "solar_ok": b.solar_ok,
            "non_solar_ok": b.non_solar_ok,
            "battery_chemistries": b.battery_chemistries,
            "max_panel_Wp": b.max_panel_Wp,
            "notes": b.notes,
            "github": b.github,
            "power": b.power.__dict__
        })
    BOARD_FILE.write_text(json.dumps(raw, indent=2), encoding='utf-8')

REGISTRY = load_boards()

# ---------------------------
# Physics / Energy logic (Optimized Winter Model)
# ---------------------------
def daily_energy_mWh(power: PowerProfile, samples_per_hour:int=1) -> float:
    sleep_mA = power.sleep_uA / 1000.0
    sleep_mWh = sleep_mA * 3.3 * 24.0  

    sense_events = samples_per_hour * 24
    sense_mAh = (power.sample_mA * (power.sample_ms/1000.0)) * sense_events / 3600.0
    sense_mWh = sense_mAh * 3.3

    tx_events = max(1, int(24*60 / power.tx_interval_min))
    tx_mAh = (power.tx_mA * power.tx_s) * tx_events / 3600.0
    tx_mWh = tx_mAh * 3.8  

    overhead_mWh = power.system_overhead_mW * 24

    total = sleep_mWh + sense_mWh + tx_mWh + overhead_mWh
    return total

def calculate_winter_derate(latitude: float) -> float:

    lat_rad = math.radians(abs(latitude))
    declination_rad = math.radians(23.45) 
    
    avg_intensity = math.cos(lat_rad)
    winter_intensity = math.cos(lat_rad + declination_rad)
    
    if avg_intensity == 0: return 0.0
    
    ratio = winter_intensity / avg_intensity
    return max(0.0, ratio)

def solar_yield_mWh_per_day(GHI_kWh_m2_day: float,
                            panel_Wp: float,
                            shade: str,
                            latitude: float,
                            temp_C: float=25.0,
                            tilt_factor: float=0.9,
                            wiring_derate: float=0.96,
                            ctrl_derate: float=0.95) -> dict:

    shade_map = {
        "Full sun": 1.0,
        "Partial shade": 0.6,
        "Under canopy": 0.35,
    }
    shade_k = shade_map.get(shade, 0.8)

    # Temperature coefficient for generic poly/mono crystalline Silicon
    temp_derate_avg = 1.0 - max(0.0, (temp_C - 25.0)) * 0.004
    temp_derate_winter = 1.0 - max(0.0, (10.0 - 25.0)) * 0.004 

    # Base Yield (Annual Average)
    base_avg_wh = panel_Wp * GHI_kWh_m2_day * shade_k * tilt_factor * wiring_derate * ctrl_derate * temp_derate_avg

    # Winter Scenario Calculation
    # 1. Geometric reduction (Sun angle)
    winter_geo_factor = calculate_winter_derate(latitude)
    
    # 2. Weather penalty (Cloud cover heuristic for Chile)
    # North (-18) is clear, South (-40+) is cloudy in winter.
    weather_penalty = 1.0
    if latitude < -30:
        weather_penalty = max(0.4, 1.0 - (abs(latitude) - 30) * 0.03)

    winter_wh = (panel_Wp * GHI_kWh_m2_day) * winter_geo_factor * weather_penalty
    winter_wh = winter_wh * shade_k * tilt_factor * wiring_derate * ctrl_derate * temp_derate_winter
    
    return {
        "avg_mWh": max(0.0, base_avg_wh * 1000.0),
        "winter_mWh": max(0.0, winter_wh * 1000.0)
    }

# ---------------------------
# Solar data (Global Solar Atlas rasters)
# ---------------------------
def sample_safe(src, lon, lat):
    try:
        row, col = src.index(lon, lat)
        if (0 <= row < src.height) and (0 <= col < src.width):
            val = src.read(1)[row, col]
            if np.isnan(val) or val == src.nodata:
                return None
            return float(val)
    except Exception as e:
        print(f"[WARN] Sampling failed: {e}")
    return None

def get_solar_GHI_GSA(lat: float, lon: float) -> tuple[float, float]:
    ghi_path  = DATA_DIR / "GHI.asc"
    temp_path = DATA_DIR / "TEMP.asc"
    
    ghi_val, temp_val = 5.0, 15.0 # Defaults

    if ghi_path.exists():
        with rasterio.open(ghi_path) as ghi_src:
            val = sample_safe(ghi_src, lon, lat)
            if val is not None and val >= 0:
                ghi_val = val
                
    if temp_path.exists():
        with rasterio.open(temp_path) as t_src:
            val = sample_safe(t_src, lon, lat)
            if val is not None and val > -99:
                temp_val = val

    return ghi_val, temp_val

# ---------------------------
# Coverage model (Subtel towers)
# ---------------------------
@st.cache_data(show_spinner=False)
def _load_subtel_layer_cached(operator: str, tech: str) -> gpd.GeoDataFrame | None:
    bases = [
        SUBTEL_DIR / f"{operator}_{tech}_marzo2025.parquet",
        SUBTEL_DIR / f"{operator}_{tech}_marzo2025.geojson",
        SUBTEL_DIR / f"{operator}_{tech}_marzo2025.json",
    ]
    
    for p in bases:
        if p.exists():
            try:
                if p.suffix == ".parquet":
                    gdf = gpd.read_parquet(p)
                else:
                    gdf = gpd.read_file(p)
                
                if gdf.crs is None:
                    gdf = gdf.set_crs("EPSG:4326")
                if gdf.crs != "EPSG:3857":
                    gdf = gdf.to_crs("EPSG:3857")
                

                _ = gdf.sindex
                return gdf
            except Exception as e:
                print(f"Error loading {p}: {e}")
                
    return None

def _nearest_tower_m(lon: float, lat: float, operator: str, tech: str) -> float | None:
    gdf = _load_subtel_layer_cached(operator, tech)
    if gdf is None or gdf.empty:
        return None
    
    pt_gdf = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs("EPSG:3857")
    pt_geom = pt_gdf.iloc[0]
    
    try:
        nearest_indices = gdf.sindex.nearest(pt_geom, return_all=False)
        idx = nearest_indices[1][0] 
        nearest_tower = gdf.geometry.iloc[idx]
        return float(nearest_tower.distance(pt_geom))
    except Exception:
        return float(gdf.geometry.distance(pt_geom).min())

def get_cell_coverage_subtel(lat: float, lon: float, operator: str | None = None) -> dict[str, bool]:
    lon, lat = float(lon), float(lat)
    R4G = 15_000   
    RNB = 25_000   

    cov = {"4G": False, "LTE-M": False, "NB-IoT": False}
    ops_to_check = [operator] if operator else ["Entel", "Movistar", "Claro", "Wom"]
    
    found_4g = False
    for op in ops_to_check:
        d = _nearest_tower_m(lon, lat, op, "4G")
        if (d is not None) and (d < R4G):
            found_4g = True
            break 
            
    cov["4G"] = found_4g
    cov["LTE-M"] = found_4g 

    d_entel = _nearest_tower_m(lon, lat, "Entel", "4G")
    cov["NB-IoT"] = (d_entel is not None) and (d_entel < RNB)
    
    return cov

def get_lora_gateway_hint_stub(lat: float, lon: float) -> bool:
    return True

# ---------------------------
# Recommendation engine
# ---------------------------
def score_board(board: Board,
                ghi: float,
                tempC: float,
                shade: str,
                lat: float,
                comms_required: List[str],
                coverage: Dict[str,bool],
                samples_per_hour: int,
                battery_only: bool) -> Tuple[bool, float, Dict]:

    base = {"need_mWh_day": 0.0, "solar_avg_mWh": 0.0, "solar_winter_mWh": 0.0, "ok": False}
    
    for c in comms_required:
        if c not in board.comms:
            base["reason"] = f"Board lacks required comms: {c}"
            return (False, 0.0, base)

        if c in ("Cellular", "LTE-M", "NB-IoT", "4G"):
            req_cov = c if c in coverage else "4G"
            if not coverage.get(req_cov, False):
                return (False, 0.0, {"reason": f"No {c} coverage at site"})

    if battery_only:
        if not board.non_solar_ok:
            return (False, 0.0, {"reason": "Not validated for battery-only"})
        need_mWh = daily_energy_mWh(board.power, samples_per_hour=samples_per_hour)
        ok = need_mWh <= 150.0  
        solar_res = {"avg_mWh": 0, "winter_mWh": 0}

    else:
        need_mWh = daily_energy_mWh(board.power, samples_per_hour=samples_per_hour)
        solar_res = solar_yield_mWh_per_day(ghi, board.max_panel_Wp, shade, lat, tempC)
        
        ok_winter = solar_res["winter_mWh"] >= (need_mWh * 1.1)
        ok_avg    = solar_res["avg_mWh"] >= (need_mWh * 1.25)
        
        ok = ok_winter and ok_avg

    score = 0.0
    if ok:
        score += 60.0
        if not battery_only:
            headroom = max(0.0, min(40.0, (solar_res["winter_mWh"] - need_mWh) / max(need_mWh,1e-6) * 40.0))
            score += headroom
        else:
            score += max(0.0, min(40.0, (150.0 - need_mWh) / 150.0 * 40.0))
    else:
        val = solar_res["winter_mWh"] if not battery_only else 0
        ratio = (val / need_mWh) if need_mWh > 0 else 0
        score += max(0.0, min(30.0, ratio * 30.0))

    details = {
        "need_mWh_day": round(need_mWh, 1),
        "solar_avg_mWh": round(solar_res["avg_mWh"], 1),
        "solar_winter_mWh": round(solar_res["winter_mWh"], 1),
        "ok": ok,
        "coverage": coverage,
        "mode": "Battery" if battery_only else "Solar"
    }
    
    if not ok and not battery_only and solar_res["avg_mWh"] > need_mWh:
        details["reason"] = "Fails Winter Solstice check (IEEE 1562)."
        
    return (ok, round(score,1), details)

def recommend(boards: List[Board],
              lat: float, lon: float,
              shade: str,
              comms_required: List[str],
              samples_per_hour: int,
              battery_only: bool,
              operator: Optional[str]=None) -> pd.DataFrame:

    ghi, tempC = get_solar_GHI_GSA(lat, lon)
    
    st.markdown(
        f"<small>☀️ **GHI (Avg):** {ghi:.2f} kWh/m²/day  🌡️ **Temp:** {tempC:.1f} °C</small>",
        unsafe_allow_html=True,
    )

    cov = get_cell_coverage_subtel(lat, lon, operator)
    lora_ok = get_lora_gateway_hint_stub(lat, lon)
    
    st.caption(f"Coverage check → 4G: {cov['4G']} | LTE-M: {cov['LTE-M']} | NB-IoT: {cov['NB-IoT']}")

    rows = []
    for b in boards:

        is_battery_scoring = battery_only
        if not battery_only and not b.solar_ok and b.non_solar_ok:
            is_battery_scoring = True

        ok, score, det = score_board(
            b, ghi, tempC, shade, lat, comms_required, cov, samples_per_hour, is_battery_scoring
        )
        if "LoRa" in comms_required and not lora_ok:
            score *= 0.7
            det["reason_penalty"] = "LoRa feasibility uncertain"

        rows.append({
            "board_key": b.key,
            "board": b.name,
            "sensors": ", ".join(b.sensors),
            "comms": ", ".join(b.comms),
            "solar_ok": b.solar_ok,
            "non_solar_ok": b.non_solar_ok,
            "need_mWh_day": det.get("need_mWh_day", None),
            "solar_winter_mWh": det.get("solar_winter_mWh", None),
            "solar_avg_mWh": det.get("solar_avg_mWh", None),
            "coverage_4G": cov.get("4G", False),
            "coverage_LTE_M": cov.get("LTE-M", False),
            "coverage_NB_IoT": cov.get("NB-IoT", False),
            "ok": det.get("ok", False),
            "score": score,
            "notes": b.notes,
            "reason": det.get("reason", ""),
            "mode_used": det.get("mode", "Solar")
        })

    df = pd.DataFrame(rows)
    if not df.empty and "score" in df.columns:
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
    else:
        st.warning("No compatible boards found for the selected conditions.")
        df = pd.DataFrame()
    return df

# ---------------------------
# Map display
# ---------------------------
def show_site_map(lat: float, lon: float, operator="Entel", radius_m=15000):
    m = folium.Map(location=[lat, lon], zoom_start=10, control_scale=True)
    
    folium.Marker([lat, lon], tooltip="Selected site", icon=folium.Icon(color="blue")).add_to(m)
    folium.Circle(
        location=[lat, lon],
        radius=radius_m,
        color="#3186cc",
        fill=True,
        fill_color="#3186cc",
        fill_opacity=0.05,
        weight=1,
        tooltip="15km Search Radius"
    ).add_to(m)

    gdf = _load_subtel_layer_cached(operator, "4G")
    
    if gdf is not None and not gdf.empty:
        pt_m = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
        
        bbox = pt_m.buffer(radius_m).bounds
        possible_matches_index = list(gdf.sindex.intersection(bbox))
        possible_matches = gdf.iloc[possible_matches_index]
        
        nearby = possible_matches[possible_matches.geometry.distance(pt_m) < radius_m].copy()
        
        nearby["_dist"] = nearby.geometry.distance(pt_m)
        nearby = nearby.sort_values("_dist")

        LIMIT = 500
        if len(nearby) > LIMIT:
            st.caption(f"⚠️ High density area: Showing nearest {LIMIT} towers (out of {len(nearby)} found).")
            nearby = nearby.head(LIMIT)
            
        nearby_4326 = nearby.to_crs("EPSG:4326")
        
        if not nearby_4326.empty:
            for _, r in nearby_4326.iterrows():
                y, x = r.geometry.y, r.geometry.x
                folium.CircleMarker(
                    [y, x], 
                    radius=4, 
                    color="red", 
                    fill=True, 
                    fill_color="red",
                    tooltip=f"{operator} 4G Tower"
                ).add_to(m)

            bounds = nearby_4326.total_bounds # [minx, miny, maxx, maxy]
            min_lon, min_lat, max_lon, max_lat = bounds[0], bounds[1], bounds[2], bounds[3]
            
            min_lat = min(min_lat, lat)
            min_lon = min(min_lon, lon)
            max_lat = max(max_lat, lat)
            max_lon = max(max_lon, lon)
            
            lat_pad = (max_lat - min_lat) * 0.1 if max_lat != min_lat else 0.01
            lon_pad = (max_lon - min_lon) * 0.1 if max_lon != min_lon else 0.01
            
            m.fit_bounds([[min_lat - lat_pad, min_lon - lon_pad], [max_lat + lat_pad, max_lon + lon_pad]])


    st_folium(m, width=700, height=500, key=f"map_{lat}_{lon}_{operator}")
# ---------------------------
# Streamlit UI
# ---------------------------
tab1, tab2 = st.tabs(["📊 Recommender", "⚙️ Board Manager"])
with tab1:
    st.set_page_config(page_title="Weather Station Recommender (Chile)", page_icon="⛅", layout="wide")
    st.title("⛅ Weather Station Recommender — Chile")

    
    with st.sidebar:
        st.header("Site & Preferences")
        colat, colon = st.columns(2)
        lat = colat.number_input("Latitude (−56…−17, Chile)", value=-33.45, min_value=-56.0, max_value=-17.0, step=0.01, format="%.4f")
        lon = colon.number_input("Longitude (−76…−66, Chile)", value=-70.67, min_value=-76.0, max_value=-66.0, step=0.01, format="%.4f")

        shade = st.selectbox("Shading", ["Full sun", "Partial shade", "Under canopy"], index=1)
        samples_per_hour = st.slider("Samples per hour", min_value=1, max_value=12, value=2, step=1)

        st.divider()
        st.subheader("Communications")
        comms = st.multiselect("Required links", ["LoRa", "Cellular", "LTE-M", "NB-IoT", "WiFi", "BLE"], default=["LoRa"])

        st.checkbox("Battery-only (no solar)", value=False, key="battery_only")
        operator = st.text_input("Preferred operator (optional)", placeholder="Entel / Movistar / Claro / WOM")


    if st.button("Recommend"):
        with st.spinner("Analyzing Winter Solstice data and Coverage..."):
            df = recommend(REGISTRY, lat, lon, shade, comms, samples_per_hour, st.session_state["battery_only"], operator)
            st.session_state["last_df"] = df
            st.session_state["last_inputs"] = (lat, lon, shade, comms, operator)

    if "last_df" in st.session_state:
        df = st.session_state["last_df"]
        lat, lon, shade, comms, operator = st.session_state.get("last_inputs", (None, None, "", [], ""))

        if df.empty:
            st.warning("No boards matched. Try reducing constraints.")
        else:
            st.subheader("Recommendation")
            top = df.iloc[0]
            ok_badge = "✅ OK" if top["ok"] else "⚠️ Unsafe / Fail"
            
            mode_display = f"({top['mode_used']} Mode)"
            st.markdown(f"**Top pick:** {top['board']} — {ok_badge} {mode_display}")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Load", f"{top['need_mWh_day']:.0f} mWh/day")
            
            if top['mode_used'] == "Battery":
                c2.metric("Solar (Avg)", "N/A (Battery)")
                c3.metric("Solar (Winter)", "N/A (Battery)")
            else:
                c2.metric("Solar (Avg)", f"{top['solar_avg_mWh']:.0f} mWh/day")
                c3.metric("Solar (Winter)", f"{top['solar_winter_mWh']:.0f} mWh/day", delta_color="normal")
            
            st.write(f"- **Score:** {top['score']}")
            st.write(f"- **Comms:** {top['comms']}   |   **Sensors:** {top['sensors']}")
            
            if top["reason"]:
                st.error(f"Issue: {top['reason']}")

            if isinstance(top["notes"], str) and top["notes"]:
                st.info(top["notes"])

            github_link = getattr(next((b for b in REGISTRY if b.name == top["board"]), None), "github", "")
            if github_link:
                st.markdown(f"[🔗 View design files]({github_link})")

            st.divider()
            st.subheader("All options (ranked)")
            st.dataframe(df[[
                "board","score","ok","mode_used","need_mWh_day","solar_winter_mWh","comms","reason"
            ]], use_container_width=True)

            show_site_map(lat, lon, operator if operator else "Entel")

    else:
        st.info("Set your site and click **Recommend**.")

with tab2:
    st.divider()
    st.header("⚙️ Board Manager")
    
    with st.expander("➕ Add new board design"):
        new_name  = st.text_input("Board name")
        new_key   = st.text_input("Unique key (ID)")
        new_github = st.text_input("GitHub / docs link")

        new_sensors = st.text_area("Sensors (comma-separated)", "temp,rh,pressure")
        new_comms   = st.multiselect("Comms", ["LoRa","Cellular","LTE-M","NB-IoT","WiFi","BLE"])
        solar_ok    = st.checkbox("Solar-compatible", True)
        non_solar   = st.checkbox("Battery-only OK", True)
        new_panel   = st.number_input("Max panel Wp", 0.0, 50.0, 3.0, step=0.5)
        new_notes   = st.text_area("Notes")

        st.markdown("### Power profile")
        c1,c2,c3 = st.columns(3)
        sleep_uA = c1.number_input("Sleep µA", 0.0, 1000.0, 2.0)
        sample_mA = c2.number_input("Sample mA", 0.0, 100.0, 2.0)
        sample_ms = c3.number_input("Sample ms", 1, 10000, 80)
        tx_mA = c1.number_input("TX mA", 0.0, 1000.0, 45.0)
        tx_s  = c2.number_input("TX s", 0.0, 60.0, 1.2)
        tx_int = c3.number_input("TX interval min", 1, 1440, 15)
        sys_oh = c1.number_input("Overhead mW", 0.0, 10.0, 0.8)

        if st.button("➕ Add board"):
            new_board = Board(
                key=new_key,
                name=new_name,
                sensors=[s.strip() for s in new_sensors.split(",") if s.strip()],
                comms=new_comms,
                solar_ok=solar_ok,
                non_solar_ok=non_solar,
                battery_chemistries=["LiPo"],
                max_panel_Wp=new_panel,
                notes=new_notes,
                github=new_github,
                power=PowerProfile(
                    sleep_uA=sleep_uA, sample_mA=sample_mA,
                    sample_ms=sample_ms, tx_mA=tx_mA,
                    tx_s=tx_s, tx_interval_min=tx_int, system_overhead_mW=sys_oh
                )
            )
            REGISTRY.append(new_board)
            save_boards(REGISTRY)
            st.success(f"✅ Board **{new_name}** added.")