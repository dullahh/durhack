import polars as pl
import streamlit as st
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd
import math
import os
import glob
from pathlib import Path

# ------------ SMART DATA LOADING BASED ON SCENARIO DATES ------------
def load_targeted_flight_data(availability_start: datetime, availability_end: datetime):
    """Load flight data specifically for the date range needed (±30 days)"""
    try:
        base_path = "/shared/challenge_data/schedules"
        
        if not os.path.exists(base_path):
            st.warning(f"Challenge data directory not found at {base_path}. Using sample data.")
            return generate_comprehensive_flight_data()
        
        # Calculate minimal search window (±2 days for travel buffer)
        search_start = availability_start - timedelta(days=2)
        search_end = availability_end + timedelta(days=2)
        
        # Target years for data loading (prioritize 2025-2026)
        target_years = [search_start.year, search_end.year]
        if search_start.year != search_end.year:
            target_years = list(range(search_start.year, search_end.year + 1))
        
        priority_years = [2025, 2026] + [year for year in target_years if year not in [2025, 2026]]
        
        st.info(f"🗓️ Loading targeted flight data for: {search_start.strftime('%Y-%m-%d')} to {search_end.strftime('%Y-%m-%d')}")
        st.info(f"📅 Prioritizing years: {priority_years}")
        
        # Find relevant CSV files
        csv_files = []
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    
                    # Extract year from path
                    path_parts = Path(file_path).parts
                    file_year = None
                    if len(path_parts) >= 3:
                        try:
                            file_year = int(path_parts[-3])
                        except (ValueError, IndexError):
                            continue
                    
                    # Only include files from priority years
                    if file_year in priority_years:
                        csv_files.append((file_path, file_year))
        
        if not csv_files:
            st.warning("No CSV files found for target years. Using sample data.")
            return generate_comprehensive_flight_data()
        
        # Sort files by relevance (prioritize 2025-2026)
        csv_files.sort(key=lambda x: (
            0 if x[1] in [2025, 2026] else 1,  # Prioritize 2025-2026
            abs(x[1] - search_start.year),      # Then by closeness to target year
            x[0]                                # Then by filename
        ))
        
        # Limit to 50 files for performance
        selected_files = [file_path for file_path, _ in csv_files[:50]]
        
        st.info(f"📂 Selected {len(selected_files)} files from relevant years")
        
        # Load files in smaller chunks
        chunk_size = 10
        combined_df = None
        
        progress_bar = st.progress(0)
        
        for chunk_start in range(0, len(selected_files), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(selected_files))
            chunk_dfs = []
            
            for i, file_path in enumerate(selected_files[chunk_start:chunk_end]):
                try:
                    progress_bar.progress((chunk_start + i + 1) / len(selected_files))
                    
                    df = pl.read_csv(
                        file_path,
                        infer_schema_length=10000,
                        schema_overrides={"ARRDAY": pl.Utf8},
                        null_values=["P"]
                    )
                    
                    # Add metadata
                    path_parts = Path(file_path).parts
                    if len(path_parts) >= 3:
                        year = path_parts[-3]
                        month = path_parts[-2]
                        day = path_parts[-1].split('.')[0]
                        df = df.with_columns(
                            pl.lit(f"{year}-{month}-{day}").alias("FILE_DATE"),
                            pl.lit(int(year)).alias("YEAR")
                        )
                    
                    chunk_dfs.append(df)
                    
                except Exception as e:
                    st.warning(f"Error loading {file_path}: {e}")
                    continue
            
            if chunk_dfs:
                chunk_combined = pl.concat(chunk_dfs, how="vertical_relaxed")
                if combined_df is None:
                    combined_df = chunk_combined
                else:
                    combined_df = pl.concat([combined_df, chunk_combined], how="vertical_relaxed")
                
                chunk_dfs.clear()
        
        progress_bar.empty()
        
        if combined_df is None:
            st.warning("No valid flight data could be loaded. Using sample data.")
            return generate_comprehensive_flight_data()
        
        st.success(f"✅ Loaded {combined_df.shape[0]:,} flights from {len(selected_files)} targeted files")
        
        # Add datetime columns
        result = add_datetime_cols(combined_df)
        
        # Filter data to target date range
        if "DEP_UTC" in result.columns:
            before_filter = result.height
            result = result.filter(
                (pl.col("DEP_UTC") >= pl.lit(search_start)) & 
                (pl.col("DEP_UTC") <= pl.lit(search_end))
            )
            after_filter = result.height
            st.info(f"🎯 Filtered to {after_filter:,} flights within target window (was {before_filter:,})")
        
        # Show airport coverage
        if "DEPAPT" in result.columns:
            unique_airports = result.select("DEPAPT").unique().to_pandas()["DEPAPT"].tolist()
            st.info(f"🛫 Available airports: {len(unique_airports)} ({', '.join(unique_airports[:15])}{'...' if len(unique_airports) > 15 else ''})")
        
        del combined_df
        return result
        
    except Exception as e:
        st.error(f"Error loading targeted flight data: {str(e)}")
        return generate_comprehensive_flight_data()

# ------------ EXPANDED MAPPINGS WITH ALL OFFICE LOCATIONS ------------
CITY_TO_IATA = {
    "London": ["LHR", "LGW", "LTN", "STN", "LCY"],
    "Paris": ["CDG", "ORY", "BVA"],
    "New York": ["JFK", "EWR", "LGA"],
    "Sydney": ["SYD"],
    "Mumbai": ["BOM"],
    "Shanghai": ["PVG", "SHA"],
    "Hong Kong": ["HKG"],
    "Singapore": ["SIN"],
    "Tokyo": ["NRT", "HND"],
    "Dubai": ["DXB", "DWC"],
    "Frankfurt": ["FRA"],
    "Amsterdam": ["AMS"],
    "Bangkok": ["BKK", "DMK"],
    "Seoul": ["ICN", "GMP"],
    "Los Angeles": ["LAX"],
    "Chicago": ["ORD", "MDW"],
    "Toronto": ["YYZ"],
    "Delhi": ["DEL"],
    "Beijing": ["PEK", "PKX"],
    # Additional office locations
    "Zurich": ["ZUR"],
    "Geneva": ["GVA"],
    "Aarhus": ["AAR"],
    "Wroclaw": ["WRO"],
    "Budapest": ["BUD"],
    # Additional major cities for potential meetings
    "Berlin": ["BER"],
    "Rome": ["FCO", "CIA"],
    "Madrid": ["MAD"],
    "Barcelona": ["BCN"],
    "Vienna": ["VIE"],
    "Warsaw": ["WAW"],
    "Prague": ["PRG"],
    "Stockholm": ["ARN"],
    "Copenhagen": ["CPH"],
    "Oslo": ["OSL"],
    "Helsinki": ["HEL"],
    "Brussels": ["BRU"],
    "Milan": ["MXP", "LIN"],
    "Istanbul": ["IST"],
    "Cairo": ["CAI"],
    "Johannesburg": ["JNB"],
    "Cape Town": ["CPT"],
    "Tel Aviv": ["TLV"],
    "Riyadh": ["RUH"],
    "Kuwait": ["KWI"],
    "Doha": ["DOH"],
    "Bahrain": ["BAH"],
    "Muscat": ["MCT"]
}

# Comprehensive city coordinates for visualization
CITY_COORDS = {
    "London": (51.5074, -0.1278),
    "Paris": (48.8566, 2.3522),
    "New York": (40.7128, -74.0060),
    "Sydney": (-33.8688, 151.2093),
    "Mumbai": (19.0760, 72.8777),
    "Shanghai": (31.2304, 121.4737),
    "Hong Kong": (22.3193, 114.1694),
    "Singapore": (1.3521, 103.8198),
    "Tokyo": (35.6762, 139.6503),
    "Dubai": (25.2048, 55.2708),
    "Frankfurt": (50.1109, 8.6821),
    "Amsterdam": (52.3676, 4.9041),
    "Bangkok": (13.7563, 100.5018),
    "Seoul": (37.5665, 126.9780),
    "Los Angeles": (34.0522, -118.2437),
    "Chicago": (41.8781, -87.6298),
    "Toronto": (43.6532, -79.3832),
    "Delhi": (28.7041, 77.1025),
    "Beijing": (39.9042, 116.4074),
    # Office locations
    "Zurich": (47.3769, 8.5417),
    "Geneva": (46.2044, 6.1432),
    "Aarhus": (56.1629, 10.2039),
    "Wroclaw": (51.1079, 17.0385),
    "Budapest": (47.4979, 19.0402),
    # Additional cities
    "Berlin": (52.5200, 13.4050),
    "Rome": (41.9028, 12.4964),
    "Madrid": (40.4168, -3.7038),
    "Barcelona": (41.3851, 2.1734),
    "Vienna": (48.2082, 16.3738),
    "Warsaw": (52.2297, 21.0122),
    "Prague": (50.0755, 14.4378),
    "Stockholm": (59.3293, 18.0686),
    "Copenhagen": (55.6761, 12.5683),
    "Oslo": (59.9139, 10.7522),
    "Helsinki": (60.1699, 24.9384),
    "Brussels": (50.8503, 4.3517),
    "Milan": (45.4642, 9.1900),
    "Istanbul": (41.0082, 28.9784),
    "Cairo": (30.0444, 31.2357),
    "Johannesburg": (-26.2041, 28.0473),
    "Cape Town": (-33.9249, 18.4241),
    "Tel Aviv": (32.0853, 34.7818),
    "Riyadh": (24.7136, 46.6753),
    "Kuwait": (29.3117, 47.4818),
    "Doha": (25.2854, 51.5310),
    "Bahrain": (26.0667, 50.5577),
    "Muscat": (23.5859, 58.4059)
}

# ------------ CRYSTAL BALL CONSTANTS ------------
# Tree offset calculations (conservative estimates)
TREE_CO2_CONSTANTS = {
    "co2_per_tree_per_year_kg": 22,  # Average kg CO2 absorbed per mature tree per year
    "tree_maturity_years": 10,       # Years for tree to reach full absorption capacity
    "tree_survival_rate": 0.8,       # Survival rate of planted trees
    "co2_per_tree_lifetime_kg": 220, # Total CO2 absorbed over tree's productive lifetime (10 years * 22kg)
    "tree_planting_cost_usd": 1.5,   # Average cost to plant one tree
    "hectare_trees": 1000,           # Trees per hectare for forest restoration
}

# Carbon offset alternatives
OFFSET_ALTERNATIVES = {
    "Solar Panel (1kW)": {"co2_offset_tonnes_year": 1.2, "cost_usd": 800},
    "Wind Turbine (small)": {"co2_offset_tonnes_year": 3.5, "cost_usd": 3000},
    "Carbon Credit": {"co2_offset_tonnes": 1.0, "cost_usd": 15},
    "Electric Car vs Gas": {"co2_offset_tonnes_year": 4.6, "cost_usd": 5000},
}

# ------------ HELPER FUNCTIONS ------------
def parse_iso_utc(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))

def calculate_distance(city1: str, city2: str) -> float:
    """Calculate great circle distance between two cities in km"""
    # Handle direct coordinate input in string format like "(lat,lon)"
    if city1.startswith("(") and city1.endswith(")"):
        lat1, lon1 = eval(city1)
    elif city1 not in CITY_COORDS:
        return 8000
    else:
        lat1, lon1 = CITY_COORDS[city1]
        
    if city2.startswith("(") and city2.endswith(")"):
        lat2, lon2 = eval(city2)
    elif city2 not in CITY_COORDS:
        return 8000
    else:
        lat2, lon2 = CITY_COORDS[city2]
    
    # Haversine formula
    R = 6371  # Earth's radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2) * math.sin(dlat/2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon/2) * math.sin(dlon/2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def estimate_flight_time(distance_km: float) -> float:
    """Estimate flight duration in hours based on distance"""
    # Account for takeoff/landing time and average speed
    if distance_km < 500:
        return max(1.0, distance_km / 400)  # Short haul, lower speed
    elif distance_km < 3000:
        return distance_km / 600  # Medium haul
    else:
        return distance_km / 800  # Long haul, higher speed

def estimate_co2(distance_km: float) -> float:
    """Estimate CO2 emissions in tonnes based on distance"""
    # kg CO2 per passenger-km, converted to tonnes
    return distance_km * 0.000115

def add_datetime_cols(df: pl.DataFrame) -> pl.DataFrame:
    """Add datetime columns to flight data with robust error handling"""
    try:
        # Ensure required columns exist
        required_cols = ["SCHEDULED_DEPARTURE_DATE_TIME_UTC", "SCHEDULED_ARRIVAL_DATE_TIME_UTC"]
        for col in required_cols:
            if col not in df.columns:
                st.warning(f"Missing column {col} in flight data")
                return df
        
        df = df.with_columns(
            pl.col("SCHEDULED_DEPARTURE_DATE_TIME_UTC").cast(pl.Utf8),
            pl.col("SCHEDULED_ARRIVAL_DATE_TIME_UTC").cast(pl.Utf8),
        )
        
        # Try multiple datetime formats
        formats_to_try = [
            "%Y-%m-%d %H:%M:%S%.f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S%.fZ",
            "%Y-%m-%dT%H:%M:%SZ", 
            "%Y-%m-%dT%H:%M:%S%.f",
            "%Y-%m-%dT%H:%M:%S",
            "%d/%m/%Y %H:%M",
            "%Y%m%d%H%M%S"
        ]
        
        for fmt in formats_to_try:
            try:
                df_with_datetime = df.with_columns(
                    pl.col("SCHEDULED_DEPARTURE_DATE_TIME_UTC")
                      .str.strptime(pl.Datetime, fmt, strict=False)
                      .dt.replace_time_zone("UTC")
                      .alias("DEP_UTC"),
                    pl.col("SCHEDULED_ARRIVAL_DATE_TIME_UTC")
                      .str.strptime(pl.Datetime, fmt, strict=False)
                      .dt.replace_time_zone("UTC")
                      .alias("ARR_UTC"),
                )
                
                valid_rows = df_with_datetime.filter(pl.col("DEP_UTC").is_not_null()).height
                if valid_rows > 0:
                    st.info(f"✅ Successfully parsed {valid_rows:,} flight records")
                    return df_with_datetime.drop(["SCHEDULED_DEPARTURE_DATE_TIME_UTC", "SCHEDULED_ARRIVAL_DATE_TIME_UTC"])
                    
            except Exception:
                continue
        
        st.warning("Could not parse datetime columns. Using null values.")
        return df.with_columns(
            pl.lit(None, dtype=pl.Datetime(time_zone="UTC")).alias("DEP_UTC"),
            pl.lit(None, dtype=pl.Datetime(time_zone="UTC")).alias("ARR_UTC"),
        )
        
    except Exception as e:
        st.warning(f"Error adding datetime columns: {e}")
        return df

def generate_comprehensive_flight_data():
    """Generate comprehensive sample flight data for 2025-2026"""
    import random
    
    cities = list(CITY_TO_IATA.keys())
    data = []
    
    st.info(f"Generating flight data for {len(cities)} cities...")
    
    # Generate flights for 2025-2026
    for year in [2025, 2026]:
        base_date = datetime(year, 1, 1)
        
        for i, city1 in enumerate(cities):
            for j, city2 in enumerate(cities):
                if city1 == city2:
                    continue
                    
                airports1 = CITY_TO_IATA[city1]
                airports2 = CITY_TO_IATA[city2]
                
                distance = calculate_distance(city1, city2)
                flight_time_hours = estimate_flight_time(distance)
                daily_flights = max(1, min(3, int(4000 / distance)))
                
                for day in range(0, 365, 7):  # Weekly flights
                    for flight_num in range(daily_flights):
                        for airport1 in airports1[:1]:
                            for airport2 in airports2[:1]:
                                
                                dep_time = base_date + timedelta(
                                    days=day,
                                    hours=random.randint(6, 22),
                                    minutes=random.randint(0, 59)
                                )
                                
                                arr_time = dep_time + timedelta(hours=flight_time_hours)
                                co2_estimate = estimate_co2(distance)
                                
                                data.append({
                                    "CARRIER": random.choice(["BA", "AF", "LH", "AA", "UA", "SQ", "CX", "EK"]),
                                    "FLTNO": f"{random.randint(1, 9999):04d}",
                                    "DEPAPT": airport1,
                                    "ARRAPT": airport2,
                                    "SCHEDULED_DEPARTURE_DATE_TIME_UTC": dep_time.strftime("%Y-%m-%d %H:%M:%S.000"),
                                    "SCHEDULED_ARRIVAL_DATE_TIME_UTC": arr_time.strftime("%Y-%m-%d %H:%M:%S.000"),
                                    "ELPTIM": int(flight_time_hours * 60),
                                    "DISTANCE": int(distance),
                                    "STOPS": 0,
                                    "YEAR": year,
                                    "ESTIMATED_CO2_TOTAL_TONNES": co2_estimate
                                })
    
    st.info(f"Generated {len(data)} sample flights for 2025-2026")
    return pl.DataFrame(data)

def slice_flights(
    df: pl.DataFrame,
    dep_iata: List[str],
    arr_iata: List[str],
    tmin: datetime,
    tmax: datetime,
    direct_only: bool = True,
    debug: bool = False
) -> pl.DataFrame:
    """Find flights with enhanced debugging and optimized filtering"""
    try:
        if "DEP_UTC" not in df.columns:
            if debug:
                st.warning("DEP_UTC column not found")
            return pl.DataFrame()
        
        # Filter by date range first - this will significantly reduce the dataset
        date_filtered = df.filter(
            (pl.col("DEP_UTC") >= pl.lit(tmin)) &
            (pl.col("DEP_UTC") <= pl.lit(tmax))
        )
        
        if debug:
            st.write(f"🔍 Searching: {dep_iata} → {arr_iata}")
            st.write(f"📅 Window: {tmin.strftime('%Y-%m-%d')} to {tmax.strftime('%Y-%m-%d')}")
            
            # Check airport coverage on reduced dataset
            available_deps = date_filtered.filter(pl.col("DEPAPT").is_in(dep_iata)).select("DEPAPT").unique().to_pandas()["DEPAPT"].tolist()
            available_arrs = date_filtered.filter(pl.col("ARRAPT").is_in(arr_iata)).select("ARRAPT").unique().to_pandas()["ARRAPT"].tolist()
            
            st.write(f"✈️ Found departure airports: {available_deps}")
            st.write(f"🛬 Found arrival airports: {available_arrs}")
        
        # Apply filters on pre-filtered dataset
        result = date_filtered.filter(
            pl.col("DEPAPT").is_in(dep_iata) &
            pl.col("ARRAPT").is_in(arr_iata) &
            pl.col("DEP_UTC").is_not_null()
        )
        
        if direct_only and "STOPS" in result.columns:
            result = result.filter(pl.col("STOPS") == 0)
        
        if debug:
            st.write(f"✅ Final result: {result.height} flights found")
        
        if result.height > 0:
            cols_to_select = ["CARRIER", "FLTNO", "DEPAPT", "ARRAPT", "DEP_UTC", "ARR_UTC"]
            optional_cols = ["ELPTIM", "DISTANCE", "STOPS", "ESTIMATED_CO2_TOTAL_TONNES"]
            
            for col in optional_cols:
                if col in result.columns:
                    cols_to_select.append(col)
            
            return result.select(cols_to_select).sort(["DEP_UTC"])
        
        return pl.DataFrame()
    
    except Exception as e:
        if debug:
            st.error(f"Error in slice_flights: {e}")
        return pl.DataFrame()

# ------------ ENHANCED FLIGHT BOOKING SYSTEM ------------
class FlightBookingSystem:
    """Enhanced flight booking with targeted data loading"""
    
    def __init__(self):
        self.flight_data = None
        
    def load_data_for_scenario(self, availability_start: datetime, availability_end: datetime):
        """Load flight data specifically for this scenario"""
        self.flight_data = load_targeted_flight_data(availability_start, availability_end)
        
    def find_flights_with_location_fallback(self, attendees: Dict, location_options: List[Dict], 
                                          availability_start: datetime, availability_end: datetime,
                                          event_duration: timedelta) -> Tuple[Dict, Dict]:
        """Find flights for attendees, trying different locations until flights are found"""
        
        # Load targeted data first
        if self.flight_data is None:
            st.info("🔄 Loading targeted flight data...")
            self.load_data_for_scenario(availability_start, availability_end)
        
        st.info("🔄 Searching for location with available flights...")
        
        for i, location_result in enumerate(location_options):
            destination = location_result['destination']
            event_start = location_result['event_start']
            event_end = location_result['event_end']
            
            st.text(f"Trying location {i+1}: {destination}")
            
            booking_results = self._find_flights_for_all_attendees(
                attendees, destination, event_start, event_end,
                availability_start, availability_end
            )
            
            successful_bookings = sum(1 for result in booking_results.values() 
                                    if result.get('status') in ['success', 'local'])
            total_offices = len(attendees)
            
            success_rate = successful_bookings / total_offices
            
            if success_rate >= 0.3:  # 30% success threshold
                st.success(f"✅ Found flights for {successful_bookings}/{total_offices} offices at {destination}")
                return location_result, booking_results
            else:
                st.warning(f"⚠️ Only {successful_bookings}/{total_offices} offices have flights at {destination}")
        
        # Use best location even with limited flights
        st.warning("🔄 Using best location even with limited flight availability")
        best_location = location_options[0]
        booking_results = self._find_flights_for_all_attendees(
            attendees, best_location['destination'], best_location['event_start'], 
            best_location['event_end'], availability_start, availability_end
        )
        
        return best_location, booking_results
    
    def _find_flights_for_all_attendees(self, attendees: Dict, destination: str, 
                                      event_start: datetime, event_end: datetime,
                                      availability_start: datetime, availability_end: datetime) -> Dict:
        """Find specific flights for all attendees"""
        
        booking_results = {}
        
        for origin_city, attendee_count in attendees.items():
            if origin_city == destination:
                booking_results[origin_city] = {
                    'status': 'local',
                    'attendees': attendee_count,
                    'message': 'No flights needed - local attendees'
                }
                continue
            
            flights = self._find_flexible_flights_for_city(
                origin_city, destination, event_start, event_end,
                availability_start, availability_end, attendee_count
            )
            
            booking_results[origin_city] = flights
            
        return booking_results
    
    def _find_flexible_flights_for_city(self, origin_city: str, dest_city: str,
                                       event_start: datetime, event_end: datetime,
                                       availability_start: datetime, availability_end: datetime,
                                       attendee_count: int) -> Dict:
        """Find flights with focused search on event dates"""
        
        origin_airports = CITY_TO_IATA.get(origin_city, [])
        dest_airports = CITY_TO_IATA.get(dest_city, [])
        
        if not origin_airports or not dest_airports:
            return {
                'status': 'error',
                'attendees': attendee_count,
                'message': f'No airport codes found for {origin_city} or {dest_city}'
            }
        
        # Tight search windows around event dates
        search_start = event_start - timedelta(days=1)  # 1 day before event
        search_end = event_end + timedelta(days=1)     # 1 day after event
        
        strategies = [
            {
                'name': 'Wide search window',
                'outbound_min': search_start,
                'outbound_max': search_end,
                'return_min': search_start,
                'return_max': search_end,
                'direct_only': True,
                'debug': True
            },
            {
                'name': 'Any flights with connections',
                'outbound_min': search_start,
                'outbound_max': search_end,
                'return_min': search_start,
                'return_max': search_end,
                'direct_only': False,
                'debug': False
            }
        ]
        
        for strategy in strategies:
            # Search for outbound and return flights
            outbound_flights = slice_flights(
                self.flight_data, origin_airports, dest_airports,
                strategy['outbound_min'], strategy['outbound_max'], 
                direct_only=strategy['direct_only'],
                debug=strategy.get('debug', False)
            )
            
            return_flights = slice_flights(
                self.flight_data, dest_airports, origin_airports,
                strategy['return_min'], strategy['return_max'], 
                direct_only=strategy['direct_only']
            )
            
            # Find any available combination
            best_outbound, best_return = self._select_any_flight_combination(
                outbound_flights, return_flights
            )
            
            if best_outbound is not None and best_return is not None:
                return self._format_flight_booking(
                    best_outbound, best_return, origin_city, dest_city, 
                    attendee_count, strategy['name']
                )
        
        return {
            'status': 'no_flights',
            'attendees': attendee_count,
            'message': f'No flights found for {origin_city} → {dest_city}',
            'search_period': f"{search_start.strftime('%Y-%m-%d')} to {search_end.strftime('%Y-%m-%d')}"
        }
    
    def _select_any_flight_combination(self, outbound_df: pl.DataFrame, 
                                     return_df: pl.DataFrame) -> Tuple:
        """Select any available flight combination"""
        
        if outbound_df.height == 0 or return_df.height == 0:
            return None, None
        
        try:
            outbound_pd = outbound_df.to_pandas()
            return_pd = return_df.to_pandas()
            
            if len(outbound_pd) > 0 and len(return_pd) > 0:
                # Take first available flights
                best_outbound = outbound_pd.iloc[0]
                best_return = return_pd.iloc[0]
                return best_outbound, best_return
            
            return None, None
            
        except Exception as e:
            st.warning(f"Error selecting flights: {e}")
            return None, None
    
    def _format_flight_booking(self, outbound_flight, return_flight, origin_city: str, 
                             dest_city: str, attendee_count: int, strategy_used: str = "") -> Dict:
        """Format flight booking information"""
        
        try:
            outbound_duration = outbound_flight.get('ELPTIM', 300)
            return_duration = return_flight.get('ELPTIM', 300)
            total_travel_time = (outbound_duration + return_duration) / 60
            
            base_cost = 500
            distance = outbound_flight.get('DISTANCE', 5000)
            cost_per_person = base_cost + (distance * 0.1)
            total_cost = cost_per_person * attendee_count
            
            return {
                'status': 'success',
                'attendees': attendee_count,
                'search_strategy': strategy_used,
                'outbound': {
                    'carrier': outbound_flight['CARRIER'],
                    'flight_number': outbound_flight['FLTNO'],
                    'departure_airport': outbound_flight['DEPAPT'],
                    'arrival_airport': outbound_flight['ARRAPT'],
                    'departure_time': outbound_flight['DEP_UTC'].strftime('%Y-%m-%d %H:%M UTC'),
                    'arrival_time': outbound_flight['ARR_UTC'].strftime('%Y-%m-%d %H:%M UTC'),
                    'duration_minutes': outbound_duration
                },
                'return': {
                    'carrier': return_flight['CARRIER'],
                    'flight_number': return_flight['FLTNO'],
                    'departure_airport': return_flight['DEPAPT'],
                    'arrival_airport': return_flight['ARRAPT'],
                    'departure_time': return_flight['DEP_UTC'].strftime('%Y-%m-%d %H:%M UTC'),
                    'arrival_time': return_flight['ARR_UTC'].strftime('%Y-%m-%d %H:%M UTC'),
                    'duration_minutes': return_duration
                },
                'totals': {
                    'total_travel_hours': round(total_travel_time, 1),
                    'cost_per_person_usd': round(cost_per_person, 0),
                    'total_cost_usd': round(total_cost, 0),
                    'route': f"{origin_city} → {dest_city} → {origin_city}"
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'attendees': attendee_count,
                'message': f'Error formatting flight information: {e}'
            }

# ------------ CRYSTAL BALL FUNCTIONS ------------
class CarbonCrystalBall:
    """Tree and CO2 Emission calculator for carbon offsets and tree planting requirements"""
    
    @staticmethod
    def calculate_trees_needed(co2_tonnes: float) -> Dict:
        """Calculate tree planting requirements to offset CO2"""
        try:
            co2_kg = co2_tonnes * 1000  # Convert to kg
            
            # Trees needed (accounting for survival rate)
            ideal_trees = co2_kg / TREE_CO2_CONSTANTS["co2_per_tree_lifetime_kg"]
            trees_accounting_survival = ideal_trees / TREE_CO2_CONSTANTS["tree_survival_rate"]
            
            # Costs and other metrics
            planting_cost = trees_accounting_survival * TREE_CO2_CONSTANTS["tree_planting_cost_usd"]
            hectares_needed = trees_accounting_survival / TREE_CO2_CONSTANTS["hectare_trees"]
            years_to_offset = TREE_CO2_CONSTANTS["tree_maturity_years"]
            
            # Annual absorption once mature
            annual_absorption_tonnes = (trees_accounting_survival * 
                                     TREE_CO2_CONSTANTS["tree_survival_rate"] * 
                                     TREE_CO2_CONSTANTS["co2_per_tree_per_year_kg"]) / 1000
            
            return {
                "trees_needed": int(trees_accounting_survival),
                "planting_cost_usd": planting_cost,
                "hectares_needed": hectares_needed,
                "years_to_full_offset": years_to_offset,
                "annual_absorption_tonnes": annual_absorption_tonnes,
                "co2_input_tonnes": co2_tonnes,
                "equivalent_cars_off_road": co2_tonnes / 4.6,  # Average car emissions per year
                "equivalent_household_energy": co2_tonnes / 6.8,  # Average household energy emissions per year
            }
        except Exception as e:
            st.error(f"Error in tree calculation: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def calculate_offset_alternatives(co2_tonnes: float) -> Dict:
        """Calculate alternative offset methods"""
        try:
            alternatives = {}
            for method, data in OFFSET_ALTERNATIVES.items():
                if "co2_offset_tonnes_year" in data:
                    # Annual offset capacity
                    units_needed = co2_tonnes / data["co2_offset_tonnes_year"]
                    cost = units_needed * data["cost_usd"]
                    time_years = 1  # Offset achieved in 1 year
                else:
                    # One-time offset
                    units_needed = co2_tonnes / data["co2_offset_tonnes"]
                    cost = units_needed * data["cost_usd"]
                    time_years = 0  # Immediate offset
                
                alternatives[method] = {
                    "units_needed": units_needed,
                    "total_cost_usd": cost,
                    "offset_time_years": time_years
                }
            
            return alternatives
        except Exception as e:
            return {"error": str(e)}

def create_crystal_ball_visualizations(tree_data: Dict, alternatives_data: Dict):
    """Create visualizations for the crystal ball section"""
    try:
        # Tree offset visualization
        fig_trees = go.Figure()
        
        # Tree planting timeline
        years = list(range(0, 21))  # 20 year timeline
        absorption = []
        for year in years:
            if year < tree_data["years_to_full_offset"]:
                # Growing phase - linear increase
                yearly_absorption = (year / tree_data["years_to_full_offset"]) * tree_data["annual_absorption_tonnes"]
            else:
                # Mature phase - full absorption
                yearly_absorption = tree_data["annual_absorption_tonnes"]
            absorption.append(yearly_absorption)
        
        fig_trees.add_trace(go.Scatter(
            x=years, 
            y=absorption,
            mode='lines+markers',
            name='Annual CO2 Absorption',
            line=dict(color='green', width=3),
            fill='tozeroy',
            fillcolor='rgba(0,255,0,0.1)'
        ))
        
        # Add horizontal line for target CO2
        fig_trees.add_hline(
            y=tree_data["co2_input_tonnes"], 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Target: {tree_data['co2_input_tonnes']:.1f} tonnes CO2"
        )
        
        fig_trees.update_layout(
            title="🌳 Tree Absorption Timeline",
            xaxis_title="Years After Planting",
            yaxis_title="Annual CO2 Absorption (tonnes)",
            height=400
        )
        
        # Cost comparison chart
        if "error" not in alternatives_data:
            methods = list(alternatives_data.keys()) + ["Tree Planting"]
            costs = [alternatives_data[method]["total_cost_usd"] for method in alternatives_data.keys()]
            costs.append(tree_data["planting_cost_usd"])
            
            fig_costs = go.Figure(data=[
                go.Bar(x=methods, y=costs, marker_color=['skyblue'] * len(methods))
            ])
            
            fig_costs.update_layout(
                title="💰 Offset Method Cost Comparison",
                xaxis_title="Offset Method",
                yaxis_title="Cost (USD)",
                height=400
            )
            
            return fig_trees, fig_costs
        
        return fig_trees, None
        
    except Exception as e:
        st.error(f"Error creating visualizations: {e}")
        return None, None

# ------------ LOCATION OPTIMIZER CLASSES ------------
class FlightOptimizer:
    def __init__(self, flight_data: pl.DataFrame):
        self.flight_data = flight_data
        
    def find_optimal_flights(self, origin_city: str, dest_city: str, 
                           event_start: datetime, event_end: datetime,
                           availability_start: datetime, availability_end: datetime) -> Dict:
        """Find optimal outbound and return flights with fallback mechanisms"""
        
        origin_airports = CITY_TO_IATA.get(origin_city, [])
        dest_airports = CITY_TO_IATA.get(dest_city, [])
        
        if not origin_airports or not dest_airports:
            # Fallback: estimate based on distance
            return self._create_fallback_plan(origin_city, dest_city, event_start, event_end)
        
        # Time windows with smaller buffers initially
        arrival_buffer = timedelta(hours=2)
        departure_buffer = timedelta(hours=1)
        
        # Try multiple search strategies
        for buffer_multiplier in [1, 2, 4]:  # Progressively relax constraints
            
            # Outbound flights: must arrive before event start
            outbound_max = event_start - arrival_buffer * buffer_multiplier
            outbound_min = availability_start - timedelta(days=14)
            
            outbound = slice_flights(
                self.flight_data, origin_airports, dest_airports,
                outbound_min, outbound_max, direct_only=True
            )
            
            # Return flights: must depart after event end  
            return_min = event_end + departure_buffer * buffer_multiplier
            return_max = availability_end + timedelta(days=14)
            
            inbound = slice_flights(
                self.flight_data, dest_airports, origin_airports,
                return_min, return_max, direct_only=True
            )
            
            # Find best combination
            best_combo = self._find_best_flight_combination(outbound, inbound)
            
            if best_combo is not None:
                return best_combo
            
            # Try allowing connecting flights
            if buffer_multiplier == 2:
                outbound = slice_flights(
                    self.flight_data, origin_airports, dest_airports,
                    outbound_min, outbound_max, direct_only=False
                )
                inbound = slice_flights(
                    self.flight_data, dest_airports, origin_airports,
                    return_min, return_max, direct_only=False
                )
                
                best_combo = self._find_best_flight_combination(outbound, inbound)
                if best_combo is not None:
                    return best_combo
        
        # Final fallback: create estimated flight plan
        return self._create_fallback_plan(origin_city, dest_city, event_start, event_end)
    
    def _find_best_flight_combination(self, outbound: pl.DataFrame, inbound: pl.DataFrame) -> Optional[Dict]:
        """Find the best outbound + inbound flight combination"""
        
        if outbound.height == 0 or inbound.height == 0:
            return None
        
        try:
            # Convert to pandas for easier manipulation
            out_df = outbound.to_pandas()
            in_df = inbound.to_pandas()
            
            best_combo = None
            min_total_time = float('inf')
            
            # Try top 3 outbound and inbound flights to avoid excessive computation
            for _, out_flight in out_df.head(3).iterrows():
                for _, in_flight in in_df.head(3).iterrows():
                    # Calculate total travel time
                    out_duration = out_flight.get('ELPTIM', 300)  # minutes, default if missing
                    in_duration = in_flight.get('ELPTIM', 300)   # minutes, default if missing
                    total_time = out_duration + in_duration
                    
                    if total_time < min_total_time:
                        min_total_time = total_time
                        
                        # Calculate CO2 
                        out_co2 = out_flight.get('ESTIMATED_CO2_TOTAL_TONNES', 
                                                out_flight.get('DISTANCE', 5000) * 0.000115)
                        in_co2 = in_flight.get('ESTIMATED_CO2_TOTAL_TONNES',
                                              in_flight.get('DISTANCE', 5000) * 0.000115)
                        
                        best_combo = {
                            'outbound_flight': out_flight.to_dict(),
                            'inbound_flight': in_flight.to_dict(),
                            'total_travel_hours': total_time / 60,
                            'total_co2_tonnes': out_co2 + in_co2,
                            'outbound_departure': out_flight['DEP_UTC'],
                            'inbound_arrival': in_flight['ARR_UTC'],
                            'is_estimated': False
                        }
            
            return best_combo
            
        except Exception as e:
            st.warning(f"Error in flight combination: {e}")
            return None
    
    def _create_fallback_plan(self, origin_city: str, dest_city: str, 
                             event_start: datetime, event_end: datetime) -> Dict:
        """Create fallback travel plan based on distance estimates"""
        
        distance = calculate_distance(origin_city, dest_city)
        travel_hours = estimate_flight_time(distance)
        co2_estimate = estimate_co2(distance)
        
        # Create synthetic flight plan
        outbound_dep = event_start - timedelta(hours=travel_hours + 3)
        outbound_arr = event_start - timedelta(hours=2)
        inbound_dep = event_end + timedelta(hours=2)
        inbound_arr = event_end + timedelta(hours=travel_hours + 3)
        
        return {
            'outbound_flight': {
                'CARRIER': 'EST',
                'FLTNO': '0000',
                'DEPAPT': origin_city[:3],
                'ARRAPT': dest_city[:3],
                'DEP_UTC': outbound_dep,
                'ARR_UTC': outbound_arr,
                'ELPTIM': int(travel_hours * 60),
                'DISTANCE': int(distance),
                'ESTIMATED_CO2_TOTAL_TONNES': co2_estimate
            },
            'inbound_flight': {
                'CARRIER': 'EST', 
                'FLTNO': '0001',
                'DEPAPT': dest_city[:3],
                'ARRAPT': origin_city[:3],
                'DEP_UTC': inbound_dep,
                'ARR_UTC': inbound_arr,
                'ELPTIM': int(travel_hours * 60),
                'DISTANCE': int(distance),
                'ESTIMATED_CO2_TOTAL_TONNES': co2_estimate
            },
            'total_travel_hours': travel_hours * 2,
            'total_co2_tonnes': co2_estimate * 2,
            'outbound_departure': outbound_dep,
            'inbound_arrival': inbound_arr,
            'is_estimated': True
        }

class LocationOptimizer:
    def __init__(self, flight_optimizer: FlightOptimizer):
        self.flight_optimizer = flight_optimizer
        
    def optimize_location(self, scenario: Dict, co2_weight: float = 0.5, 
                         fairness_weight: float = 0.5) -> List[Dict]:
        """Find optimal meeting locations balancing CO2 and fairness"""
        
        attendees = scenario["attendees"]
        start = parse_iso_utc(scenario["availability_window"]["start"])
        end = parse_iso_utc(scenario["availability_window"]["end"])
        duration = timedelta(
            days=scenario["event_duration"]["days"],
            hours=scenario["event_duration"]["hours"]
        )
        
        # Event timing - allow more flexibility for longer events
        event_start = start + timedelta(days=3)  # Give people time to arrive for global meeting
        event_end = event_start + duration
        
        # Ensure event fits in window
        if event_end > end - timedelta(days=2):
            event_start = start + timedelta(days=1)
            event_end = event_start + duration
        
        # Get candidate destinations - prioritize major hubs for large global meetings
        all_cities = set(CITY_TO_IATA.keys())
        attendee_cities = set(attendees.keys())
        
        # For large global meetings, prioritize major hub cities
        major_hubs = {"London", "Paris", "Frankfurt", "Amsterdam", "Dubai", "Singapore", 
                     "Hong Kong", "Tokyo", "New York", "Sydney", "Zurich", "Geneva"}
        
        # Prioritize hubs, then add other cities
        candidates = list((all_cities & major_hubs) - attendee_cities)
        candidates.extend(list((all_cities - major_hubs) - attendee_cities))
        candidates.extend(list(attendee_cities))  # Add attendee cities as potential hosts
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_attendees = sum(attendees.values())
        st.info(f"🌍 Optimizing for {len(attendee_cities)} offices with {total_attendees} total attendees")
        
        # Evaluate MORE candidates to have more fallback options
        max_candidates = min(40, len(candidates))  # Increased from 30 to 40
        
        for i, dest_city in enumerate(candidates[:max_candidates]):
            status_text.text(f"Evaluating {dest_city}... ({i+1}/{max_candidates})")
            progress_bar.progress((i + 1) / max_candidates)
            
            try:
                result = self._evaluate_destination(dest_city, attendees, event_start, event_end, 
                                                 start, end)
                if result and 'error' not in result:
                    results.append(result)
            except Exception as e:
                st.warning(f"Error evaluating {dest_city}: {e}")
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        if not results:
            # Create at least one result as absolute fallback
            fallback_city = max(attendees.items(), key=lambda x: x[1])[0]  # Use largest office
            results = [self._create_fallback_result(fallback_city, attendees, event_start, event_end)]
        
        # Normalize and score results
        results = self._normalize_and_score_results(results, co2_weight, fairness_weight)
        return sorted(results, key=lambda x: x['composite_score'])
    
    def _evaluate_destination(self, dest_city: str, attendees: Dict,
                            event_start: datetime, event_end: datetime,
                            availability_start: datetime, availability_end: datetime) -> Dict:
        """Evaluate a single destination city"""
        
        travel_plans = {}
        total_co2 = 0
        travel_hours = []
        attendee_travel_hours = {}
        has_estimated_flights = False
        
        # Find flights for each origin city
        for origin_city, count in attendees.items():
            if origin_city == dest_city:
                # Local attendees - no travel needed
                attendee_travel_hours[origin_city] = 0
                travel_hours.extend([0] * count)
                continue
            
            # Get flight plan for this origin city
            flight_plan = self.flight_optimizer.find_optimal_flights(
                origin_city, dest_city, event_start, event_end,
                availability_start, availability_end
            )
            
            travel_plans[origin_city] = flight_plan
            hours = flight_plan['total_travel_hours']
            co2 = flight_plan['total_co2_tonnes']
            
            # Track metrics
            if flight_plan.get('is_estimated', False):
                has_estimated_flights = True
            total_co2 += co2 * count
            travel_hours.extend([hours] * count)
            attendee_travel_hours[origin_city] = hours
        
        # Calculate basic metrics
        travel_hours_array = np.array(travel_hours)
        mean_travel = np.mean(travel_hours_array)
        std_travel = np.std(travel_hours_array)
        max_travel = np.max(travel_hours_array)
        min_travel = np.min(travel_hours_array)
        
        # Calculate weighted impact based on attendee numbers and distances
        total_impact = 0
        total_distance_impact = 0
        total_attendees = sum(attendees.values())
        
        # Calculate average location for attendees (weighted by number of attendees)
        weighted_lat = 0
        weighted_lon = 0
        for origin_city, count in attendees.items():
            if origin_city in CITY_COORDS:
                lat, lon = CITY_COORDS[origin_city]
                weighted_lat += lat * (count / total_attendees)
                weighted_lon += lon * (count / total_attendees)
        
        # Calculate distance from optimal center
        if dest_city in CITY_COORDS:
            dest_lat, dest_lon = CITY_COORDS[dest_city]
            center_distance = calculate_distance(
                str((weighted_lat, weighted_lon)),
                str((dest_lat, dest_lon))
            ) / 1000  # Convert to relative score
        else:
            center_distance = 5000  # Penalty for unknown cities
        
        for origin_city, count in attendees.items():
            if origin_city in attendee_travel_hours:
                attendee_weight = count / total_attendees
                total_impact += attendee_travel_hours[origin_city] * attendee_weight
                
                # Add distance-based impact
                if origin_city in CITY_COORDS and dest_city in CITY_COORDS:
                    distance = calculate_distance(origin_city, dest_city)
                    total_distance_impact += distance * attendee_weight
        
        # Calculate fairness score incorporating centrality and travel distribution
        # Lower raw_score is better, will be inverted later
        raw_score = (
            (total_impact * 0.3) +          # Total travel time impact
            (std_travel * 0.3) +            # Travel time variation
            (max_travel * 0.2) +            # Maximum travel time
            (center_distance * 0.2)         # Distance from optimal center
        )
        
        max_possible_score = 50
        fairness_score = max_possible_score - min(raw_score, max_possible_score)
        
        # Return evaluation results
        return {
            'destination': dest_city,
            'total_co2_tonnes': total_co2,
            'average_travel_hours': mean_travel,
            'median_travel_hours': np.median(travel_hours_array),
            'max_travel_hours': max_travel,
            'min_travel_hours': min_travel,
            'fairness_score': fairness_score,
            'attendee_travel_hours': attendee_travel_hours,
            'travel_plans': travel_plans,
            'event_start': event_start,
            'event_end': event_end,
            'has_estimated_flights': has_estimated_flights
        }
    
    def _create_fallback_result(self, city: str, attendees: Dict, 
                               event_start: datetime, event_end: datetime) -> Dict:
        """Create a fallback result when no destinations work"""
        
        total_attendees = sum(attendees.values())
        estimated_co2 = total_attendees * 2.5  # Rough estimate for global meeting
        
        return {
            'destination': city,
            'total_co2_tonnes': estimated_co2,
            'average_travel_hours': 8.0,
            'median_travel_hours': 6.0,
            'max_travel_hours': 15.0,
            'min_travel_hours': 0.0,
            'fairness_score': 4.0,
            'attendee_travel_hours': {c: 8.0 for c in attendees.keys()},
            'travel_plans': {},
            'event_start': event_start,
            'event_end': event_end,
            'has_estimated_flights': True
        }
    
    def _normalize_and_score_results(self, results: List[Dict], co2_weight: float, fairness_weight: float) -> List[Dict]:
        """Normalize scores and calculate composite scores with exponential penalty for higher travel times"""
        
        if not results:
            return results
        
        # Get key metrics for normalization
        co2_values = [r['total_co2_tonnes'] for r in results]
        avg_travel_values = [r['average_travel_hours'] for r in results]
        max_travel_values = [r['max_travel_hours'] for r in results]
        min_co2, max_co2 = min(co2_values), max(co2_values)
        min_avg_travel = min(avg_travel_values)
        
        for result in results:
            # Normalize CO2 (0 = best, 1 = worst)
            if max_co2 > min_co2:
                co2_score = (result['total_co2_tonnes'] - min_co2) / (max_co2 - min_co2)
            else:
                co2_score = 0
                
            # Calculate travel time penalty with exponential scaling
            # This will heavily penalize locations with much higher travel times
            avg_travel_penalty = ((result['average_travel_hours'] - min_avg_travel) / 2) ** 2
            max_travel_penalty = (result['max_travel_hours'] / 8) ** 2  # Penalty grows quickly after 8 hours
            
            # Calculate overall score (lower is better)
            travel_score = (
                avg_travel_penalty * 0.6 +  # Heavy weight on average travel time
                max_travel_penalty * 0.4    # Some weight on maximum travel time
            )
            
            # Store normalized scores
            result['normalized_co2_score'] = co2_score
            result['normalized_travel_score'] = travel_score
            
            # Calculate final composite score (lower is better)
            result['composite_score'] = (
                co2_weight * co2_score +
                fairness_weight * travel_score
            )
        
        return sorted(results, key=lambda x: (x['composite_score'], x['average_travel_hours']))

def create_world_map(results: List[Dict], selected_result: Dict, attendees: Dict):
    """Create interactive world map showing travel flows"""
    
    fig = go.Figure()
    
    # Add attendee cities with size proportional to attendees
    for city, count in attendees.items():
        if city in CITY_COORDS:
            lat, lon = CITY_COORDS[city]
            fig.add_trace(go.Scattergeo(
                lon=[lon], lat=[lat],
                text=[f"{city}<br>{count} attendees"],
                mode='markers+text',
                marker=dict(size=max(8, min(count * 2, 40)), color='blue', opacity=0.7),
                name='Office Locations',
                showlegend=False
            ))
    
    # Add destination
    dest_city = selected_result['destination']
    if dest_city in CITY_COORDS:
        lat, lon = CITY_COORDS[dest_city]
        fig.add_trace(go.Scattergeo(
            lon=[lon], lat=[lat],
            text=[f"{dest_city}<br>(Meeting Location)"],
            mode='markers+text',
            marker=dict(size=30, color='red', symbol='star'),
            name='Meeting Location',
            showlegend=False
        ))
        
        # Add flight paths with varying thickness based on attendee count
        for origin, count in attendees.items():
            if origin in CITY_COORDS and origin != dest_city:
                origin_lat, origin_lon = CITY_COORDS[origin]
                travel_hours = selected_result['attendee_travel_hours'].get(origin, 0)
                
                # Line thickness proportional to attendee count
                line_width = max(1, min(count / 2, 8))
                
                fig.add_trace(go.Scattergeo(
                    lon=[origin_lon, lon], lat=[origin_lat, lat],
                    mode='lines',
                    line=dict(width=line_width, color='orange'),
                    name=f'{origin} → {dest_city}',
                    hovertext=f'{count} people, {travel_hours:.1f} hours',
                    showlegend=False
                ))
    
    flight_type = "Estimated" if selected_result.get('has_estimated_flights', False) else "Actual"
    total_attendees = sum(attendees.values())
    
    fig.update_layout(
        title=f'Global Meeting: {dest_city} ({total_attendees} attendees, {flight_type} flights)',
        geo=dict(
            projection_type='natural earth',
            showland=True,
            landcolor='rgb(243, 243, 243)',
            coastlinecolor='rgb(204, 204, 204)',
            showocean=True,
            oceancolor='rgb(230, 245, 255)'
        ),
        height=600
    )
    
    return fig

def create_comparison_chart(results: List[Dict]):
    """Create comparison chart of top destinations"""
    
    if not results:
        return None
    
    top_results = results[:10]
    
    cities = [r['destination'] for r in top_results]
    co2_values = [r['total_co2_tonnes'] for r in top_results]
    fairness_values = [r['fairness_score'] for r in top_results]
    avg_travel = [r['average_travel_hours'] for r in top_results]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Total CO2 Emissions', 'Travel Fairness Score (Higher is Better)', 
                       'Average Travel Time', 'Max Travel Time'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    fig.add_trace(go.Bar(x=cities, y=co2_values, name='CO2 (tonnes)', marker_color='red'), row=1, col=1)
    fig.add_trace(go.Bar(x=cities, y=fairness_values, name='Fairness Score', marker_color='orange'), row=1, col=2)
    fig.add_trace(go.Bar(x=cities, y=avg_travel, name='Avg Travel (hours)', marker_color='blue'), row=2, col=1)
    
    max_travel = [r['max_travel_hours'] for r in top_results]
    fig.add_trace(go.Bar(x=cities, y=max_travel, name='Max Travel (hours)', marker_color='purple'), row=2, col=2)
    
    fig.update_layout(height=600, showlegend=False, title_text="Top 10 Destination Comparison")
    fig.update_xaxes(tickangle=45)
    
    return fig

def generate_output_json(result: Dict, scenario: Dict) -> Dict:
    """Generate the required JSON output format"""
    
    travel_plans = result.get('travel_plans', {})
    if travel_plans:
        departures = []
        arrivals = []
        for plan in travel_plans.values():
            departures.append(plan['outbound_departure'])
            arrivals.append(plan['inbound_arrival'])
        
        min_arrival = min(departures)
        max_departure = max(arrivals)
    else:
        min_arrival = result['event_start'] - timedelta(hours=4)
        max_departure = result['event_end'] + timedelta(hours=4)
    
    return {
        "event_location": result['destination'],
        "event_dates": {
            "start": result['event_start'].isoformat() + 'Z',
            "end": result['event_end'].isoformat() + 'Z'
        },
        "event_span": {
            "start": min_arrival.isoformat() + 'Z',
            "end": max_departure.isoformat() + 'Z'
        },
        "total_co2": round(result['total_co2_tonnes'], 1),
        "average_travel_hours": round(result['average_travel_hours'], 1),
        "median_travel_hours": round(result['median_travel_hours'], 1),
        "max_travel_hours": round(result['max_travel_hours'], 1),
        "min_travel_hours": round(result['min_travel_hours'], 1),
        "attendee_travel_hours": {
            city: round(hours, 1) 
            for city, hours in result['attendee_travel_hours'].items()
        }
    }

# ------------ STREAMLIT APP ------------
def main():
    st.set_page_config(page_title="Meeting Location Optimizer", layout="wide")
    
    # Custom CSS for larger fonts
    st.markdown("""
        <style>
        .big-font {
            font-size:50px !important;
            font-weight:bold;
            color:#1f77b4;
        }
        .tree-counter {
            font-size:40px !important;
            color:#2ca02c;
        }
        .subtitle {
            font-size:24px !important;
            color:#666;
        }
        .flight-booking {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #1f77b4;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🌍 Meeting Location Optimizer with Smart Flight Search")
    st.markdown("*Balancing CO2 emissions with travel fairness using targeted flight data*")
    
    # Initialize systems
    flight_optimizer = None
    location_optimizer = None
    crystal_ball = CarbonCrystalBall()
    
    # Sidebar controls
    st.sidebar.header("Configuration")
    
    # Optimization weights
    st.sidebar.subheader("Optimization Weights")
    co2_weight = st.sidebar.slider("CO2 Emissions Weight", 0.0, 1.0, 0.5, 0.1)
    fairness_weight = st.sidebar.slider("Travel Fairness Weight", 0.0, 1.0, 0.5, 0.1)
    
    # Calculate normalized weights
    total_weight = co2_weight + fairness_weight
    if total_weight > 0:
        normalized_co2 = co2_weight / total_weight
        normalized_fairness = fairness_weight / total_weight
        co2_weight = normalized_co2
        fairness_weight = normalized_fairness
    else:
        co2_weight, fairness_weight = 0.5, 0.5
    
    st.sidebar.markdown(f"""
        *CO2 weight: **{co2_weight:.1%}** - Minimizes carbon emissions*
        
        *Fairness weight: **{fairness_weight:.1%}** - Balances travel times*
    """)
    
    # Sample scenarios
    sample_scenarios = {
        "Global Company All-Hands": {
            "attendees": {
                "London": 27, "Paris": 3, "Hong Kong": 19, "Singapore": 12,
                "Mumbai": 8, "Dubai": 24, "Shanghai": 6, "Zurich": 15,
                "Geneva": 29, "Aarhus": 2, "Sydney": 21, "Wroclaw": 10, "Budapest": 14
            },
            "availability_window": {"start": "2026-02-11T07:30:00Z", "end": "2026-03-11T10:00:00Z"},
            "event_duration": {"days": 4, "hours": 5}
        },
        "Asia-Pacific Meeting": {
            "attendees": {"Mumbai": 2, "Shanghai": 3, "Hong Kong": 1, "Singapore": 2, "Sydney": 2},
            "availability_window": {"start": "2025-03-10T09:00:00Z", "end": "2025-03-15T17:00:00Z"},
            "event_duration": {"days": 0, "hours": 4}
        },
        "European Meeting": {
            "attendees": {"London": 2, "Paris": 3, "Frankfurt": 1, "Zurich": 4},
            "availability_window": {"start": "2025-02-10T09:00:00Z", "end": "2025-02-15T17:00:00Z"},
            "event_duration": {"days": 0, "hours": 3}
        }
    }
    
    # Scenario input
    st.subheader("📝 Meeting Scenario")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_sample = st.selectbox(
            "Load Sample Scenario:", 
            ["Custom"] + list(sample_scenarios.keys())
        )
    
    with col2:
        if selected_sample != "Custom":
            default_scenario = sample_scenarios[selected_sample]
        else:
            default_scenario = {
                "attendees": {"London": 2, "Paris": 3},
                "availability_window": {"start": "2025-01-15T09:00:00Z", "end": "2025-01-20T17:00:00Z"},
                "event_duration": {"days": 0, "hours": 4}
            }
    
    scenario_text = st.text_area(
        "Scenario JSON:", 
        value=json.dumps(default_scenario, indent=2), 
        height=200
    )
    
    # Parse scenario
    try:
        scenario = json.loads(scenario_text)
        st.success("✅ Valid JSON scenario")
        
        # Show office summary
        total_offices = len(scenario['attendees'])
        total_attendees = sum(scenario['attendees'].values())
        st.info(f"📊 **Offices**: {total_offices} | **Total Attendees**: {total_attendees} | **Event Duration**: {scenario['event_duration']['days']} days, {scenario['event_duration']['hours']} hours")
        
    except Exception as e:
        st.error(f"❌ Invalid JSON: {e}")
        st.stop()
    
    # Check for missing cities
    missing_cities = [city for city in scenario['attendees'].keys() if city not in CITY_COORDS]
    if missing_cities:
        st.warning(f"⚠️ Unknown cities (will use fallback estimates): {', '.join(missing_cities)}")
    
    # Optimize button
    if st.button("🚀 Find Location With Available Flights", type="primary"):
        
        # Parse availability window for targeted data loading
        start_time = parse_iso_utc(scenario["availability_window"]["start"])
        end_time = parse_iso_utc(scenario["availability_window"]["end"])
        
        # Step 1: Load targeted flight data and optimize locations
        with st.spinner("Step 1: Loading targeted flight data and analyzing locations..."):
            # Load flight data for this specific scenario
            flight_data = load_targeted_flight_data(start_time, end_time)
            
            # Initialize optimizers with loaded data
            flight_optimizer = FlightOptimizer(flight_data)
            location_optimizer = LocationOptimizer(flight_optimizer)
            
            # Get location options
            location_options = location_optimizer.optimize_location(
                scenario, co2_weight, fairness_weight
            )
        
        st.success(f"✅ Found {len(location_options)} location options!")
        
        # Step 2: Find location with available flights
        with st.spinner("Step 2: Finding location with available flights for all attendees..."):
            event_duration = timedelta(
                days=scenario["event_duration"]["days"],
                hours=scenario["event_duration"]["hours"]
            )
            
            # Use the enhanced booking system that tries multiple locations
            booking_system = FlightBookingSystem()
            booking_system.flight_data = flight_data  # Use the same loaded data
            
            final_result, booking_results = booking_system.find_flights_with_location_fallback(
                scenario['attendees'],
                location_options,
                start_time,
                end_time,
                event_duration
            )
        
        # Display results
        best_result = final_result
        
        if best_result.get('has_estimated_flights', False):
            st.info("ℹ️ Some travel times are estimated due to limited flight data coverage")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 Selected Location", best_result['destination'])
        with col2:
            st.metric("🌱 Total CO2", f"{best_result['total_co2_tonnes']:.1f} tonnes")
        with col3:
            st.metric("⏱️ Avg Travel Time", f"{best_result['average_travel_hours']:.1f} hours")
        with col4:
            st.metric("📊 Fairness Score", f"{best_result['fairness_score']:.1f}")
        
        # Enhanced tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["✈️ Flight Bookings", "🗺️ World Map", "📊 Comparison", "📋 Details", "🌳 Crystal Ball", "📤 Export"])
        
        with tab1:
            st.subheader("✈️ Flight Booking Results")
            st.markdown("*Actual flights found for each office using targeted data loading*")
            
            # Summary stats
            successful_bookings = sum(1 for result in booking_results.values() if result.get('status') == 'success')
            local_attendees = sum(1 for result in booking_results.values() if result.get('status') == 'local')
            failed_bookings = len(booking_results) - successful_bookings - local_attendees
            
            scol1, scol2, scol3 = st.columns(3)
            with scol1:
                st.metric("✅ Flights Found", successful_bookings)
            with scol2:
                st.metric("🏠 Local Attendees", local_attendees)
            with scol3:
                st.metric("⚠️ No Flights", failed_bookings)
            
            # Detailed booking information
            for office, booking_info in booking_results.items():
                with st.expander(f"📍 {office} Office ({booking_info['attendees']} attendees)"):
                    
                    if booking_info['status'] == 'success':
                        st.markdown('<div class="flight-booking">', unsafe_allow_html=True)
                        
                        # Show which search strategy worked
                        if 'search_strategy' in booking_info:
                            st.info(f"🎯 Found using: {booking_info['search_strategy']}")
                        
                        # Outbound flight
                        st.markdown("#### 🛫 Outbound Flight")
                        out = booking_info['outbound']
                        st.markdown(f"""
                        **Flight**: {out['carrier']} {out['flight_number']}
                        **Route**: {out['departure_airport']} → {out['arrival_airport']}
                        **Departure**: {out['departure_time']}
                        **Arrival**: {out['arrival_time']}
                        **Duration**: {out['duration_minutes']} minutes
                        """)
                        
                        # Return flight
                        st.markdown("#### 🛬 Return Flight")
                        ret = booking_info['return']
                        st.markdown(f"""
                        **Flight**: {ret['carrier']} {ret['flight_number']}
                        **Route**: {ret['departure_airport']} → {ret['arrival_airport']}
                        **Departure**: {ret['departure_time']}
                        **Arrival**: {ret['arrival_time']}
                        **Duration**: {ret['duration_minutes']} minutes
                        """)
                        
                        # Costs and summary
                        totals = booking_info['totals']
                        st.markdown("#### 💰 Cost Summary")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Per Person", f"${totals['cost_per_person_usd']:.0f}")
                        with col2:
                            st.metric("Total Cost", f"${totals['total_cost_usd']:.0f}")
                        with col3:
                            st.metric("Total Travel Time", f"{totals['total_travel_hours']} hours")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    elif booking_info['status'] == 'local':
                        st.success(f"🏠 Local attendees in {office} - no flights needed!")
                        
                    else:
                        st.error(f"❌ Could not find flights for {office}")
                        st.markdown(f"**Message**: {booking_info.get('message', 'Unknown error')}")
            
            # Total booking summary
            total_cost = sum(
                result.get('totals', {}).get('total_cost_usd', 0)
                for result in booking_results.values()
                if result.get('status') == 'success'
            )
            
            if total_cost > 0:
                st.markdown("---")
                st.markdown(f"### 💰 Total Flight Costs: ${total_cost:,.0f}")
                st.markdown(f"**Average per attendee**: ${total_cost/sum(scenario['attendees'].values()):,.0f}")
        
        with tab2:
            st.subheader("Global Travel Flow Visualization")
            map_fig = create_world_map(location_options, best_result, scenario['attendees'])
            st.plotly_chart(map_fig, use_container_width=True)
            
            # Travel details table
            st.subheader("Travel Plan Details")
            travel_data = []
            for origin, attendee_count in scenario['attendees'].items():
                if origin in best_result['attendee_travel_hours']:
                    hours = best_result['attendee_travel_hours'][origin]
                    if origin in best_result.get('travel_plans', {}):
                        plan = best_result['travel_plans'][origin]
                        outbound = f"{plan['outbound_flight']['DEPAPT']} → {plan['outbound_flight']['ARRAPT']}"
                        return_flight = f"{plan['inbound_flight']['DEPAPT']} → {plan['inbound_flight']['ARRAPT']}"
                        co2 = plan['total_co2_tonnes']
                    else:
                        outbound = "Local (no travel)"
                        return_flight = "Local (no travel)"
                        co2 = 0.0
                    
                    travel_data.append({
                        'Office': origin,
                        'Attendees': attendee_count,
                        'Travel Hours': f"{hours:.1f}",
                        'CO2 per Person (tonnes)': f"{co2/attendee_count:.2f}" if attendee_count > 0 else "0.00",
                        'Total CO2 (tonnes)': f"{co2:.1f}",
                        'Outbound': outbound,
                        'Return': return_flight
                    })
            
            st.dataframe(pd.DataFrame(travel_data), use_container_width=True)
        
        with tab3:
            st.subheader("Destination Comparison")
            if len(location_options) > 1:
                comparison_fig = create_comparison_chart(location_options)
                if comparison_fig:
                    st.plotly_chart(comparison_fig, use_container_width=True)
                
                # Detailed comparison table
                comparison_data = []
                for i, result in enumerate(location_options[:10]):
                    comparison_data.append({
                        'Rank': i + 1,
                        'Destination': result['destination'],
                        'Total CO2 (tonnes)': f"{result['total_co2_tonnes']:.1f}",
                        'Avg Travel (hours)': f"{result['average_travel_hours']:.1f}",
                        'Max Travel (hours)': f"{result['max_travel_hours']:.1f}",
                        'Fairness Score': f"{result['fairness_score']:.2f}",
                        'Composite Score': f"{result['composite_score']:.3f}"
                    })
                
                st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
        
        with tab4:
            st.subheader("Detailed Analysis")
            insights = [
                f"**Selected location**: {best_result['destination']} - found available flights",
                f"**CO2 impact**: {best_result['total_co2_tonnes']:.1f} tonnes total emissions",
                f"**Travel equity**: {best_result['fairness_score']:.1f} calculated fairness score",
                f"**Time range**: {best_result['min_travel_hours']:.1f}h - {best_result['max_travel_hours']:.1f}h travel times",
                f"**Per person average**: {best_result['total_co2_tonnes']/sum(scenario['attendees'].values()):.2f} tonnes CO2",
                f"**Flight availability**: {successful_bookings}/{len(scenario['attendees'])} offices with confirmed flights"
            ]
            for insight in insights:
                st.markdown(f"- {insight}")
        
        with tab5:
            try:
                st.subheader("🔮 Carbon Crystal Ball - Environmental Impact Predictions")
                st.markdown("*Understanding the true environmental cost of your global meeting*")
                
                meeting_co2 = best_result['total_co2_tonnes']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<p class="big-font">Total CO2 Emissions</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="big-font">{meeting_co2:.1f} tonnes</p>', unsafe_allow_html=True)
                    
                with col2:
                    tree_data = crystal_ball.calculate_trees_needed(meeting_co2)
                    if "error" not in tree_data:
                        st.markdown('<p class="big-font">Trees Needed to Offset</p>', unsafe_allow_html=True)
                        st.markdown(f'<p class="big-font">{tree_data["trees_needed"]:,} 🌳</p>', unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Planting Cost", f"${tree_data['planting_cost_usd']:,.0f}")
                        with col2:
                            st.metric("Land Required", f"{tree_data['hectares_needed']:.1f} hectares")
                        with col3:
                            st.metric("Years to Offset", f"{tree_data['years_to_full_offset']} years")
                            
                        # Visualizations
                        alternatives = crystal_ball.calculate_offset_alternatives(meeting_co2)
                        fig_trees, fig_costs = create_crystal_ball_visualizations(tree_data, alternatives)
                        
                        if fig_trees:
                            st.plotly_chart(fig_trees, use_container_width=True)
                        
                        if fig_costs:
                            st.plotly_chart(fig_costs, use_container_width=True)
                
            except Exception as e:
                st.error(f"Crystal Ball error: {e}")
        
        with tab6:
            st.subheader("Export Results")
            
            # Enhanced JSON output with booking information
            output_json = generate_output_json(best_result, scenario)
            output_json['flight_bookings'] = booking_results
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📄 Complete Results (JSON)")
                st.code(json.dumps(output_json, indent=2, default=str), language='json')
                
                st.download_button(
                    label="📥 Download Complete Results",
                    data=json.dumps(output_json, indent=2, default=str),
                    file_name=f"complete_meeting_optimization_{best_result['destination'].lower()}.json",
                    mime="application/json"
                )
            
            with col2:
                # Summary with booking info
                booking_summary = f"""
## Flight Booking Summary
- **Total Flight Costs**: ${total_cost:,.0f}
- **Average per Attendee**: ${total_cost/sum(scenario['attendees'].values()):,.0f}
- **Successful Bookings**: {successful_bookings}/{len(booking_results)} offices
- **Flight Coverage**: {(successful_bookings + local_attendees)/len(booking_results)*100:.1f}%
"""
                
                summary = f"""
# Meeting Optimization Report

**Selected Location**: {best_result['destination']}
**Meeting Dates**: {output_json['event_dates']['start']} - {output_json['event_dates']['end']}
**Total Offices**: {len(scenario['attendees'])}
**Total Attendees**: {sum(scenario['attendees'].values())}

## Environmental Impact
- **Total CO2 Emissions**: {output_json['total_co2']} tonnes
- **Per Attendee Average**: {output_json['total_co2'] / sum(scenario['attendees'].values()):.2f} tonnes

{booking_summary}

*Generated by Meeting Location Optimizer with Smart Flight Search*
"""
                
                st.markdown(summary)
                
                st.download_button(
                    label="📥 Download Summary",
                    data=summary,
                    file_name=f"meeting_summary_{best_result['destination'].lower()}.md",
                    mime="text/markdown"
                )

if __name__ == "__main__":
    main()