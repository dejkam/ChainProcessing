import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from glob import glob
from datetime import datetime
import plotly.graph_objects as go
import sys
import os
from pathlib import Path
from tqdm import tqdm

HARD_CODED_IRL_PATH = "/home/adezhkam/MARL/MARL-COVID/ChainProcessing"

# 2) Insert that folder in sys.path
if os.path.isdir(HARD_CODED_IRL_PATH):
    if HARD_CODED_IRL_PATH not in sys.path:
        sys.path.insert(0, HARD_CODED_IRL_PATH)
else:
    print("Error: The HARD_CODED_IRL_PATH does not exist or is not a directory!")

print("Final sys.path:", sys.path)

# ------------------------------
# 1. Initial Setup and Required Paths
# ------------------------------
st.set_page_config(
    page_title="Travel Diary Analysis",
    page_icon="📊",
    layout="wide"
)
st.markdown(
    """
    <style>
    /* Make tabs prettier */
    div[data-baseweb="tab"] {
        background-color: #f0f2f6;
        padding: 10px 16px;
        margin-right: 8px;
        border-radius: 10px;
        color: black;
        font-weight: 600;
        border: 1px solid #d0d0d0;
    }

    /* Active (selected) tab */
    div[data-baseweb="tab"][aria-selected="true"] {
        background-color: #0068c9;
        color: white;
        border: 1px solid #0068c9;
    }

    /* Move tabs a little lower */
    div[data-baseweb="tab-list"] {
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)



tab1, tab2, tab3, tab4 = st.tabs(["Preprocessing","EDA and Chain Processing", "IRL Training", "IRL Results"])

with tab1:
    st.title("🔧 Preprocessing & Data Setup")

    # 1) Paths  
    SOURCE_FOLDER = '../dataset/1000_users_fixed_new/2020/04'
    DEST_FOLDER = 'dataset/1000_users_fixed_new/2020/04'
    POI_FILE = 'dataset/poi.csv'

    st.write("This tab handles data copying, deleting, optional Factorized ID generation, and adding POI class/name/category.")

    # 2) Copy Data Section
    #st.subheader("📂 Copy / Update Data")
    def copy_data(SOURCE_FOLDER, DEST_FOLDER):
        fileformat = '*travel_diary.csv'
        SOURCE_CSV_FILES = glob(os.path.join(SOURCE_FOLDER, fileformat))
        if not SOURCE_CSV_FILES:
            st.warning("⚠️ No CSV files found in source folder!")
            return

        os.makedirs(DEST_FOLDER, exist_ok=True)
        progress_bar = st.progress(0.0, "Copying data...")

        for i, filename_src in enumerate(SOURCE_CSV_FILES):
            df_src = pd.read_csv(filename_src)
            # Basic rename if needed
            if 'poi_id_num' in df_src.columns:
                df_src.rename(columns={'poi_id_num': 'poi_id'}, inplace=True)
            # Save
            base = os.path.basename(filename_src)
            out_path = os.path.join(DEST_FOLDER, base)
            df_src.to_csv(out_path, index=False)

            progress_bar.progress((i+1)/len(SOURCE_CSV_FILES))

        progress_bar.empty()
        st.success("✅ Data copied successfully!")

    # ----------------------------
    # Tab1: 1. Empty Destination Folder
    # ----------------------------
    st.subheader("Empty Destination Folder")
    st.write("Deletes **all files** in DEST_FOLDER. Use with caution!")

    def clear_destination_folder(folder_path):
        import shutil
        files = glob(os.path.join(folder_path, "*"))
        if not files:
            st.info("No files found in destination folder to delete.")
            return
        for file in files:
            try:
                if os.path.isfile(file) or os.path.islink(file):
                    os.unlink(file)
                elif os.path.isdir(file):
                    shutil.rmtree(file)
            except Exception as e:
                st.error(f"Failed to delete {file}. Reason: {e}")
        st.success(f"✅ Cleared all contents from: {folder_path}")

    if st.button("🗑️ Delete All Files in DEST_FOLDER"):
        clear_destination_folder(DEST_FOLDER)

    st.divider()
        

    # Check if any CSVs exist
    dest_file_exist = bool(glob(os.path.join(DEST_FOLDER, '*_travel_diary.csv')))

    colA, colB = st.columns([1,2])
    with colA:
        if not dest_file_exist:
            st.warning("⚠️ Destination folder is empty. Copy data to proceed.")
        else:
            st.success("✅ Destination folder has data. You can overwrite if needed.")
    with colB:
        if st.button("📂 Copy/Update Data"):
            copy_data(SOURCE_FOLDER, DEST_FOLDER)

    st.divider()

    # 3) Factorized ID Generation
    st.subheader("🆔 Factorized ID Generation")
    st.write("Generates 'FactorizedID' for each agent in the CSVs, plus a mapping file.")

    def generate_consistent_factorized_ids():
        csv_files = glob(os.path.join(DEST_FOLDER, "*_travel_diary.csv"))
        if not csv_files:
            st.warning("No CSV files found in DEST_FOLDER.")
            return

        all_data = []
        for path in csv_files:
            df_ = pd.read_csv(path)
            df_["source_file"] = path
            all_data.append(df_)

        combined = pd.concat(all_data, ignore_index=True)
        combined["FactorizedID"], unique_agent_ids = pd.factorize(combined["agent_id"])
        # Save mapping
        mapping_path = os.path.join(DEST_FOLDER, "agent_id_mapping.csv")
        pd.DataFrame({"agent_id": unique_agent_ids,
                      "FactorizedID": range(len(unique_agent_ids))
        }).to_csv(mapping_path, index=False)

        # Overwrite each CSV
        for path in csv_files:
            sub = combined[combined["source_file"] == path].drop(columns=["source_file"])
            sub.to_csv(path, index=False)

        st.success("✅ Factorized IDs generated and saved. agent_id_mapping.csv created.")

    if st.button("🔄 Generate Factorized IDs"):
        generate_consistent_factorized_ids()

    st.divider()

    # 4) Add 'class', 'name', 'category' from POI
    st.subheader("🏷 Add POI Info (Class/Name/Category)")
    st.write("Uses 'poi.csv' to enrich each row with the corresponding POI info.")
    def add_class_and_name_and_type(df_poi, df_path):
        # Create dictionary for fast lookup with both 'class' and 'name'
        poi_dict = df_poi.set_index('poi_id')[['class', 'name', 'category']].to_dict(orient='index')
        
        # Read the mobility data
        df_mobility = pd.read_csv(df_path, index_col=None)

        # Convert 'poi_id' to integer safely (handle NaNs)
        df_mobility['poi_id'] = pd.to_numeric(df_mobility['poi_id'], errors='coerce').fillna(-1).astype(int)

        # Assign 'poi_class' and 'name' only where conditions are met
        df_mobility['class'] = df_mobility.apply(
            lambda row: poi_dict[row['poi_id']]['class'] if (row['poi_id'] in poi_dict and row['mode'] == 'stationary' and pd.isna(row['is_home'])) else None,
            axis=1
        )

        df_mobility['name'] = df_mobility.apply(
            lambda row: poi_dict[row['poi_id']]['name'] if (row['poi_id'] in poi_dict and row['mode'] == 'stationary' and pd.isna(row['is_home'])) else None,
            axis=1
        )

        df_mobility['category'] = df_mobility.apply(
            lambda row: poi_dict[row['poi_id']]['category'] if (row['poi_id'] in poi_dict and row['mode'] == 'stationary' and pd.isna(row['is_home'])) else None,
            axis=1
        )
        # Save the updated CSV file
        df_mobility.to_csv(df_path, index=False)
        


    if st.button("📝 Enrich with POI data"):
        
        df_poi = pd.read_csv(POI_FILE, index_col=None)
        df_poi.rename(columns={'poi_id':'id','poi_id_num':'poi_id'}, inplace=True)
        # Possibly rename if your poi.csv uses different columns
        # Example:
        # if 'poi_id_num' in df_poi.columns:
        #     df_poi.rename(columns={'poi_id_num':'poi_id'}, inplace=True)

        csv_files = glob(os.path.join(DEST_FOLDER, "*_travel_diary.csv"))
        if not csv_files:
            st.warning("No CSVs to enrich in DEST_FOLDER.")
        else:
            total_files = len(csv_files)
            progress_bar = st.progress(0)
            i = 0

            for path in tqdm(csv_files, desc="Adding POI class/name/category"):
                i = i + 1
                add_class_and_name_and_type(df_poi, path)
                progress_bar.progress(i / total_files)
                
            st.success("✅ All files enriched with POI info.")
            progress_bar.empty()

    # 5) Let other tabs know about the DEST_FOLDER
    st.session_state["DATA_FOLDER"] = DEST_FOLDER
    st.info(f"**DATA_FOLDER** set to: {DEST_FOLDER} (for other Tabs usage)") 

    
with tab2:

    if "expander_map" not in st.session_state:
        st.session_state["expander_map"] = False

    if "map_fig" not in st.session_state:
        st.session_state["map_fig"] = None

    if "expander_chain" not in st.session_state:
        st.session_state["expander_chain"] = False

    st.title("📊 Travel Diary Analysis -- ")
    st.write(
        "This dashboard loads individuals' travel diaries and provides on‐demand functionalities "
        "for trajectory analysis."
    )

    #SOURCE_FOLDER = '../dataset/1000_users_processed'  # Original data source
    #DEST_FOLDER = 'dataset/home_location_fixed/2020/03'  # Destination folder in dashboard
    #DEST_FOLDER = 'dataset/no_imputed/2020/03'  # Destination folder in dashboard
    #SOURCE_FOLDER = '../dataset/1000_users_fixed/2020/03'  # Original data source
    #DEST_FOLDER = 'dataset/1000_users_fixed/2020/03'  # Destination folder in dashboard
    DATA_FOLDER = st.session_state.get("DATA_FOLDER", None)
    POI_INFO = 'dataset/poi.csv'

    # COLUMNS_TO_READ = [
    #     'agent_id', 'time', 'lng', 'lat', 'link_id', 'type',
    #     'mode', 'is_home', 'poi_id_num', 'home_h3_id', 'segment_id', 'poi_class', 'name', 'category'
    # ]
    SOURCE_COLUMNS = [
        'agent_id', 'time', 'lng', 'prj_lng', 'prj_lat', 'lat', 'link_id', 'type',
        'mode', 'is_home', 'poi_id_num', 'segment_id']
    # SOURCE_COLUMNS = [
    #      'agent_id', 'time', 'lng', 'lat', 'link_id', 'type',
    #      'mode', 'is_home', 'poi_id_num', 'segment_id']

    COLUMNS_TO_READ = [
        'agent_id', 'time', 'lng', 'prj_lng', 'prj_lat', 'lat', 'link_id', 'type',
        'mode', 'is_home', 'poi_id_num', 'segment_id', 'class', 'name', 'category'
    ]
    csv_files = glob(os.path.join(DATA_FOLDER, "*_travel_diary.csv"))

    file_dict = {}
    for f in csv_files:
        filename = os.path.basename(f)
        day_str = str(int(filename.split("_")[0]))
        file_dict[day_str] = f

    #sorted_days = sorted(file_dict.keys(), key=lambda x: int(x))
    #sorted_days = sorted(file_dict.keys(), key=lambda x: int(x.lstrip("0")) if x.lstrip("0") else 0)
    sorted_days = sorted(file_dict.keys(), key=int)  # Ensure correct order




    if "selected_days" not in st.session_state:
        st.session_state["selected_days"] = []

    # ------------------------------
    # 2. Day Selection UI
    # ------------------------------
    st.subheader("Select Days to Process")

    select_all = st.button("✅ Select All Days")
    clear_all = st.button("❌ Clear Selection")

    if select_all:
        st.session_state["selected_days"] = sorted_days.copy()
    if clear_all:
        st.session_state["selected_days"].clear()

    cols = st.columns(5)
    for i, d in enumerate(sorted_days):
        with cols[i % 5]:
            checked = d in st.session_state["selected_days"]
            new_val = st.checkbox(f"Day {d}", value=checked)
            if new_val and d not in st.session_state["selected_days"]:
                st.session_state["selected_days"].append(d)
            elif not new_val and d in st.session_state["selected_days"]:
                st.session_state["selected_days"].remove(d)

    # ------------------------------
    # 3. On-Demand Loading
    # ------------------------------
    def load_data_for_day(day_str):
        if day_str not in file_dict:
            return pd.DataFrame()
        return pd.read_csv(file_dict[day_str])

    def load_data_for_days(day_list):
        df_list = []
        for day_str in day_list:
            if day_str in file_dict:
                df_temp = pd.read_csv(file_dict[day_str])
                df_list.append(df_temp)
        if df_list:
            return pd.concat(df_list, ignore_index=True)
        else:
            return pd.DataFrame()

    # ------------------------------
    # 5. Travel Statistics (Selected Days)
    # ------------------------------
    st.subheader("📊 Process Travel Statistics for Selected Days")

    if "stats_loaded" not in st.session_state:
        st.session_state["stats_loaded"] = False
    if "current_day_index" not in st.session_state:
        st.session_state["current_day_index"] = 0

    def process_data(df):
        unique_agents = df["FactorizedID"].nunique() if "FactorizedID" in df.columns else 0
        if "is_imputed" in df.columns and len(df) > 0:
            imputation_pct = (df[df["is_imputed"] == True].shape[0] / len(df)) * 100
        else:
            imputation_pct = None

        home = df[(df.get("is_home") == True) & (df.get("mode") == "stationary")].shape[0]
        poi = df[(df.get("is_home") != True) & (df.get("mode") == "stationary")].shape[0]
        link = df[(df.get("mode") != "stationary")].shape[0]
        total_records = df.shape[0]

        def pct(x):
            return round((x / total_records) * 100, 2) if total_records else 0

        return {
            "unique_agents": unique_agents,
            "imputation_pct": imputation_pct,
            "home_pct": pct(home),
            "poi_pct": pct(poi),
            "link_pct": pct(link),
            "total_records": total_records
        }

    if st.button("Compute Travel Statistics"):
        st.session_state["stats_loaded"] = True
        st.session_state["current_day_index"] = 0

    day_list = sorted(st.session_state["selected_days"], key=lambda x: int(x))

    if st.session_state["stats_loaded"]:
        if not day_list:
            st.warning("No days selected for stats.")
        else:
            if st.session_state["current_day_index"] >= len(day_list):
                st.session_state["current_day_index"] = 0

            current_day = day_list[st.session_state["current_day_index"]]
            df_stats = load_data_for_day(current_day)

            if df_stats.empty:
                st.warning(f"No data for day {current_day}.")
            else:
                stats = process_data(df_stats)
                st.write(f"**Statistics for Day {current_day}**")

                colA, colB = st.columns(2)
                colA.metric("Unique Agents", stats["unique_agents"])
                if stats["imputation_pct"] is not None:
                    colB.metric("Imputation %", f"{stats['imputation_pct']:.2f}%")
                else:
                    colB.write("Imputation %: N/A")

                col1, col2, col3, col4 = st.columns([1, 4, 4, 1])
                with col1:
                    if st.session_state["current_day_index"] > 0:
                        if st.button("⬅️ Previous", key="prev_stats"):
                            st.session_state["current_day_index"] -= 1

                with col2:
                    pie_data_loc = pd.DataFrame({
                        "Category": ["Home", "POI", "Travel"],
                        "Percentage": [stats["home_pct"], stats["poi_pct"], stats["link_pct"]]
                    })
                    fig_loc = px.pie(
                        pie_data_loc, names="Category", values="Percentage",
                        title="📍Location/Travel Distribution", hole=0.4
                    )
                    st.plotly_chart(fig_loc, use_container_width=True)

                with col3:
                    if "mode" in df_stats.columns:
                        df_travel = df_stats[df_stats["mode"] != "stationary"]
                        if not df_travel.empty:
                            mode_counts = df_travel["mode"].value_counts(normalize=True)*100
                            pie_data_mode = pd.DataFrame({
                                "Mode": mode_counts.index, 
                                "Percentage": mode_counts.values
                            })
                            fig_mode = px.pie(
                                pie_data_mode, names="Mode", values="Percentage",
                                title="🚗Travel Mode Distribution", hole=0.4
                            )
                            st.plotly_chart(fig_mode, use_container_width=True)
                        else:
                            st.info("No non-stationary records for travel mode distribution.")

                with col4:
                    if st.session_state["current_day_index"] < len(day_list)-1:
                        if st.button("➡️ Next", key="next_stats"):
                            st.session_state["current_day_index"] += 1

            del df_stats
    else:
        st.info("Select days and click 'Compute Travel Statistics' to see location/travel stats.")

    # ------------------------------
    # 6. Number of Agents in ALL CSV Files
    # ------------------------------
    st.subheader("👤 Agents in ALL CSV Files")

    def commonIDs(csv_file_list):
        if not csv_file_list:
            return 0
        dfs = []
        for path in csv_file_list:
            df_ = pd.read_csv(path, usecols=["agent_id"])
            dfs.append(df_)

        if not dfs:
            return 0
        common_set = set(dfs[0]["agent_id"].unique())
        for df_ in dfs[1:]:
            common_set.intersection_update(df_["agent_id"].unique())
        return len(common_set)

    if st.button("Count Agents Appearing in ALL CSVs"):
        count_all = commonIDs(csv_files)
        st.metric("Recurring Agents in ALL Days", count_all)

    # ------------------------------
    # 7. Home Location Changes
    # ------------------------------
    st.subheader("🏠 Home Location Change Analysis (Selected Days)")

    def detect_home_changes(day_list):
        if not day_list:
            return pd.DataFrame()

        day_list_sorted = sorted(day_list, key=lambda x: int(x))
        # agent_home_history maps: FactorizedID -> ( agent_id, [ (day, home_h3_id), (day, home_h3_id), ... ] )
        agent_home_history = {}

        for d in day_list_sorted:
            # Read all three columns so we can build a dictionary
            df_d = pd.read_csv(file_dict[d], usecols=["agent_id", "FactorizedID", "home_h3_id"])
            
            # For each FactorizedID in this day, get the *first* home location
            # plus create a lookup from FactorizedID -> agent_id
            daily_home = df_d.groupby("FactorizedID")["home_h3_id"].first()
            fid_to_aid = df_d.groupby("FactorizedID")["agent_id"].first().to_dict()

            for fid, home_val in daily_home.items():
                aid = fid_to_aid[fid]  # find the agent_id that corresponds to this FactorizedID
                if fid not in agent_home_history:
                    # Store (agent_id, [ (day, home_val) ])
                    agent_home_history[fid] = (aid, [(d, home_val)])
                else:
                    # unpack
                    existing_aid, history_list = agent_home_history[fid]
                    if home_val != history_list[-1][1]:
                        history_list.append((d, home_val))
                    # (No need to re‐update agent_id if it’s the same for this fid.)

        changes = []
        # agent_home_history: { fid: (agent_id, [ (day, home), (day, home), ... ]) }
        for fid, (aid, hist_list) in agent_home_history.items():
            if len(hist_list) > 1:
                for i in range(len(hist_list) - 1):
                    changes.append({
                        "agent_id": aid,
                        "FactorizedID": fid,
                        "Previous Day": hist_list[i][0],
                        "Previous Home": hist_list[i][1],
                        "New Day": hist_list[i+1][0],
                        "New Home": hist_list[i+1][1]
                    })

        df_changes = pd.DataFrame(changes)
        return df_changes


    if st.button("Analyze Home Location Changes for Selected Days"):
        if not st.session_state["selected_days"]:
            st.warning("No days selected.")
        else:
            df_changes = detect_home_changes(st.session_state["selected_days"])
            if df_changes.empty:
                st.success("No agents changed their home location in the selected days.")
            else:
                # 1) Show the DataFrame *without* 'agent_id' column
                display_cols = [c for c in df_changes.columns if c != "agent_id"]
                st.dataframe(df_changes[display_cols])

                # 2) Still save the CSV *with* agent_id
                csv_bytes = df_changes.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Home Changes CSV",
                    csv_bytes,
                    "home_location_changes.csv",
                    "text/csv"
                )
    # -------------------------------------------------------
    # 8. Individual Mobility Pattern (Map) - Enhanced
    # -------------------------------------------------------
    st.subheader("🗺 Individual Mobility Patterns (Map)")

    with st.expander("Expand", expanded=st.session_state["expander_map"]):

        st.write("View an agent's daily or monthly trajectory on a map, now with icons for home/POI and multi-color travel segments.")

        option = st.radio("Choose Time Range", ["Daily", "Monthly"], key="mobility_range")

        fac_id = st.number_input("Enter FactorizedID", min_value=0, value=0)


        def plot_map(df_agent, title="Mobility Pattern"):

            """
            Visualizes individual mobility on a map with:
            1) prj_lat / prj_lng for map matching coordinates.
            2) Stationary points (Home/POI) plotted as (Green/Blue) filled circles respectively.
            3) Travel mode are rendered by color-coded lines.
            4) If segment_id changes but mode remains the same and time gap <= 20 min,
                they are joined into one segment; otherwise split.
            """


            # Early exit if there is no data
            if df_agent.empty:
                st.info("No data found for this agent/time range.")
                return

            #Prepare Data: 1.Copy original df 2. Convert 'time' to datetime
            # 3. Sort by chronological order
            df_agent = df_agent.copy()
            df_agent["time"] = pd.to_datetime(df_agent["time"], errors="coerce")
            df_agent.sort_values("time", inplace=True)
            #df_agent["time"] = df_agent["time"].astype(str).apply(lambda x: x.replace('"','').encode('utf-8','ignore').decode('utf-8','ignore'))


            #Split dataframe into two distinct dataframes for stay and travel events
            df_stay = df_agent[df_agent["mode"] == "stationary"].copy()
            df_travel = df_agent[df_agent["mode"] != "stationary"].copy()

            # Among stationary rows, further differentiate between Home and POI locations
            df_home = df_stay[df_stay["is_home"] == True]
            df_poi  = df_stay[df_stay["is_home"] != True]

            #Create a Plotly figure and set map style
            fig = go.Figure()
            fig.update_layout(
                mapbox_style="open-street-map",   # base style
                mapbox_zoom=11,                   # initial zoom level
                mapbox_center={"lat": 1.3521, "lon": 103.8198},  # center of the map
                margin={"r":0,"t":40,"l":0,"b":0}, # reduce margins
                height=600,                       # figure height in px
                title=title                       # chart title
            )

            #Plot Stationary Points to represent Home and POI locations
            # using prj_lat/prj_lng for location
            #Home markers
            if not df_home.empty:
                fig.add_trace(go.Scattermapbox(
                    lat=df_home["prj_lat"],       # map-matched lat
                    lon=df_home["prj_lng"],       # map-matched lng
                    mode="markers+text",          # markers with text label
                    name="🏠 Home",               # legend label
                    text=["🏠"] * len(df_home),   # an emoji for each row
                    textposition="top center",    # emoji displayed above the marker
                    hovertext=df_home["time"].astype(str),  # show time on hover
                    marker=go.scattermapbox.Marker(
                        size=14,
                        color="green"             # distinct color for home
                    ),
                    showlegend=True
                ))

            #POI markers
            if not df_poi.empty:
                fig.add_trace(go.Scattermapbox(
                    lat=df_poi["prj_lat"],
                    lon=df_poi["prj_lng"],
                    mode="markers+text",
                    name="📍 POI",
                    text=["📍"] * len(df_poi),
                    textposition="top center",
                    hovertext=df_poi["time"].astype(str),
                    marker=go.scattermapbox.Marker(
                        size=14,
                        color="blue"        # distinct color for POI
                    ),
                    showlegend=True
                ))

            #Plot Travel Segments
            #1. We define a dictionary that maps mode -> color.
            #2. If travel rows belong to the same mode & segment_id are within 20 min,
            #3. we unify them in a single line, else split.

            mode_color_map = {
                "walk":  "green",
                "bus":   "blue",
                "MRT":   "purple",
                "car":   "red",
                "cycle": "orange"
            }
            shown_modes = set()       # track which modes appear in the legend
            MAX_GAP_SEC = 1200        # 20-minute threshold in seconds

            # If there is no travel data, just skip
            if not df_travel.empty:
                # Reset index for easier indexing
                df_travel.reset_index(drop=True, inplace=True)

                # Initialize references for the first travel row
                seg_start_idx = 0                         # start index of a segment
                prev_mode = df_travel.at[0, "mode"]       # previous row's mode
                prev_seg_id = df_travel.at[0, "segment_id"]  # previous row's segment_id
                prev_time = df_travel.at[0, "time"]       # previous row's time

                # Iterate through travel rows, detect breaks where:
                # either mode changes
                # or same mode but segment_id changes + time gap > 20 min
                for i in range(1, len(df_travel)):
                    row = df_travel.iloc[i]
                    current_mode = row["mode"]
                    current_seg_id = row["segment_id"]
                    current_time = row["time"]

                    # Calculate the time gap between current row & previous row
                    time_gap = (current_time - prev_time).total_seconds() if pd.notnull(prev_time) else 999999

                    same_mode = (current_mode == prev_mode)
                    seg_id_changed = (current_seg_id != prev_seg_id)

                    # Decide if we break the segment here
                    if not same_mode:
                        # Mode changed
                        seg_df = df_travel.iloc[seg_start_idx:i].copy()
                        add_travel_segment(fig, seg_df, prev_mode, mode_color_map, shown_modes)
                        seg_start_idx = i
                    else:
                        # same mode => check segment_id logic + time gap
                        if seg_id_changed and time_gap > MAX_GAP_SEC:
                            # segment_id changed, and > 20 min gap => break
                            seg_df = df_travel.iloc[seg_start_idx:i].copy()
                            add_travel_segment(fig, seg_df, prev_mode, mode_color_map, shown_modes)
                            seg_start_idx = i

                    # Update references
                    prev_mode = current_mode
                    prev_seg_id = current_seg_id
                    prev_time = current_time

                # After the loop, handle the final segment from seg_start_idx to end
                last_seg = df_travel.iloc[seg_start_idx:].copy()
                if not last_seg.empty:
                    add_travel_segment(fig, last_seg, prev_mode, mode_color_map, shown_modes)

            # Finally we render the figure
            # We add a unique key to avoid duplicate element ID conflict
            #st.plotly_chart(fig, use_container_width=True, key="map_chart_inside")
            return fig


        def add_travel_segment(fig, seg_df, mode, color_map, shown_modes):
            """
            Helper function: add a single travel segment to the figure.

            :param fig: The Plotly figure object to add traces to.
            :param seg_df: DataFrame slice containing consecutive rows in the same travel segment.
            :param mode: The travel mode for these rows (walk/cycle/car/bus/MRT).
            :param color_map: Dictionary mapping mode -> color.
            :param shown_modes: Set of modes that have already been added to the legend.
            """
            #Determine color for this mode
            color_ = color_map.get(mode, 'black')
            #Only show the legend once per mode
            show_leg = (mode not in shown_modes)
            shown_modes.add(mode)

            #Add a Scattermapbox trace (lines)
            #using 'prj_lat' / 'prj_lng' to plot a path
            fig.add_trace(go.Scattermapbox(
                lat=seg_df["prj_lat"],
                lon=seg_df["prj_lng"],
                mode="lines",
                name=mode,
                line=dict(color=color_, width=4),
                hovertext=seg_df["time"].astype(str),
                showlegend=show_leg
            ))



        if option == "Daily":
            day_list_for_map = st.multiselect("Select exactly 1 day", st.session_state["selected_days"])
            if st.button("Show Daily Pattern"):
                if len(day_list_for_map) != 1:
                    st.warning("Please select exactly one day.")
                else:
                    df_day = load_data_for_days(day_list_for_map)
                    df_agent = df_day[df_day["FactorizedID"] == fac_id].copy()
                    if df_agent.empty:
                        st.info("No data for that FactorizedID on the chosen day.")
                        st.session_state["map_fig"] = None
                    else:
                        fig = plot_map(df_agent, title=f"Daily Mobility for FactorizedID {fac_id}")
                        st.session_state["map_fig"] = fig
                    del df_day
                st.session_state["expander_map"] = True
        else:
            if st.button("Show Monthly Pattern"):
                if not st.session_state["selected_days"]:
                    st.warning("No days selected.")
                else:
                    df_month = load_data_for_days(st.session_state["selected_days"])
                    df_agent = df_month[df_month["FactorizedID"] == fac_id].copy()
                    if df_agent.empty:
                        st.info("No data for FactorizedID on selected days.")
                        st.session_state["map_fig"] = None
                    else:
                        fig = plot_map(df_agent, title=f"Monthly Mobility for FactorizedID {fac_id}")
                        st.session_state["map_fig"] = fig
                    del df_month
                st.session_state["expander_map"] = True

        # If we have a stored figure from a previous run, re-display it with a different unique key
        if st.session_state["map_fig"] is not None:
            st.plotly_chart(st.session_state["map_fig"], use_container_width=True, key="map_chart_outside")

    # ------------------------------
    # 9. Top N% of Active Agents
    # ------------------------------
    st.subheader("🔝 Top N% Most Active Agents (Selected Days)")

    with st.expander("Find Top N% Agents"):
        st.write(
            "Definition: 'active' = agents with the **largest number of records** across the selected days."
        )
        N_percent = st.number_input("Enter N% (1-100)", min_value=1, max_value=100, value=10)
        st.session_state["N_percent"] = N_percent
        if st.button("Get Top N% Agents"):
            if not st.session_state["selected_days"]:
                st.warning("No days selected.")
            else:
                df_act = load_data_for_days(st.session_state["selected_days"])
                if df_act.empty:
                    st.warning("No data found.")
                else:
                    if "FactorizedID" not in df_act.columns:
                        st.warning("No FactorizedID column found.")
                    else:
                        count_df = df_act.groupby("FactorizedID").size().reset_index(name="record_count")
                        count_df = count_df.sort_values("record_count", ascending=False)
                        cutoff = int(np.ceil(len(count_df)*(N_percent/100)))
                        top_n_df = count_df.head(cutoff)
                        st.session_state["top_n_df"] = top_n_df
                        st.write(f"**Total Agents:** {len(count_df)}")
                        st.write(f"**Top {N_percent}% ->** {cutoff} agents")
                        st.dataframe(top_n_df)
                        csv_data = top_n_df.to_csv(index=False).encode("utf-8")
                        st.download_button("📥 Download CSV", csv_data, f"top_{N_percent}_percent_agents.csv", "text/csv")
                del df_act

    # ------------------------------
    # 10. Agents Active Exactly X Days
    # ------------------------------
    st.subheader("🔎 Agents Active for Exactly X Days")

    with st.expander("Query Agents Active in Exactly X (Selected) Days"):
        X_days = st.number_input("X (integer # of days)", min_value=1, value=10)
        if st.button("Find Agents Active X Days"):
            if not st.session_state["selected_days"]:
                st.warning("No days selected.")
            else:
                fid_day_map = {}
                for d in st.session_state["selected_days"]:
                    df_d = pd.read_csv(file_dict[d], usecols=["FactorizedID"])
                    unique_fids = df_d["FactorizedID"].unique()
                    for fid in unique_fids:
                        fid_day_map.setdefault(fid, set()).add(d)

                result = []
                for fid, day_set in fid_day_map.items():
                    if len(day_set) == X_days:
                        result.append({
                            "FactorizedID": fid,
                            "DaysCount": len(day_set),
                            "Days": list(day_set)
                        })
                result_df = pd.DataFrame(result)
                st.write(f"Found {len(result_df)} agents active for exactly {X_days} days (among selected).")
                st.dataframe(result_df)
                csv_data = result_df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download CSV", csv_data, f"agents_active_{X_days}_days.csv", "text/csv")

    # ------------------------------
    # 11. Agents Active in EACH Day in [StartDay, EndDay]
    # ------------------------------
    st.subheader("📅 Agents Active on EVERY Day in a Specified Sub-Range (Selected Days)")

    with st.expander("Query Day Range from the SELECTED Days"):
        if len(st.session_state["selected_days"]) < 2:
            st.info("Need at least 2 selected days to define a range.")
        else:
            sel_sorted = sorted(st.session_state["selected_days"], key=lambda x: int(x))
            start_day_chain = st.selectbox("Start Day", sel_sorted, key="start_day_chain_key")
            end_day_chain = st.selectbox("End Day", sel_sorted, index=len(sel_sorted)-1, key="end_day_chain_key")

            if int(start_day_chain) > int(end_day_chain):
                st.warning("Start day cannot be after end day.")
            else:
                if st.button("Find Agents in Range"):
                    full_range = [str(d) for d in range(int(start_day_chain), int(end_day_chain)+1)]
                    used_days = [d for d in full_range if d in st.session_state["selected_days"]]
                    if not used_days:
                        st.warning("No valid days in that range among selected days.")
                    else:
                        day_fid_sets = []
                        for d in used_days:
                            df_d = pd.read_csv(file_dict[d], usecols=["FactorizedID"])
                            day_fid_sets.append(set(df_d["FactorizedID"].unique()))
                        final_set = set.intersection(*day_fid_sets)
                        df_result = pd.DataFrame({"FactorizedID": list(final_set)})
                        st.write(
                            f"Found {len(df_result)} agents active on EACH day from Day {start_day_chain} to {end_day_chain}."
                        )
                        st.dataframe(df_result)
                        csv_bytes = df_result.to_csv(index=False).encode("utf-8")
                        st.download_button("📥 Download CSV", csv_bytes, f"agents_in_range_{start_day_chain}_{end_day_chain}.csv", "text/csv")

    # ------------------------------
    # 12. Anomaly Detection (Selected Days)
    # ------------------------------
    st.subheader("🚨 Anomaly Detection")

    with st.expander("Potential Mismatches & Duplicates"):
        st.write("Check 'mode vs. link_id' mismatch and exact duplicates among selected days.")
        if st.button("Run Anomaly Detection"):
            if not st.session_state["selected_days"]:
                st.warning("No days selected.")
            else:
                df_ano = load_data_for_days(st.session_state["selected_days"])
                if df_ano.empty:
                    st.warning("No data loaded for selected days.")
                else:
                    mismatch_mask = (df_ano["mode"] == "stationary") & (df_ano["link_id"].notna())
                    mismatch_df = df_ano[mismatch_mask]
                    st.write(f"Mismatched mode/link_id records: {len(mismatch_df)}")
                    if len(mismatch_df) > 0:
                        st.dataframe(mismatch_df.head(200))
                        csv_mismatch = mismatch_df.to_csv(index=False).encode("utf-8")
                        st.download_button("📥 Download Mismatches", csv_mismatch, "mismatch_mode_link.csv", "text/csv")

                    req_cols = ["FactorizedID", "time", "lat", "lng"]
                    missing_cols = [c for c in req_cols if c not in df_ano.columns]
                    if missing_cols:
                        st.warning(f"Cannot detect duplicates. Missing columns: {missing_cols}")
                    else:
                        dups_df = df_ano[df_ano.duplicated(subset=req_cols, keep=False)]
                        st.write(f"Exact duplicates found: {len(dups_df)}")
                        if len(dups_df) > 0:
                            st.dataframe(dups_df.head(200))
                            csv_dups = dups_df.to_csv(index=False).encode("utf-8")
                            st.download_button("📥 Download Duplicates", csv_dups, "exact_duplicates.csv", "text/csv")

                del df_ano

    # ------------------------------
    # 13. Temporal Analysis (Selected Days)
    # ------------------------------
    st.subheader("🕑 Temporal Analysis (Selected Days)")

    with st.expander("Time-based Patterns"):
        if st.button("Run Temporal Analysis"):
            if not st.session_state["selected_days"]:
                st.warning("No days selected.")
            else:
                df_temp = load_data_for_days(st.session_state["selected_days"])
                if df_temp.empty or "time" not in df_temp.columns:
                    st.warning("No data or no 'time' column found.")
                else:
                    df_temp["time"] = pd.to_datetime(df_temp["time"], errors="coerce")

                    df_temp["hour"] = df_temp["time"].dt.hour
                    hour_counts = df_temp["hour"].value_counts().reset_index()
                    hour_counts.columns = ["hour", "count"]
                    hour_counts.sort_values("hour", inplace=True)
                    fig_hour = px.bar(hour_counts, x="hour", y="count", title="Records by Hour of Day")
                    st.plotly_chart(fig_hour, use_container_width=True)

                    df_temp["day_col"] = df_temp["time"].dt.day
                    daily_counts = df_temp.groupby(["FactorizedID","day_col"]).size().reset_index(name="count")
                    fig_daily = px.scatter(daily_counts, x="day_col", y="count",
                                        color="FactorizedID",
                                        title="Daily Record Count per Agent",
                                        hover_data=["FactorizedID"])
                    st.plotly_chart(fig_daily, use_container_width=True)

                    df_temp["TravelFlag"] = np.where(df_temp["mode"]=="stationary", "Stationary", "Travel")
                    trav_counts = df_temp["TravelFlag"].value_counts().reset_index()
                    trav_counts.columns = ["Category","Count"]
                    fig_trav = px.pie(trav_counts, names="Category", values="Count",
                                    title="Stationary vs. Travel")
                    st.plotly_chart(fig_trav, use_container_width=True)

                    df_temp["weekday"] = df_temp["time"].dt.weekday
                    df_temp["is_weekend"] = df_temp["weekday"].isin([5,6])
                    weekend_mode = df_temp.groupby(["is_weekend","mode"]).size().reset_index(name="count")
                    fig_weekend = px.bar(weekend_mode, x="mode", y="count", color="is_weekend",
                                        barmode="group",
                                        title="Mode Usage: Weekend vs. Weekday")
                    st.plotly_chart(fig_weekend, use_container_width=True)

                del df_temp

    # -------------------------------------------------------
    # 14. Spatial Analysis (Selected Days) - Now Day-by-day
    # -------------------------------------------------------
    st.subheader("🗺 Spatial Analysis (Selected Days)")

    if "spatial_loaded" not in st.session_state:
        st.session_state["spatial_loaded"] = False
    if "spatial_day_index" not in st.session_state:
        st.session_state["spatial_day_index"] = 0

    with st.expander("Spatial Patterns"):
        N_CLUSTERS = st.number_input("How many clusters:", value=5, min_value=2)
        if st.button("Run Spatial Analysis"):
            st.session_state["spatial_loaded"] = True
            st.session_state["spatial_day_index"] = 0

        day_list_spatial = sorted(st.session_state["selected_days"], key=lambda x: int(x))
        if st.session_state["spatial_loaded"]:
            if not day_list_spatial:
                st.warning("No days selected for Spatial Analysis.")
            else:
                if st.session_state["spatial_day_index"] >= len(day_list_spatial):
                    st.session_state["spatial_day_index"] = 0

                current_day_spatial = day_list_spatial[st.session_state["spatial_day_index"]]
                df_spatial = load_data_for_day(current_day_spatial)

                if df_spatial.empty or ("lat" not in df_spatial.columns or "lng" not in df_spatial.columns):
                    st.warning(f"No data or missing lat/lng for day {current_day_spatial}.")
                else:
                    st.write(f"**Spatial Analysis for Day {current_day_spatial}**")

                    # 14.1 Unique POI
                    if "poi_id" in df_spatial.columns:
                        unique_pois = df_spatial["poi_id"].dropna().unique()
                        st.write(f"Total Unique POIs: {len(unique_pois)}")
                    else:
                        st.write("No 'poi_id' column found.")

                    # 14.2 Unique home locations
                    if "home_h3_id" in df_spatial.columns:
                        unique_homes = df_spatial["home_h3_id"].dropna().unique()
                        st.write(f"Total Unique Home Locations: {len(unique_homes)}")

                    # 14.3 Top 10 frequent home_h3_id
                    if "home_h3_id" in df_spatial.columns:
                        top_homes = df_spatial["home_h3_id"].value_counts().head(10).reset_index()
                        top_homes.columns = ["home_h3_id", "count"]
                        fig_home = px.bar(top_homes, x="home_h3_id", y="count", 
                                        title="Top 10 Frequent Home h3_ids")
                        st.plotly_chart(fig_home, use_container_width=True)

                    # 14.4 Heatmap
                    fig_heat = px.density_mapbox(
                        df_spatial,
                        lat="lat", lon="lng",
                        radius=5,
                        center={"lat":1.3521,"lon":103.8198},
                        zoom=10,
                        mapbox_style="open-street-map",
                        title="Density Heatmap of Visited Coordinates"
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)

                    # 14.5 Optional KMeans
                    st.write("KMeans Clustering for POIs")
                    try:
                        from sklearn.cluster import KMeans
                        poi_df = df_spatial.dropna(subset=["poi_id"])
                        if not poi_df.empty:
                            coords = poi_df[["lat","lng"]].values
                            kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42).fit(coords)
                            poi_df["cluster"] = kmeans.labels_
                            fig_clus = px.scatter_mapbox(
                                poi_df,
                                lat="lat", lon="lng",
                                color="cluster",
                                center={"lat":1.3521,"lon":103.8198},
                                zoom=10,
                                mapbox_style="open-street-map",
                                title=f"POI Clusters (K={N_CLUSTERS})"
                            )
                            st.plotly_chart(fig_clus, use_container_width=True)
                        else:
                            st.info("No POI records found for clustering.")
                    except ImportError:
                        st.warning("sklearn not installed; cannot run clustering demo.")

                # Navigation
                colL, colR = st.columns([1,1])
                with colL:
                    if st.session_state["spatial_day_index"] > 0:
                        if st.button("⬅️ Previous Day", key="prev_spatial"):
                            st.session_state["spatial_day_index"] -= 1
                with colR:
                    if st.session_state["spatial_day_index"] < len(day_list_spatial)-1:
                        if st.button("➡️ Next Day", key="next_spatial"):
                            st.session_state["spatial_day_index"] += 1
                del df_spatial
        else:
            st.info("Select days and click 'Run Spatial Analysis' to see day-by-day results.")

    # ------------------------------
    # 15. Agent-Level Summary
    # ------------------------------
    st.subheader("👤 Agent-Level Summary (Selected Days)")

    with st.expander("Trips, Time at Home/POI/Transit"):
        st.write(
            "A naive demonstration of deriving trip counts (non-stationary segments) and summarizing time spent."
        )
        if st.button("Generate Agent-Level Summary"):
            if not st.session_state["selected_days"]:
                st.warning("No days selected.")
            else:
                df_combo = load_data_for_days(st.session_state["selected_days"])
                if df_combo.empty:
                    st.warning("No data for selected days.")
                else:
                    if "mode" not in df_combo.columns or "time" not in df_combo.columns:
                        st.warning("Missing 'mode' or 'time' columns for trip analysis.")
                    else:
                        df_combo["time"] = pd.to_datetime(df_combo["time"], errors="coerce")
                        travel_df = df_combo[df_combo["mode"] != "stationary"].copy()
                        travel_df.sort_values(["FactorizedID","time"], inplace=True)
                        travel_df["time_shift"] = travel_df.groupby("FactorizedID")["time"].shift(1)
                        travel_df["gap_sec"] = (travel_df["time"] - travel_df["time_shift"]).dt.total_seconds()
                        travel_df["new_trip"] = (travel_df["gap_sec"]>1800) | travel_df["gap_sec"].isna()
                        travel_df["trip_id"] = travel_df.groupby("FactorizedID")["new_trip"].cumsum()

                        trip_counts = travel_df.groupby("FactorizedID")["trip_id"].nunique().reset_index()
                        trip_counts.columns = ["FactorizedID","trip_count"]

                        def categorize(row):
                            if row.get("is_home") == True and row.get("mode") == "stationary":
                                return "Home"
                            elif row.get("mode") == "stationary":
                                return "POI"
                            else:
                                return "Transit"

                        df_combo["LocationType"] = df_combo.apply(categorize, axis=1)
                        loc_counts = df_combo.groupby(["FactorizedID","LocationType"]).size().reset_index(name="count")
                        loc_pivot = loc_counts.pivot(index="FactorizedID", columns="LocationType", values="count").fillna(0)

                        summary_df = trip_counts.merge(loc_pivot, how="outer", on="FactorizedID").fillna(0)
                        st.dataframe(summary_df.head(50))

                        csv_sum = summary_df.to_csv(index=False).encode("utf-8")
                        st.download_button("📥 Download Summary CSV", csv_sum, "agent_level_summary.csv", "text/csv")

                    del df_combo

# -------------------------------------------------------
# 16. Stay Points & Travel Chains (Only Time-Based)
# -------------------------------------------------------
    st.subheader("🔗 Stay Points & Travel Chains (Time-Based Only)")

    with st.expander("Extract Detailed Activity Chain", expanded=st.session_state["expander_chain"]):

        # Show available FactorizedIDs in selected days
        if st.session_state["selected_days"]:
            all_fids = []
            for d in st.session_state["selected_days"]:
                tmp_df = pd.read_csv(file_dict[d], usecols=["FactorizedID"])
                all_fids.extend(tmp_df["FactorizedID"].unique())
            if all_fids:
                fmin, fmax = min(all_fids), max(all_fids)
                st.write(f"**Valid FactorizedID Range (Selected Days):** {fmin} — {fmax}")
            else:
                st.warning("No FactorizedIDs found among selected days.")
        else:
            st.info("No days selected. FactorizedID range unavailable.")

        # Radio choice
        chain_range_choice = st.radio(
            "Choose Extraction Mode",
            ["Single Agent; Single Day", "Single Agent; Date Range", "All Agents; Date Range"],
            key="chain_radio2"
        )

        # Input FactorizedID for single agent modes
        chain_fac_id = st.number_input("FactorizedID for chain analysis", min_value=0, value=0)

        # POI label selection
        use_name = st.checkbox("Use 'name' for POI label", value=True)
        use_class = st.checkbox("Use 'class' for POI label", value=True)
        use_category = st.checkbox("Use 'category' for POI label", value=False)

        # Helper: Format durations
        def format_hms(seconds):
            if pd.isna(seconds) or seconds < 0:
                return "0s"
            h, rem = divmod(int(seconds), 3600)
            m, s = divmod(rem, 60)
            return " ".join(f"{v}{l}" for v, l in zip([h, m, s], ["h", "m", "s"]) if v) or "0s"

        # Chain Extraction Function
        def extract_timebased_chain(df):
            if df.empty:
                return pd.DataFrame(columns=["StartTime", "EndTime", "EventType", "Label", "Duration(h/m/s)"])

            df = df.copy()
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            df.sort_values("time", inplace=True)
            df.reset_index(drop=True, inplace=True)

            def get_type_label(row):
                if row["mode"] == "stationary":
                    label_parts = []
                    if row.get("is_home") == True:
                        return "Stay", "Home"
                    if use_name:
                        label_parts.append(str(row.get("name", "UnknownName")))
                    if use_class:
                        label_parts.append(str(row.get("class", "UnknownClass")))
                    if use_category:
                        label_parts.append(str(row.get("category", "UnknownCategory")))
                    if not label_parts:
                        label_parts = ["POI"]
                    return "Stay", "_".join(label_parts)
                else:
                    return "Travel", row["mode"]

            events = []
            cur_type, cur_label, start_time = None, None, None

            for i, row in df.iterrows():
                etype, lbl = get_type_label(row)
                t = row["time"]
                if cur_type is None:
                    cur_type, cur_label, start_time = etype, lbl, t
                elif etype == cur_type and lbl == cur_label:
                    continue
                else:
                    end_time = df.iloc[i-1]['time']
                    duration_s = (end_time - start_time).total_seconds() if pd.notnull(end_time) else 0
                    events.append({
                        "agent_id": row["agent_id"],
                        "StartTime": start_time,
                        "EndTime": end_time,
                        "EventType": cur_type,
                        "Label": cur_label,
                        "Duration(h/m/s)": format_hms(duration_s)
                    })
                    cur_type, cur_label, start_time = etype, lbl, t

            if cur_type and start_time is not None:
                end_time = df.iloc[-1]["time"]
                duration_s = (end_time - start_time).total_seconds() if pd.notnull(end_time) else 0
                events.append({
                    "agent_id": row["agent_id"],
                    "StartTime": start_time,
                    "EndTime": end_time,
                    "EventType": cur_type,
                    "Label": cur_label,
                    "Duration(h/m/s)": format_hms(duration_s)
                })

            chain_df = pd.DataFrame(events)
            chain_df.sort_values("StartTime", inplace=True)
            chain_df.reset_index(drop=True, inplace=True)
            return chain_df

        # Helper to load days
        def load_days(day_list):
            dfs = []
            for d in day_list:
                try:
                    tmp = pd.read_csv(file_dict[d])
                    dfs.append(tmp)
                except Exception as e:
                    st.error(f"Error loading day {d}: {e}")
            return pd.concat(dfs) if dfs else pd.DataFrame()

        # Main runner
        def run_chain_extraction(day_list, fid=None, output_dir=None, restrict_to_top=False, custom_fids=None):
            """
            day_list: e.g., ["1","2","3"]
            fid: single FactorizedID (for single-agent mode)
            output_dir: folder to store CSV if extracting for all
            restrict_to_top: bool => only topN or all
            custom_fids: optional set/list of FactorizedID from the uploaded file
            """
            df_all = load_days(day_list)
            if df_all.empty:
                st.warning("No data found for the selected days.")
                return

            if fid is not None:
                # Single agent
                df_agent = df_all[df_all["FactorizedID"] == fid]
                if df_agent.empty:
                    st.info("No data for that FactorizedID.")
                    return
                chain_df = extract_timebased_chain(df_agent)
                st.dataframe(chain_df)
                csv_time = chain_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Time-Based Chain CSV",
                    csv_time,
                    f"chain_timebased_{fid}.csv",
                    "text/csv"
                )
            else:
                # All agents mode
                os.makedirs(output_dir, exist_ok=True)

                # if topN => use st.session_state["top_n_df"] factorized IDs
                if restrict_to_top and "top_n_df" in st.session_state:
                    agent_ids = st.session_state["top_n_df"]["FactorizedID"].unique()
                    st.success(f"✅ Using {len(agent_ids)} Top N% agents.")
                else:
                    agent_ids = df_all["FactorizedID"].unique()
                    st.info(f"ℹ️ Using ALL {len(agent_ids)} agents.")

                # If user uploaded custom factorizedIDs => intersect
                if custom_fids is not None:
                    old_count = len(agent_ids)
                    agent_ids = set(agent_ids).intersection(set(custom_fids))
                    st.info(f"Custom FactorizedIDs used. Original: {old_count}, after intersection: {len(agent_ids)}")

                total_agents = len(agent_ids)
                if total_agents == 0:
                    st.warning("No valid FactorizedIDs left after applying filters.")
                    return

                progress_bar = st.progress(0)
                for i, _fid in enumerate(agent_ids, start=1):
                    df_agent = df_all[df_all["FactorizedID"] == _fid]
                    chain_df = extract_timebased_chain(df_agent)
                    out_csv = os.path.join(output_dir, f"{_fid}_activity_chain.csv")
                    chain_df.to_csv(out_csv, index=False)
                    progress_bar.progress(i / total_agents)
                progress_bar.empty()
                st.success(f"✅ All agents' chains saved in '{output_dir}'.")

        # =============== UI Logic ===============

        if chain_range_choice == "Single Agent; Single Day":
            day_for_chain = st.selectbox("Select 1 Day", st.session_state["selected_days"], key="chain_single_day")
            if st.button("Extract Chain for Selected Day"):
                run_chain_extraction([day_for_chain], fid=chain_fac_id)
                st.session_state["expander_chain"] = True

        elif chain_range_choice == "Single Agent; Date Range":
            if len(st.session_state["selected_days"]) < 1:
                st.warning("Select at least 2 days.")
            else:
                sorted_days = sorted(st.session_state["selected_days"], key=int)
                sd = st.selectbox("Start Day", sorted_days, key="chain_range_start")
                ed = st.selectbox("End Day", sorted_days, index=len(sorted_days)-1, key="chain_range_end")
                if int(sd) > int(ed):
                    st.warning("Start day cannot be after end day.")
                else:
                    if st.button("Extract Chain for Selected Date Range"):
                        day_range = [str(d) for d in range(int(sd), int(ed)+1) if str(d) in st.session_state["selected_days"]]
                        run_chain_extraction(day_range, fid=chain_fac_id)
                        st.session_state["expander_chain"] = True

        else:
            # All Agents mode
            st.write("⚡ Extract time-based chains for ALL agents over a date range.")
            output_folder = st.text_input("Destination Folder:", value="dataset/1000_users_chain_new")
            use_top_n_agents = st.checkbox("✅ Only use Top N% agents", value=False)

            # New: file uploader for custom FactorizedID
            custom_fid_file = st.file_uploader("Upload a CSV containing FactorizedID if you only want those agents (optional)", type=["csv"])
            custom_fids = None
            if custom_fid_file is not None:
                df_custom = pd.read_csv(custom_fid_file)
                if "FactorizedID" not in df_custom.columns:
                    st.error("The uploaded file must have a column named 'FactorizedID'.")
                else:
                    custom_fids = df_custom["FactorizedID"].unique()
                    st.info(f"Uploaded {len(custom_fids)} FactorizedIDs from file.")
            
            if len(st.session_state["selected_days"]) < 1:
                st.warning("Select at least 2 days.")
            else:
                sorted_days = sorted(st.session_state["selected_days"], key=int)
                sd = st.selectbox("Start Day", sorted_days, key="chain_range_start_all")
                ed = st.selectbox("End Day", sorted_days, index=len(sorted_days)-1, key="chain_range_end_all")
                if int(sd) > int(ed):
                    st.warning("Start day cannot be after end day.")
                else:
                    if st.button("Extract Chains for ALL Agents"):
                        day_range = [str(d) for d in range(int(sd), int(ed)+1) if str(d) in st.session_state["selected_days"]]
                        run_chain_extraction(
                            day_range,
                            fid=None,
                            output_dir=output_folder,
                            restrict_to_top=use_top_n_agents,
                            custom_fids=custom_fids
                        )
                        st.session_state["expander_chain"] = True
    #--------------------------------------------------------
    # 17. Factorized ID to Device ID
    #--------------------------------------------------------
    st.subheader("🔍 Retrieve Device ID from Factorized ID")

    with st.expander("Factorized ID → Device ID Lookup"):
        valid_factorized_ids = set()

        if st.session_state["selected_days"]:
            # Efficiently gather unique FactorizedIDs
            for day in st.session_state["selected_days"]:
                df_day = pd.read_csv(file_dict[day], usecols=['FactorizedID'])
                valid_factorized_ids.update(df_day['FactorizedID'].unique())

            if valid_factorized_ids:
                fmin, fmax = min(valid_factorized_ids), max(valid_factorized_ids)
                st.info(f"Valid FactorizedID Range (selected days): **{fmin} – {fmax}**")
            else:
                st.warning("⚠️ No FactorizedIDs found in the selected days.")
        else:
            st.warning("⚠️ Please select at least one day to enable retrieval.")

        fac_id = st.number_input(
            "Enter Factorized ID:",
            min_value=0,
            value=0,
            step=1
        )

        # Load mapping just once, cache for performance
        @st.cache_data
        def load_agent_id_mapping():
            mapping_path = os.path.join(DATA_FOLDER, "agent_id_mapping.csv")
            return pd.read_csv(mapping_path)

        mapping_df = load_agent_id_mapping()

        def retrieve_device_id(fid):
            result = mapping_df[mapping_df['FactorizedID'] == fid]['agent_id']
            return result.values[0] if not result.empty else None

        # Disable the button if no valid IDs are found
        retrieve_disabled = len(valid_factorized_ids) == 0 or fac_id not in valid_factorized_ids

        if st.button("🔎 Retrieve Device ID", disabled=retrieve_disabled):
            device_id = retrieve_device_id(fac_id)
            if device_id:
                st.success(f"**Device ID for FactorizedID {fac_id}:**")
                st.text_area("Device ID:", value=device_id, height=80)
            else:
                st.error(f"❌ No Device ID found for FactorizedID {fac_id}.")

    #--------------------------------------------------------
    # 18. POI Information
    #--------------------------------------------------------
    st.subheader("🔍 POI Details")

    with st.expander("Details about Point of Interets"):
        df = pd.read_csv(POI_INFO, index_col=None)
        poi_types = df['class'].unique()
        poi_names = df['name'].unique()
        poi_cats = df['category'].unique()
        if st.button("POI Stats."):
            

            st.success(f"Number of POI **types**: **{poi_types.shape[0]}**")
            st.write(f"Number of POI unique **categories**: **{poi_cats.shape[0]}**")
            st.write(f"{poi_cats.tolist()}")
            st.write(f"Number of POI unique **names**: **{poi_names.shape[0]}**")
        
        if st.button("Display and Store POI Types"):
            st.text_area("POI Types: ", value=f"{poi_types.tolist()}")
            df_types = pd.DataFrame(poi_types, columns=['Type'], index=None)
            df_types.to_csv('dataset/poi_types.csv', index=False)


    st.write("---")
    st.write("**End of Dashboard**")

with tab3:
    ############################################
    ##  IRL Tab 2:  
    ############################################
    import time
    from irl_module import (
            setup_logger,
            parse_csv_to_demo,
            MaxCausalEntIRL
        )
    st.write("Inverse Reinforcement Learning Model")

    # 1) Read the top-N% DataFrame from Tab 1, if it exists
    if "top_n_df" not in st.session_state:
        st.warning("No Top N% Agents found. Please go to Tab 1 and run 'Get Top N% Agents'.")
    else:
        st.write("Top N% Agents (from Tab 1):")
        st.dataframe(st.session_state["top_n_df"])
        #top_n_df = st.session_state["top_n_df"]
        

        # Let user select which agents to train:
        selected_agents = st.multiselect(
            "Which Agents to run IRL on?",
            options=st.session_state["top_n_df"]["FactorizedID"].tolist()
        )

        # 2) Let user specify time-based chain folder (where your chain CSVs live)
        chain_folder = st.text_input("Folder containing each agent's time-based chain CSV:", "dataset/timebased_chain")

        # 3) IRL hyperparameters
        st.write("**IRL Training Settings**")
        lr = st.number_input("Learning rate", value=5e-4)
        iterations = st.number_input("Number of iterations", value=300, step=50)
        gamma = st.number_input("Discount factor (gamma)", value=0.97, step=0.01)
        l2_reg = st.number_input("L2 regularization", value=1e-3, format="%.1e")

        out_folder = st.text_input("Output folder for IRL logs/CSVs:", "irl_output")
        os.makedirs(out_folder, exist_ok=True)

        # 4) Pause/Resume logic
        if "irl_pause" not in st.session_state:
            st.session_state["irl_pause"] = False

        colPAUSE, colRESUME = st.columns(2)
        with colPAUSE:
            if st.button("Pause Training"):
                st.session_state["irl_pause"] = True
        with colRESUME:
            if st.button("Resume Training"):
                st.session_state["irl_pause"] = False

        # 5) Real-time line charts for mismatch, NLL, etc.
        mismatch_placeholder = st.empty()
        nll_placeholder = st.empty()

        # We store metrics in a DataFrame in session_state for live updates
        if "irl_metrics_df" not in st.session_state:
            st.session_state["irl_metrics_df"] = pd.DataFrame(columns=["agent","iteration","mismatch","nll"])
        
        # 6) Press "Train IRL" to start
        if st.button("Train IRL"):
            if not selected_agents:
                st.warning("No agents selected. Please select at least one above.")
            else:
                progress_bar = st.progress(0)
                num_agents = len(selected_agents)
                st.write(f"Starting IRL on {num_agents} agent(s)...")

                for idx, fid in enumerate(selected_agents, start=1):
                    st.write(f"**Agent {fid}** ...")
                    chain_csv = os.path.join(chain_folder, f"{fid}_activity_chain.csv")
                    if not os.path.isfile(chain_csv):
                        st.warning(f"No chain CSV for agent {fid} at {chain_csv}. Skipping.")
                        continue

                    # Setup logger
                    log_file = os.path.join(out_folder, f"{fid}_irl.log")
                    logger = setup_logger(log_file, f"IRL_{fid}")

                    # Parse chain -> episodes
                    demo_episodes = parse_csv_to_demo(chain_csv, logger)
                    if not demo_episodes:
                        st.warning(f"No demonstration steps found for agent {fid}. Skipping.")
                        continue

                    # Build IRL object
                    irl = MaxCausalEntIRL(
                        demo_episodes=demo_episodes,
                        logger=logger,
                        lr=lr,
                        iterations=iterations,
                        gamma=gamma,
                        horizon=50,  # not used in episodic approach
                        l2_reg=l2_reg
                    )

                    # We do an iteration loop ourselves, calling fit_one_iteration each time
                    for it in range(int(iterations)):
                        # Check if paused
                        while st.session_state["irl_pause"]:
                            st.warning("Training paused. Click 'Resume Training' to continue.")
                            st.stop()  # halts until next run

                        # One iteration
                        metrics = irl.fit_one_iteration(it)
                        # We can store them in session_state IRL metrics
                        # st.session_state["irl_metrics_df"] = st.session_state["irl_metrics_df"].append({
                        #     "agent": fid,
                        #     "iteration": metrics["iteration"],
                        #     "mismatch": metrics["feature_mismatch"],
                        #     "nll": metrics["neg_log_likelihood"]
                        # }, ignore_index=True)
                        #s = type(st.session_state["irl_metrics_df"])
                        #st.info(f"{s}")
                        
                        new_df = pd.DataFrame([{
                            "agent": fid,
                            "iteration": metrics["iteration"],
                            "mismatch": metrics["feature_mismatch"],
                            "nll": metrics["neg_log_likelihood"]
                        }])
                        st.session_state["irl_metrics_df"] = st.session_state["irl_metrics_df"].concat(new_df,
                                                                                                       ignore_index=True)

                        # Live-plot mismatch vs iteration, NLL vs iteration
                        # Filter agent
                        agent_df = st.session_state["irl_metrics_df"].query("agent == @fid")
                        mismatch_placeholder.line_chart(
                            agent_df[["mismatch"]].reset_index(drop=True),
                            height=200
                        )
                        nll_placeholder.line_chart(
                            agent_df[["nll"]].reset_index(drop=True),
                            height=200
                        )

                    # finalize weights
                    irl.finalize_weights()

                    # Save final weights to CSV
                    w_path = os.path.join(out_folder, f"{fid}_weights.csv")
                    df_w = pd.DataFrame(irl.w, columns=["weight"])
                    df_w.to_csv(w_path, index=False)

                    st.success(f"Agent {fid} training done. Weights saved to {w_path}.")

                    progress_bar.progress(int(idx / num_agents * 100))

                st.success("All IRL training done.")
                progress_bar.empty()
                

# -------------------------------------
# IRL Tab 4: Agent-by-Agent Analysis
# -------------------------------------

    with tab4:
    # ────────────────────────────────────────────────
    # 1. Folder layout (NEW)
    # ────────────────────────────────────────────────
        BASE_10U_FOLDER = "../IRL_Behavioral_Cloning/output/10_users"

        # categorical (non‑time) results
        OUTPUT_FOLDER       = os.path.join(BASE_10U_FOLDER, "before", "cat")
        COMPARE_FOLDER      = os.path.join(BASE_10U_FOLDER, "After",  "cat")

        # time‑based results
        OUTPUT_FOLDER_Time  = os.path.join(BASE_10U_FOLDER, "before", "time")
        COMPARE_FOLDER_Time = os.path.join(BASE_10U_FOLDER, "After",  "time")

        # agent mapping
        mapping_path = os.path.join(BASE_10U_FOLDER, "agent_mapping.csv")

        # ────────────────────────────────────────────────
        # 2. Load mapping (FactorizedID ↔ agent_id)
        # ────────────────────────────────────────────────
        try:
            mapping_df = pd.read_csv(mapping_path)
        except Exception as e:
            st.error(f"Error loading agent_mapping.csv: {e}")
            mapping_df = pd.DataFrame(columns=["FactorizedID", "agent_id"])

        def get_agent_id_from_fid(fid: str) -> str:
            try:
                row = mapping_df[mapping_df["FactorizedID"] == int(fid)]
                return row.iloc[0]["agent_id"] if not row.empty else "Unknown"
            except Exception:
                return "Unknown"

        # ────────────────────────────────────────────────
        # 3. Which FactorizedIDs are available?
        # ────────────────────────────────────────────────
        available_fids = sorted([
            f for f in os.listdir(OUTPUT_FOLDER)
            if os.path.isdir(os.path.join(OUTPUT_FOLDER, f)) and f.isdigit()
        ])

        if "current_irl_agent_index" not in st.session_state:
            st.session_state["current_irl_agent_index"] = 0

        # ────────────────────────────────────────────────
        # 4. Navigation widgets
        # ────────────────────────────────────────────────
        col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 2])

        with col_nav1:
            if st.button("⬅️ Previous Agent") and st.session_state["current_irl_agent_index"] > 0:
                st.session_state["current_irl_agent_index"] -= 1
        with col_nav2:
            if st.button("➡️ Next Agent") and st.session_state["current_irl_agent_index"] < len(available_fids) - 1:
                st.session_state["current_irl_agent_index"] += 1
        with col_nav3:
            manual_fid = st.text_input("🔎 Jump to FactorizedID:", value="")
            if manual_fid.strip().isdigit() and manual_fid.strip() in available_fids:
                st.session_state["current_irl_agent_index"] = available_fids.index(manual_fid.strip())

        st.write("---")
        compare_check = st.checkbox("🔁 Compare with After (CB)")

        # optional helper
        def normalize_to_minus1_plus1(arr):
            arr = np.asarray(arr, dtype=float)
            mn, mx = arr.min(), arr.max()
            return np.zeros_like(arr) if mx == mn else 2 * (arr - mn) / (mx - mn) - 1

        # ────────────────────────────────────────────────
        # 5. Main display
        # ────────────────────────────────────────────────
        if not available_fids:
            st.warning("No IRL output folders found.")
            st.stop()

        current_fid = available_fids[st.session_state["current_irl_agent_index"]]
        agent_id    = get_agent_id_from_fid(current_fid)
        st.info(f"**FactorizedID:** `{current_fid}`   |   **Agent ID:** `{agent_id}`")

        # ---------- file paths (CAT) ----------
        fb           = os.path.join(OUTPUT_FOLDER, current_fid)
        feat_before  = os.path.join(fb, "empirical_feats.csv")
        wght_before  = os.path.join(fb, "weights.csv")
        met_before   = os.path.join(fb, "training_metrics.csv")  # optional

        # ---------- file paths (TIME) ----------
        fbt           = os.path.join(OUTPUT_FOLDER_Time, current_fid)
        feat_before_t = os.path.join(fbt, "empirical_feats.csv")
        wght_before_t = os.path.join(fbt, "weights.csv")
        met_before_t  = os.path.join(fbt, "training_metrics.csv")  # optional

        # optionally compare
        if compare_check:
            fa            = os.path.join(COMPARE_FOLDER, current_fid)
            feat_after    = os.path.join(fa, "empirical_feats.csv")
            wght_after    = os.path.join(fa, "weights.csv")
            met_after     = os.path.join(fa, "training_metrics.csv")

            fat           = os.path.join(COMPARE_FOLDER_Time, current_fid)
            feat_after_t  = os.path.join(fat, "empirical_feats.csv")
            wght_after_t  = os.path.join(fat, "weights.csv")
            met_after_t   = os.path.join(fat, "training_metrics.csv")

        # ────────────────────────────────────────────────
        # 6. Labels & indices (unchanged)
        # ────────────────────────────────────────────────
        loc_feat_labels = [
            "Home", "Comm & Gov", "Arts & Ent", "Landmarks & Outdr",
            "Travel & Transp", "Dining & Drink", "Business & Prof Serv", "Sports & Rec",
            "Retail", "Health & Med", "Event", "Education"
        ]
        mode_feat_labels = ["walk", "cycle", "car", "bus", "mrt"]
        loc_feat_time_labels  = [f"Time:{l}" for l in loc_feat_labels]
        mode_feat_time_labels = ["Time:" + m for m in mode_feat_labels]

        loc_indices  = list(range(12))   # 0‑11
        mode_indices = list(range(12, 17))
        loc_time_indices  = list(range(12))
        mode_time_indices = list(range(12, 17))

        # ────────────────────────────────────────────────
        # 7. Load CSVs & plot
        # ────────────────────────────────────────────────
        try:
            # ---- BEFORE ----
            df_feat_b  = pd.read_csv(feat_before)
            df_wght_b  = pd.read_csv(wght_before)

            df_feat_b_t = pd.read_csv(feat_before_t)
            df_wght_b_t = pd.read_csv(wght_before_t)

            # ---- AFTER (optional) ----
            if compare_check:
                df_feat_a  = pd.read_csv(feat_after)
                df_wght_a  = pd.read_csv(wght_after)

                df_feat_a_t = pd.read_csv(feat_after_t)
                df_wght_a_t = pd.read_csv(wght_after_t)

            # ──────────────────────────
            # A) Non‑time  • Empirical & Weights
            # ──────────────────────────
            st.subheader("🔹 Non‑Time (IRL) – Empirical Features & Learned Weights")
            col1, col2 = st.columns(2)

            # --- 1. location empirical ---
            with col1:
                if compare_check:
                    df = pd.DataFrame({
                        "Feature": loc_feat_labels,
                        "Before": df_feat_b.iloc[0, loc_indices].values,
                        "After":  df_feat_a.iloc[0, loc_indices].values
                    })
                    fig = px.bar(df.melt("Feature", var_name="Period", value_name="Value"),
                                x="Feature", y="Value", color="Period",
                                title="Empirical (Location) – Before vs After",
                                barmode="group")
                else:
                    fig = px.bar(x=loc_feat_labels,
                                y=df_feat_b.iloc[0, loc_indices].values,
                                labels={"x":"Feature","y":"Value"},
                                title="Empirical (Location) – Before")
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

            # --- 2. location weights ---
            with col2:
                if compare_check:
                    df = pd.DataFrame({
                        "Feature": loc_feat_labels,
                        "Before": df_wght_b.iloc[-1, loc_indices].values,
                        "After":  df_wght_a.iloc[-1, loc_indices].values
                    })
                    fig = px.bar(df.melt("Feature", var_name="Period", value_name="Weight"),
                                x="Feature", y="Weight", color="Period",
                                title="Weights (Location) – Before vs After",
                                barmode="group")
                else:
                    fig = px.bar(x=loc_feat_labels,
                                y=df_wght_b.iloc[-1, loc_indices].values,
                                labels={"x":"Feature","y":"Weight"},
                                title="Weights (Location) – Before")
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

            # --- MODE‑based (empirical & weight) ---
            st.write("---")
            col3, col4 = st.columns(2)

            with col3:
                if compare_check:
                    df = pd.DataFrame({
                        "Feature": mode_feat_labels,
                        "Before": df_feat_b.iloc[0, mode_indices].values,
                        "After":  df_feat_a.iloc[0, mode_indices].values
                    })
                    fig = px.bar(df.melt("Feature", var_name="Period", value_name="Value"),
                                x="Feature", y="Value", color="Period",
                                title="Empirical (Mode) – Before vs After",
                                barmode="group")
                else:
                    fig = px.bar(x=mode_feat_labels,
                                y=df_feat_b.iloc[0, mode_indices].values,
                                labels={"x":"Feature","y":"Value"},
                                title="Empirical (Mode) – Before")
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

            with col4:
                if compare_check:
                    df = pd.DataFrame({
                        "Feature": mode_feat_labels,
                        "Before": df_wght_b.iloc[-1, mode_indices].values,
                        "After":  df_wght_a.iloc[-1, mode_indices].values
                    })
                    fig = px.bar(df.melt("Feature", var_name="Period", value_name="Weight"),
                                x="Feature", y="Weight", color="Period",
                                title="Weights (Mode) – Before vs After",
                                barmode="group")
                else:
                    fig = px.bar(x=mode_feat_labels,
                                y=df_wght_b.iloc[-1, mode_indices].values,
                                labels={"x":"Feature","y":"Weight"},
                                title="Weights (Mode) – Before")
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

            # ──────────────────────────
            # B) Time‑based  • Empirical & Weights
            # ──────────────────────────
            st.write("---")
            st.subheader("🔹 Time‑Based (IRL) – Empirical & Weights")

            tf_b = df_feat_b_t.iloc[-1, :17].values   # 17 feature cols + feature_set
            tw_b = df_wght_b_t.iloc[-1, :17].values
            if compare_check:
                tf_a = df_feat_a_t.iloc[-1, :17].values
                tw_a = df_wght_a_t.iloc[-1, :17].values

            # ---- Location‑time ----
            colT1, colT2 = st.columns(2)
            with colT1:
                if compare_check:
                    df = pd.DataFrame({
                        "Feature": loc_feat_time_labels,
                        "Before": tf_b[loc_time_indices],
                        "After":  tf_a[loc_time_indices]
                    })
                    fig = px.bar(df.melt("Feature", var_name="Period", value_name="Value"),
                                x="Feature", y="Value", color="Period",
                                title="Time Empirical (Location) – B vs A",
                                barmode="group")
                else:
                    fig = px.bar(x=loc_feat_time_labels, y=tf_b[loc_time_indices],
                                title="Time Empirical (Location) – Before")
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

            with colT2:
                if compare_check:
                    df = pd.DataFrame({
                        "Feature": loc_feat_time_labels,
                        "Before": tw_b[loc_time_indices],
                        "After":  tw_a[loc_time_indices]
                    })
                    fig = px.bar(df.melt("Feature", var_name="Period", value_name="Weight"),
                                x="Feature", y="Weight", color="Period",
                                title="Time Weights (Location) – B vs A",
                                barmode="group")
                else:
                    fig = px.bar(x=loc_feat_time_labels, y=tw_b[loc_time_indices],
                                title="Time Weights (Location) – Before")
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

            # ---- Mode‑time ----
            colT3, colT4 = st.columns(2)
            with colT3:
                if compare_check:
                    df = pd.DataFrame({
                        "Feature": mode_feat_time_labels,
                        "Before": tf_b[mode_time_indices],
                        "After":  tf_a[mode_time_indices]
                    })
                    fig = px.bar(df.melt("Feature", var_name="Period", value_name="Value"),
                                x="Feature", y="Value", color="Period",
                                title="Time Empirical (Mode) – B vs A",
                                barmode="group")
                else:
                    fig = px.bar(x=mode_feat_time_labels, y=tf_b[mode_time_indices],
                                title="Time Empirical (Mode) – Before")
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

            with colT4:
                if compare_check:
                    df = pd.DataFrame({
                        "Feature": mode_feat_time_labels,
                        "Before": tw_b[mode_time_indices],
                        "After":  tw_a[mode_time_indices]
                    })
                    fig = px.bar(df.melt("Feature", var_name="Period", value_name="Weight"),
                                x="Feature", y="Weight", color="Period",
                                title="Time Weights (Mode) – B vs A",
                                barmode="group")
                else:
                    fig = px.bar(x=mode_feat_time_labels, y=tw_b[mode_time_indices],
                                title="Time Weights (Mode) – Before")
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Failed to load IRL result files for `{current_fid}`: {e}")