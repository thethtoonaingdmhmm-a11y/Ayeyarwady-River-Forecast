# ==============================================================================
# 📦 1. REQUIRED LIBRARIES IMPORT
# ==============================================================================
import os
import datetime
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import streamlit as st
import time
import json
import gspread 
from google.oauth2.service_account import Credentials
from tensorflow.keras.models import load_model

# ==============================================================================
# 🌟 SYSTEM INITIALIZATION (ကုဒ်၏ အပေါ်ဆုံးတွင် ထားရန်)
# ==============================================================================
if 'base_date' not in st.session_state:
    st.session_state.base_date = datetime.date.today()

if 'ts_extended' not in st.session_state:
    st.session_state.ts_extended = None

if 'upload_done' not in st.session_state:
    st.session_state.upload_done = False

# Plotly Import
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# TensorFlow Warnings များကို UI ထဲမရောက်အောင် ပိတ်ခြင်း
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(page_title="DMH AI Flood Dashboard", layout="wide")

# ==============================================================================
# 🌐 2. DYNAMIC PATH ENVIRONMENT GUARD & GOOGLE SHEETS SETUP
# ==============================================================================
base_models_dir = '.'
meta_path = 'Station_Meta.csv'
csv_path = 'Station_Data.csv'
creds_json_path = 'google_creds.json'
SHEET_ID = '1kj8vblfPnBCbfJJ5fJTXH-OUY9ZIzZe5qsnaTwlYS4o'
GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/1kj8vb1fPnBCbfJJ5fJTXH-OUY9ZIzZe5qsnaTwlYS4o/edit?gid=0#gid=0"

if not os.path.exists(meta_path) and os.path.exists('station_Meta.csv'): 
    meta_path = 'station_Meta.csv'
if not os.path.exists(csv_path) and os.path.exists('station_Data.csv'): 
    csv_path = 'station_Data.csv'

# 💡 [GLOBAL DATA LOAD] မက်တာဒေတာကို ကြိုတင်ဖတ်ထားခြင်း
if os.path.exists(meta_path):
    meta_df = pd.read_csv(meta_path)
else:
    st.error(f"❌ '{meta_path}' ဖိုင်ကို ရှာမတွေ့ပါ။ ကျေးဇူးပြု၍ ဖိုင်အမည်ကို စစ်ဆေးပေးပါ။")
    st.stop()

stations_list = [
    'Myitkyina', 'Mandalay', 'Sagaing', 'Myinmu', 'Pakokku', 'NyaungOo', 
    'Chauk', 'Minbu', 'Magway', 'Aunglan', 'Pyay', 
    'Seiktha', 'Hinthada', 'Zalun'
]

# ==============================================================================
# 🌐 DATA LOADER & AUTO-SYNC FUNCTIONS (STREAMLIT SECRETS SUPPORTED)
# ==============================================================================
def load_data_from_sheets(url=None):
    try:
        if 'uploaded_df' in st.session_state and st.session_state.uploaded_df is not None:
            df = st.session_state.uploaded_df.copy()
        elif os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            return None
            
        # 💡 ရက်စွဲ Format မှားယူမှု မဖြစ်အောင် format='mixed' အသုံးပြုထားပါသည်
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
        df = df.dropna(subset=['Date'])

        for col in df.columns:
            if col != 'Date':
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if not df.empty:
            max_live_date = df['Date'].max()
            cutoff_live_date = max_live_date - pd.Timedelta(days=90)
            df = df[df['Date'] >= cutoff_live_date].sort_values('Date').reset_index(drop=True)
            
        return df
        
    except Exception as e:
        st.warning(f"⚠️ Local CSV ဒေတာ ဖတ်၍မရသေးပါ - {e}")
        return None

def fetch_and_sync_google_sheet():
    df = load_data_from_sheets()
    if df is not None and not df.empty:
        st.session_state.base_date = pd.to_datetime(df['Date']).max().date()
        return df
    else:
        if 'base_date' not in st.session_state or st.session_state.base_date is None:
            st.session_state.base_date = datetime.date.today()
        if os.path.exists(csv_path):
            df_local = pd.read_csv(csv_path)
            df_local['Date'] = pd.to_datetime(df_local['Date'], format='mixed', errors='coerce')
            return df_local
        return None

def save_data_to_sheets_and_cloud(df):
    status_placeholder = st.empty()
    try:
        status_placeholder.info("⏳ ဒေတာများကို မူရင်း Google Sheet နှင့်ပေါင်းစပ်ပြီး Cloud ပေါ်သို့ ပို့နေပါသည်...")
        
        df_to_save = df.copy()
        df_to_save['Date'] = pd.to_datetime(df_to_save['Date']).dt.strftime('%Y-%m-%d')
        df_to_save.to_csv(csv_path, index=False)
        
        scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = None
        
        # 💡 1. Streamlit Cloud Secrets စစ်ဆေးခြင်း
        if "gcp_service_account" in st.secrets:
            creds_dict = dict(st.secrets["gcp_service_account"])
            creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
        # 💡 2. Local google_creds.json ဖိုင် စစ်ဆေးခြင်း
        elif os.path.exists(creds_json_path):
            creds = Credentials.from_service_account_file(creds_json_path, scopes=scope)
            
        if creds is not None:
            client = gspread.authorize(creds)
            workbook = client.open_by_key(SHEET_ID)
            worksheet = workbook.get_worksheet(0)
            worksheet.clear()
            
            df_filled = df_to_save.fillna('')
            data_to_upload = [df_filled.columns.values.tolist()] + df_filled.values.tolist()
            worksheet.update(data_to_upload)
            
            status_placeholder.empty()
            st.session_state.ts_extended = df
            st.session_state.upload_done = True
            
            st.toast("🎉 Google Sheet Cloud သို့ ဒေတာများ Sync လုပ်ပြီးပါပြီ။", icon="✅")
            time.sleep(0.5)
            st.rerun()
        else:
            status_placeholder.empty()
            st.session_state.ts_extended = df
            st.warning("⚠️ Cloud Credentials (Secrets/JSON) မရှိပါသော်လည်း Dashboard ထဲတွင် ဒေတာ အောင်မြင်စွာ Update ဖြစ်သွားပါပြီ။")
            
    except Exception as e:
        status_placeholder.empty()
        st.error(f"❌ Google Sheet သို့ Sync လုပ်ရာတွင် အဆင်မပြေပါ - {e}")

if 'ts_extended' not in st.session_state or st.session_state.ts_extended is None:
    df_init = load_data_from_sheets(GOOGLE_SHEET_URL)
    if df_init is not None:
        st.session_state.ts_extended = df_init
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
        st.session_state.ts_extended = df
    else:
        dates_idx = pd.date_range(end=datetime.date.today(), periods=35)
        df = pd.DataFrame({'Date': dates_idx})
        for st_name in stations_list:
            df[f'{st_name}_WL'] = 400
            df[f'{st_name}_RF'] = 0.0
        st.session_state.ts_extended = df

@st.cache_resource
def load_ai_model(m_path):
    from tensorflow.keras.layers import Dense
    return load_model(m_path, compile=False, custom_objects={'Dense': lambda **kwargs: Dense(**{k: v for k, v in kwargs.items() if k != 'quantization_config'})})

@st.cache_resource
def load_scaler(s_path):
    return joblib.load(s_path)

# ==============================================================================
# 🌧️ 4. OPEN-METEO WEATHER API CALL FUNCTION
# ==============================================================================
def get_weather_forecast_array(lat, lon, lt, base_date_str):
    start_date = pd.to_datetime(base_date_str)
    days_to_get = int(lt)
    end_date = start_date + pd.Timedelta(days=days_to_get)
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=precipitation_sum&start_date={start_str}&end_date={end_str}&timezone=Asia/Yangon"
    try:
        r = requests.get(url, timeout=10).json()
        if 'daily' in r and 'precipitation_sum' in r['daily']:
            p_list = r['daily']['precipitation_sum']
            cleaned_list = [float(x) if x is not None else 0.0 for x in p_list]
            if len(cleaned_list) >= days_to_get:
                return cleaned_list[:days_to_get]
            else:
                return cleaned_list + [0.0] * (days_to_get - len(cleaned_list))
    except Exception as e:
        pass
    return [0.0] * days_to_get

# ==============================================================================
# 🎨 UI TITLE SYSTEM & ULTRA-COMPACT BOLD CSS STYLING
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Pyidaungsu:wght@400;700&display=swap');
    
    /* Font နှင့် Global Formatting */
    [data-testid="stMainBlockContainer"] p, 
    [data-testid="stMainBlockContainer"] h1, 
    [data-testid="stMainBlockContainer"] h2, 
    [data-testid="stMainBlockContainer"] h3, 
    [data-testid="stMainBlockContainer"] span, 
    [data-testid="stMainBlockContainer"] label, 
    [data-testid="stMainBlockContainer"] button {
        font-family: 'Pyidaungsu', 'Segoe UI', sans-serif !important;
    }
    
    /* Screen Outer Padding ကို အနည်းဆုံးအထိ နိမ့်လိုက်ခြင်း */
    [data-testid="stMainBlockContainer"] {
        max-width: 99% !important;
        padding-top: 0.1rem !important;
        padding-bottom: 0.1rem !important;
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
        margin: 0 auto !important;
    }
    
    header[data-testid="stHeader"] {
        background-color: rgba(0,0,0,0) !important;
        height: 0rem !important;
    }
    
    /* Tabs Padding & Font */
    div[data-testid="stTabs"] button [data-testid="stMarkdownContainer"] p,
    div[data-testid="stTabs"] button p,
    .stTabs [role="tab"] p,
    button[id^="tabs-bndry"] p {
        font-family: 'Pyidaungsu', sans-serif !important;
        font-size: 15px !important;
        font-weight: bold !important;
        color: #1e5494 !important;
        padding-top: 0px !important;
        padding-bottom: 0px !important;
    }
    
    label[data-testid="stWidgetLabel"] p {
        font-size: 13px !important;
        font-weight: bold !important;
        color: #1e5494 !important;
        margin-bottom: 0px !important;
    }
    
    div[data-baseweb="input"] input, div[data-baseweb="select"] * {
        font-size: 13px !important;
        height: 34px !important;
    }
    
    /* 💡 1. Header Titles Font Size များကို ပိုကြီးပေးလိုက်ပါသည် */
    .dept-title {
        color: #0b6623;
        font-size: 26px !important; /* 👈 19px မှ 26px သို့ ကြီးပေးထားသည် */
        font-weight: bold;
        line-height: 1.3;
        margin-bottom: 2px;
    }
    .dash-title {
        color: #1e5494;
        font-size: 18px !important; /* 👈 14px မှ 18px သို့ ကြီးပေးထားသည် */
        font-weight: bold;
        line-height: 1.2;
        margin-bottom: 0px;
    }
    
    hr {
        margin-top: 2px !important;
        margin-bottom: 4px !important;
    }
    
    /* 💡 DATAFRAME / TABLE STYLING (BOLD & ULTRA-COMPACT ROWS) */
    div[data-testid="stDataFrame"] {
        font-family: 'Pyidaungsu', 'Segoe UI', sans-serif !important;
        margin-top: 0px !important;
    }
    
    /* ဇယားကွက်ထဲမှ စာလုံး အကုန်လုံးကို Bold ထူပေးပြီး Size ချိန်ပေးခြင်း */
    div[data-testid="stDataFrame"] * {
        font-size: 13.5px !important;
        font-weight: 700 !important; /* Bold ပြုလုပ်ထားပါသည် */
    }

    /* Row Vertical Height များကို ကျုံ့၍ ဇလွန်စခန်းအထိ အကုန်ပေါ်အောင် ပြုလုပ်ခြင်း */
    div[data-testid="stDataFrame"] [data-testid="stCanvas"] div {
        line-height: 1.2 !important;
    }

    /* Streamlit Vertical Layout Gap ကို လျှော့ချခြင်း */
    [data-testid="stVerticalBlock"] > div {
        gap: 0.15rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Logo ဖိုင် ရှာဖွေခြင်း
logo_file_candidates = ['DMH Logo.png', 'DMH_Logo.png', 'dmh_logo.png', 'logo.png']
found_logo = None
for l_name in logo_file_candidates:
    if os.path.exists(l_name):
        found_logo = l_name
        break
        
# 💡 2. Logo နှင့် Header Column Ratio ကို ခေါင်းစဉ်ကြီးသည်နှင့်အမျှ ကွက်တိဖြစ်အောင် ချိန်ပေးထားပါသည်
head_col1, head_col2 = st.columns([1, 9])

with head_col1:
    if found_logo:
        st.image(found_logo, width=80) # 👈 Logo Width ကိုပါ 80px သို့ မျှအောင် ကြီးပေးထားသည်

with head_col2:
    st.markdown("<div class='dept-title'>မိုးလေဝသနှင့်ဇလဗေဒညွှန်ကြားမှုဦးစီးဌာန</div>", unsafe_allow_html=True)
    st.markdown("<div class='dash-title'>DMH AI Flood Dashboard (Ayeyarwady River)</div>", unsafe_allow_html=True)

st.markdown("<hr style='border: 0; height: 1px; background: #e0e0e0;'>", unsafe_allow_html=True)

# ==============================================================================
# 🗂️ TAB DEFINITION
# ==============================================================================
tab1, tab2, tab3 = st.tabs(["ဒေတာအသစ်တင်ရန်(CSV)", "ခန့်မှန်းချက်တွက်ရန်", "Graph များကြည့်ရန်"])

# ==============================================================================
# 📂 TAB 1: CSV DATA UPLOAD PANEL (PASSWORD PROTECTED)
# ==============================================================================
with tab1:
    st.markdown("### 📂 CSV File Bulk Upload (Admin Access)")
    
    # 🔐 Secret Password သတ်မှတ်ခြင်း (ဒီနေရာမှာ မိမိကြိုက်နှစ်သက်ရာ Password ပြောင်းနိုင်ပါသည်)
    ADMIN_PASSWORD = "Zala@NPT"  # 👈 မိမိထားချင်သည့် Password ကို ဒီမှာ ပြောင်းပါ (ဥပမာ- "dmh2026", "1234")
    
    # Session State တွင် Login Status ကို မှတ်ထားခြင်း
    if 'upload_authenticated' not in st.session_state:
        st.session_state.upload_authenticated = False

    # 🔑 Password မရိုက်ရသေးပါက Password တောင်းသည့် Box ကို ပြသမည်
    if not st.session_state.upload_authenticated:
        st.info("🔒 ဒေတာ မမှားယွင်းစေရန်အတွက် CSV File တင်ခြင်းကို Password ဖြင့် ထိန်းချုပ်ထားပါသည်။")
        
        pwd_col1, pwd_col2 = st.columns([3, 1])
        with pwd_col1:
            input_pwd = st.text_input("🔑 Admin Password ရိုက်ထည့်ပါ-", type="password", key="tab1_pwd_input")
        with pwd_col2:
            st.write("")
            st.write("")
            if st.button("🔓 Unlock", type="primary", use_container_width=True):
                if input_pwd == ADMIN_PASSWORD:
                    st.session_state.upload_authenticated = True
                    st.success("✅ Password မှန်ကန်ပါသည်။")
                    st.rerun()
                else:
                    st.error("❌ Password မှားယွင်းနေပါသည်။")
    
    # 🔓 Password မှန်သွားပါက CSV File Upload တင်သည့် မူရင်း UI ပေါ်လာမည်
    else:
        # Logout / Lock ပြန်လုပ်ချင်ပါက နှိပ်ရန် Button
        top_col1, top_col2 = st.columns([5, 1])
        with top_col2:
            if st.button("🔒 Lock Upload", type="secondary", use_container_width=True):
                st.session_state.upload_authenticated = False
                st.rerun()

        st.caption("✨ စခန်းအားလုံး၏ နေ့စဉ်ဒေတာပါဝင်သော CSV ဖိုင်ကို တင်လိုက်လျှင် Dashboard ထဲသို့ တိုက်ရိုက် Update ရောက်သွားမည်ဖြစ်သည်")
        
        uploaded_file = st.file_uploader("ကျေးဇူးပြု၍ ဒေတာဖိုင် (CSV) ကို ရွေးချယ်တင်ပေးပါ-", type=['csv'], key="tab1_csv_uploader")
        
        if uploaded_file is not None:
            try:
                # CSV ဖတ်ယူခြင်း
                df_uploaded = pd.read_csv(uploaded_file)
                st.session_state.uploaded_df = df_uploaded
                
                # Cloud သို့ ပို့ခြင်း / Update လုပ်ခြင်း
                save_data_to_sheets_and_cloud(df_uploaded)
                
            except Exception as e:
                st.error(f"❌ CSV ဖိုင်ဖတ်ရာတွင် အမှားအယွင်းရှိပါသည်- {e}")

# ==============================================================================
# 🔮 TAB 2: FORECAST SYSTEM ENGINE (COMPACT & COLLAPSIBLE CONTROLS)
# ==============================================================================
with tab2:
    # Controls များကို Expander သို့မဟုတ် 1-Row Inline Layout ပြုလုပ်ထားသဖြင့် Space အလွန်သက်သာသည်
    c_col1, c_col2, c_col3, c_col4 = st.columns([2, 1.5, 2, 2.5])
    
    with c_col1:
        date_forecast = st.date_input("📅 Base Date:", datetime.date.today(), key="fc_dt")
    with c_col2:
        lt_forecast = st.selectbox("🔮 Lead Time:", [1, 3], index=1)
    with c_col3:
        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
        run_btn = st.button("🚀 Run Forecast", type="primary", use_container_width=True)
    with c_col4:
        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
        sync_btn = st.button("🔄 Sync Sheet", type="secondary", use_container_width=True)

    if sync_btn:
        with st.spinner("Syncing..."):
            df_sync = load_data_from_sheets(GOOGLE_SHEET_URL)
            if df_sync is not None:
                st.session_state.ts_extended = df_sync
                st.toast("✅ Google Sheet ဒေတာ Sync ပြုလုပ်ပြီးပါပြီ။", icon="✅")
                st.rerun()

    base_date = date_forecast.strftime('%Y-%m-%d')

    if run_btn:
        lt_val = int(lt_forecast)
        target_dt = pd.to_datetime(date_forecast)
        ts_df = st.session_state.ts_extended
        
        with st.spinner("⏳ AI Forecast တွက်ချက်နေပါသည်..."):
            summary_results = []
            graphs_store = {}

            for st_name in stations_list:
                try:
                    st_info = meta_df[meta_df['Station'] == st_name].iloc[0]
                    danger_level = st_info['Danger Level_cm']
                    wl_col = f'{st_name}_WL'
                    rf_col = f'{st_name}_RF'
                    wd_col = f'{st_name}_Width'
                    el_col = f'{st_name}_Elev'
                    base_features = [wl_col, rf_col, wd_col, el_col]
                    
                    available_past_data = ts_df[ts_df['Date'] <= target_dt].copy()
                    if available_past_data.empty or available_past_data[wl_col].isna().all():
                        available_past_data = ts_df[ts_df[wl_col].notna()].copy()
                    
                   # 💡 ရက်စွဲ Format ပြဿနာမတက်အောင် ISO Format အဖြစ် သေချာပြောင်းလဲခြင်း
                    current_window_df = available_past_data.tail(30).copy()
                    
                    if not current_window_df.empty:
                        # ISO Format (YYYY-MM-DD) အဖြစ် အတိအကျပြောင်းယူခြင်း
                        last_dt_val = pd.to_datetime(current_window_df.iloc[-1]['Date'], errors='coerce')
                        actual_data_date_str = last_dt_val.strftime('%Y-%m-%d') if pd.notna(last_dt_val) else "No Data"
                    else:
                        actual_data_date_str = "No Data"
                    
                    dynamic_wl_header = f"📉 Last Obs WL ({actual_data_date_str}) (cm)"
                    
                    for col in base_features:
                        if col not in current_window_df.columns:
                            if '_Width' in col: current_window_df[col] = 100.0
                            elif '_Elev' in col: current_window_df[col] = 10.0
                            else: current_window_df[col] = 0.0
                            
                    forecasted_rainfalls = get_weather_forecast_array(st_info['Latitude'], st_info['Longitude'], lt_val, date_forecast)
                    
                    last_obs_wl = current_window_df.iloc[-1][wl_col] if not current_window_df.empty else 400
                    daily_change = last_obs_wl - current_window_df.iloc[-2][wl_col] if len(current_window_df) >= 2 else 0
                    station_f_dates, f_wl, f_rf = [], [], []
                    
                    for i in range(1, lt_val + 1):
                        curr_f_date = target_dt + pd.Timedelta(days=i)
                        m_path = f"{base_models_dir}/{st_name}_model_{i}day.h5"
                        if not os.path.exists(m_path): m_path = f"{base_models_dir}/{st_name}_model_1day.h5"
                        s_path = f"{base_models_dir}/{st_name}_scaler_1day.pkl"
                        
                        if os.path.exists(m_path) and os.path.exists(s_path):
                            m = load_ai_model(m_path)
                            s = load_scaler(s_path)
                            n_features = s.n_features_in_ if hasattr(s, 'n_features_in_') else len(s.scale_)
                            if n_features == 1:
                                final_input = np.hstack([s.transform(current_window_df[[wl_col]].values), current_window_df[[rf_col, wd_col, el_col]].values])
                                pred_wl_raw = int(round(s.inverse_transform(m(np.expand_dims(final_input, axis=0), training=False).numpy())[0, 0]))
                            else:
                                pred_wl_raw = int(round(m(np.expand_dims(s.transform(current_window_df[base_features].values), axis=0), training=False).numpy()[0, 0] * s.scale_[0] + s.min_[0]))
                            
                            pred_wl = int(round(pred_wl_raw))
                            if pred_wl <= 0 or abs(pred_wl - last_obs_wl) > 500:
                                pred_wl = int(last_obs_wl + (daily_change * i * 0.3))
                        else:
                            pred_wl = int(last_obs_wl + (daily_change * i * 0.3))
                            
                        station_f_dates.append(curr_f_date)
                        f_wl.append(pred_wl)
                        f_rf.append(forecasted_rainfalls[i - 1])
                        
                        new_row = current_window_df.iloc[-1].copy()
                        new_row['Date'] = curr_f_date
                        new_row[wl_col] = pred_wl
                        current_window_df = pd.concat([current_window_df.iloc[1:], pd.DataFrame([new_row])])
                        
                    final_pred_wl = f_wl[-1]
                    diff_from_danger = final_pred_wl - danger_level
                    wl_net_change = final_pred_wl - last_obs_wl
                    wl_change_display = f"📈 တက်မည် (+{int(wl_net_change)} cm)" if wl_net_change > 0 else (f"📉 ကျမည် ({int(wl_net_change)} cm)" if wl_net_change < 0 else "⚖️ မပြောင်းလဲပါ")
                    
                    if final_pred_wl >= danger_level:
                        status = "🚨 ရေကြီးနိုင်"
                    elif diff_from_danger >= -100:
                        status = "🟡 စောင့်ကြည့်ရန်လို"
                    else:
                        status = "🟢 စိတ်ချရ"
                        
                    summary_results.append({
                        "📍 Station": st_name, 
                        dynamic_wl_header: int(last_obs_wl), 
                        "⚠️ 24-hr Change (cm)": f"+{int(daily_change)}" if daily_change > 0 else f"{int(daily_change)}",
                        "🌧️ Forecast RF (mm)": round(sum(f_rf), 1), 
                        f"🔮 Forecasted WL ({lt_val}-Day)": final_pred_wl, 
                        "📈/📉 Forecast WL Change": wl_change_display,
                        "⚠️ Danger Level (cm)": danger_level, 
                        "➕/➖ Diff (cm)": f"+{diff_from_danger}" if diff_from_danger >= 0 else f"{diff_from_danger}", 
                        "📢 Status": status
                    })
                    graphs_store[st_name] = {
                        'f_dates': station_f_dates, 'f_wl': f_wl, 'f_rf': f_rf, 'danger_level': danger_level,
                        'recent_df': ts_df[ts_df['Date'] <= target_dt].tail(15).copy(), 'last_obs_wl': last_obs_wl
                    }
                except Exception as e:
                    st.error(f"❌ Error in {st_name}: {e}")
                    
            if summary_results:
                st.session_state.summary_results = summary_results
                st.session_state.GLOBAL_GRAPHS_DATA = graphs_store

   # 📊 Result Table ကို Height 580 ထားပေးခြင်းဖြင့် Zalun အထိ တိုက်ရိုက် အကုန်ပေါ်ပါမည်
    if getattr(st.session_state, 'summary_results', None) and len(st.session_state.summary_results) > 0:
        st.dataframe(
            pd.DataFrame(st.session_state.summary_results), 
            use_container_width=True, 
            height=580
        )
        
# ==============================================================================
# 📊 TAB 3: SELECTIVE GRAPH VIEW PANEL (OPTIMIZED FOR SINGLE SCREEN)
# ==============================================================================
with tab3:
    # 💡 1. Heading စာလုံး အရွယ်အစားကို compact ဖြစ်အောင် အသေးပြောင်းခြင်း
    st.markdown("<h5 style='margin-bottom: 0px;'>📈 Interactive Forecast Hydrograph & Rainfall</h5>", unsafe_allow_html=True)
    
    if not st.session_state.get('GLOBAL_GRAPHS_DATA'):
        st.warning("⚠️ ကျေးဇူးပြု၍ '🔮 ခန့်မှန်းချက်တွက်ရန်' Tab တွင် Run Forecast Engine ကို အရင်နှိပ်ပေးပါရန်။")
    else:
        # 💡 2. Selector နှင့် Button ကို ဘေးချင်းယှဉ် Column အဖြစ် ကျုံ့ထားခြင်း
        col_sel, col_btn, _ = st.columns([2.5, 1.2, 2.3])
        
        with col_sel:
            graph_station_selector = st.selectbox(
                "📍 Select Station (ကြည့်ရှုလိုသော စခန်းကို ရွေးချယ်ပါ):", 
                list(stations_list),
                key="graph_st_selector_plotly_final_fixed"
            )
            
        with col_btn:
            st.write("") # Vertical spacing အတွက်
            st.write("")
            btn_clicked = st.button("📊 Show Forecast Graph", type="primary", key="btn_trigger_plotly_final_fixed", use_container_width=True)
        
        # 💡 Button နှိပ်ခဲ့လျှင် သို့မဟုတ် ရွေးချယ်ပြီးသားဖြစ်ပါက Graph တန်းပြရန် Session State ထိန်းခြင်း
        if btn_clicked:
            st.session_state['active_graph_station'] = graph_station_selector

        st_name = st.session_state.get('active_graph_station', list(stations_list)[0])

        if st_name in st.session_state.GLOBAL_GRAPHS_DATA:
            p_pkg = st.session_state.GLOBAL_GRAPHS_DATA[st_name]
            
            forecast_dates = [pd.to_datetime(d).date() for d in p_pkg['f_dates']]
            f_wl_vals = p_pkg['f_wl']
            f_rf_vals = p_pkg['f_rf']
            
            observed_dates = []
            obs_wl = []
            obs_rf = []
            
            if 'ts_extended' in st.session_state and st.session_state.ts_extended is not None:
                try:
                    df_copy = st.session_state.ts_extended.copy()
                    df_copy['Parsed_Date'] = pd.to_datetime(df_copy['Date'])
                    
                    base_date_limit = pd.to_datetime(p_pkg['f_dates'][0]) - pd.Timedelta(days=1)
                    past_filtered_df = df_copy[df_copy['Parsed_Date'] <= base_date_limit].tail(10)
                    
                    observed_dates = past_filtered_df['Parsed_Date'].dt.date.tolist()
                    
                    wl_col = f'{st_name}_WL'
                    rf_col = f'{st_name}_RF'
                    
                    if wl_col in past_filtered_df.columns:
                        obs_wl = past_filtered_df[wl_col].values.tolist()
                    if rf_col in past_filtered_df.columns:
                        obs_rf = past_filtered_df[rf_col].values.tolist()
                    else:
                        obs_rf = [0] * len(observed_dates)
                except Exception as df_err:
                    st.error(f"❌ ဒေတာဖတ်ရာတွင် အခက်အခဲရှိပါသည်- {df_err}")
                
            if observed_dates and obs_wl:
                extended_forecast_dates = [observed_dates[-1]] + forecast_dates
                extended_forecast_wl = [obs_wl[-1]] + f_wl_vals
            else:
                extended_forecast_dates = forecast_dates
                extended_forecast_wl = f_wl_vals
                
            all_dates = observed_dates + forecast_dates
            min_date = min(all_dates) if all_dates else forecast_dates[0]
            max_date = max(all_dates) if all_dates else forecast_dates[-1]
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            if obs_wl:
                fig.add_trace(
                    go.Scatter(
                        x=observed_dates, y=obs_wl,
                        mode='lines+markers', name='Past Observed WL',
                        line=dict(color='#1f77b4', width=2.5),
                        marker=dict(size=5)
                    ),
                    secondary_y=False
                )
                
            fig.add_trace(
                go.Scatter(
                    x=extended_forecast_dates, y=extended_forecast_wl,
                    mode='lines+markers', name='AI Forecast WL',
                    line=dict(color='red', width=2.5, dash='dash'),
                    marker=dict(size=5, symbol='square')
                ),
                secondary_y=False
            )
            
            danger_val = p_pkg.get('danger_level', 1200)
            fig.add_trace(
                go.Scatter(
                    x=[min_date, max_date], y=[danger_val]*2,
                    mode='lines', name=f"Danger Level ({danger_val} cm)",
                    line=dict(color='darkred', width=2, dash='dot')
                ),
                secondary_y=False
            )
            
            if obs_rf:
                fig.add_trace(
                    go.Bar(
                        x=observed_dates, y=obs_rf,
                        name='Observed RF (mm)',
                        marker_color='#4A90E2', opacity=0.4
                    ),
                    secondary_y=True
                )
                
            fig.add_trace(
                go.Bar(
                    x=forecast_dates, y=f_rf_vals,
                    name='Forecast RF (mm)',
                    marker_color='#9B5DE5', opacity=0.5
                ),
                secondary_y=True
            )
            
            # 💡 3. Layout Margin နှင့် Height ကို မလိုအပ်ဘဲ အောက်မဆွဲရအောင် 420px သို့ ညှိထားခြင်း
            fig.update_layout(
    title=dict(
        text=f"<b>📊 Hydrograph, Rainfall Trend & Forecast: {st_name}</b>", 
        x=0.5, 
        y=0.98,          # 👈 Title ကို အပေါ်သို့ ပိုမြှင့်လိုက်သည်
        font=dict(size=12) # 👈 Title Font size ကို သေးလိုက်သည်
    ),
    margin=dict(l=10, r=10, t=50, b=10), # 👈 Top Margin ကို 50px ပေးပြီး Space ချပေးထားသည်
    xaxis=dict(
        type='date',
        tickformat='%Y-%m-%d',
        tickangle=-30,
        dtick=86400000,
        gridcolor='rgba(128,128,128,0.15)'
    ),
    yaxis=dict(
        title="Water Level (cm)", 
        side="left", 
        showgrid=True, 
        gridcolor='rgba(128,128,128,0.15)'
    ),
    yaxis2=dict(
        title="Rainfall (mm)",
        side="right",
        overlaying="y",
        showgrid=False,
        autorange="reversed"  
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.0,           # 👈 Legend ကို Title အောက်နားတွင် ကွက်တိ ကျစေရန် ညှိထားသည်
        xanchor="right",
        x=1,
        font=dict(size=10)
    ),
    hovermode="x unified",
    height=400 # 👈 Viewport ထဲ ကွက်တိဝင်အောင် Height 400px ထားပေးထားပါသည်
)
            
            # 🚀 Streamlit ပေါ်တွင် Graph ပြသခြင်း
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.error(f"❌ {st_name} စခန်းအတွက် Forecast Graph ဒေတာ မတွေ့ရှိရပါ။")
