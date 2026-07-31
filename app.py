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

# 💡 [GLOBAL DATA LOAD] Tab 2 တွင် တိုင်မပတ်စေရန် မက်တာဒေတာကို ကြိုတင်ဖတ်ထားခြင်း
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
# 🌐 LOCAL & UPLOADED CSV DATA LOADER (GOOGLE SHEETS BYPASSED)
# ==============================================================================
def load_data_from_sheets(url=None):
    # Google Sheet ကို မသွားတော့ဘဲ ဆရာကြီး တင်ထားတဲ့ Uploaded Data သို့မဟုတ် Local CSV ကိုပဲ တိုက်ရိုက်ဖတ်စေခြင်း
    try:
        # 1. သုံးစွဲသူ တင်ထားတဲ့ ဒေတာ ရှိမရှိ စစ်ဆေးခြင်း
        if 'uploaded_df' in st.session_state and st.session_state.uploaded_df is not None:
            df = st.session_state.uploaded_df.copy()
        elif os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            return None
            
        # 2. ရက်စွဲ Format ကို DD/MM/YYYY ပုံစံဖြင့် ကွက်တိ Parse လုပ်ခြင်း
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Date'])

        # 3. တန်ဖိုးများကို Numeric ပြောင်းလဲခြင်း
        for col in df.columns:
            if col != 'Date':
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 4. နောက်ဆုံးရက်စွဲမှ နောက်ကြောင်းပြန် ရက်ပေါင်း ၉၀ စာကို ယူခြင်း
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
    if df is not None:
        if not df.empty:
            # Base Date ကို ဒေတာထဲက နောက်ဆုံးရက်စွဲ (ဥပမာ- 2026-06-28) ကို အလိုအလျောက် ပေးစေခြင်း
            st.session_state.base_date = pd.to_datetime(df['Date']).max().date()
        return df
    else:
        if 'base_date' not in st.session_state or st.session_state.base_date is None:
            st.session_state.base_date = datetime.date.today()
        if os.path.exists(csv_path):
            df_local = pd.read_csv(csv_path)
            df_local['Date'] = pd.to_datetime(df_local['Date'], format='%d/%m/%Y', dayfirst=True, errors='coerce')
            return df_local
        return None

def save_data_to_sheets_and_cloud(df):
    status_placeholder = st.empty()
    try:
        status_placeholder.info("⏳ ဒေတာများကို မူရင်း Google Sheet နှင့်ပေါင်းစပ်ပြီး Cloud ပေါ်သို့ ပို့နေပါသည်...")
        
        df_to_save = df.copy()
        df_to_save['Date'] = df_to_save['Date'].dt.strftime('%Y-%m-%d')
        df_to_save.to_csv(csv_path, index=False)
        
        if os.path.exists(creds_json_path):
            try:
                scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
                creds = ServiceAccountCredentials.from_json_keyfile_name(creds_json_path, scope)
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
                
            except Exception as cloud_err:
                status_placeholder.empty()
                st.error(f"❌ Google Sheet Cloud ပေါ်သို့ ဒေတာလှမ်းတင်ရာတွင် Error တက်နေပါသည် - {cloud_err}")
        else:
            status_placeholder.empty()
            st.error("❌ 'google_creds.json' ဖိုင်ကို ဤစက်၏ Folder ထဲတွင် မတွေ့ရှိရသေးပါ။")
            
    except Exception as e:
        status_placeholder.empty()
        st.error(f"❌ ဒေတာများကို Local တွင် သိမ်းဆည်းရာတွင် အခက်အခဲရှိပါသည် - {e}")

if 'ts_extended' not in st.session_state or st.session_state.ts_extended is None:
    df_init = load_data_from_sheets(GOOGLE_SHEET_URL)
    if df_init is not None:
        st.session_state.ts_extended = df_init
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'])
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
# 🎨 UI TITLE SYSTEM & CSS STYLING
# ==============================================================================
import base64
logo_path = 'DMH Logo.png'
logo_base64 = ""
if os.path.exists(logo_path):
    with open(logo_path, "rb") as img_file:
        logo_base64 = base64.b64encode(img_file.read()).decode()

style_css = """<style>
    @import url('https://fonts.googleapis.com/css2?family=Pyidaungsu:wght@400;700&display=swap');
    
    [data-testid="stMainBlockContainer"] p, 
    [data-testid="stMainBlockContainer"] h1, 
    [data-testid="stMainBlockContainer"] h2, 
    [data-testid="stMainBlockContainer"] h3, 
    [data-testid="stMainBlockContainer"] span, 
    [data-testid="stMainBlockContainer"] label, 
    [data-testid="stMainBlockContainer"] button {
        font-family: 'Pyidaungsu', 'Segoe UI', sans-serif !important;
    }
    
    [data-testid="stMainBlockContainer"] {
        max-width: 95% !important;
        padding-top: 0.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        margin: 0 auto !important;
    }
    
    header[data-testid="stHeader"] {
        background-color: rgba(0,0,0,0) !important;
        height: 1.5rem !important;
    }
    
    div[data-testid="stTabs"] button [data-testid="stMarkdownContainer"] p,
    div[data-testid="stTabs"] button p,
    .stTabs [role="tab"] p,
    button[id^="tabs-bndry"] p {
        font-family: 'Pyidaungsu', sans-serif !important;
        font-size: 22px !important;
        font-weight: bold !important;
        color: #1e5494 !important;
    }
    
    label[data-testid="stWidgetLabel"] p {
        font-size: 18px !important;
        font-weight: bold !important;
        color: #1e5494 !important;
    }
    
    div[data-baseweb="input"] input, div[data-baseweb="select"] * {
        font-size: 18px !important;
    }
    div[data-testid="stDataFrame"] {
        font-family: 'Pyidaungsu', 'Segoe UI', sans-serif !important;
    }
    
    div[data-testid="stDataFrame"] * {
        font-size: 18px !important;
    }
    
    div[data-testid="stDataFrame"] [role="gridcell"],
    div[data-testid="stDataFrame"] [role="columnheader"],
    div[data-testid="stDataFrame"] td,
    div[data-testid="stDataFrame"] th {
        text-align: left !important;
        justify-content: flex-start !important;
    }
</style>"""
st.markdown(style_css, unsafe_allow_html=True)

# Logo HTML ပြင်ဆင်ခြင်း
logo_html = f"<div style='flex-shrink: 0; display: flex; align-items: center;'><img src='data:image/png;base64,{logo_base64}' width='125'></div>" if logo_base64 else ""

# Header UI HTML ပြင်ဆင်ခြင်း
ui_html = f"""<div style="display: flex; align-items: center; justify-content: center; gap: 25px; width: 100%; margin-bottom: 20px; padding: 10px 0;">
    {logo_html}
    <div style="display: flex; flex-direction: column; align-items: flex-start; text-align: left;">
        <h1 style="color: #0b6623 !important; font-size: 28px !important; font-weight: bold !important; margin: 0 0 4px 0 !important; padding: 0 !important; line-height: 1.2;">
            မိုးလေဝသနှင့်ဇလဗေဒညွှန်ကြားမှုဦးစီးဌာန
        </h1>
        <div style="display: flex; align-items: center; gap: 12px;">
            <h2 style="color: #1e5494 !important; font-size: 19px !important; font-weight: bold !important; margin: 0 !important; padding: 0 !important; line-height: 1.2;">
                DMH AI Flood Dashboard (Ayeyarwady River)
            </h2>
        </div>
    </div>
</div><hr style="border: 0; height: 1px; background: #e0e0e0; margin-bottom: 20px;">"""

st.markdown(ui_html, unsafe_allow_html=True)

if 'GLOBAL_GRAPHS_DATA' not in st.session_state:
    st.session_state.GLOBAL_GRAPHS_DATA = {}

if 'summary_results' not in st.session_state:
    st.session_state.summary_results = []

# ==============================================================================
# 🗂️ TAB DEFINITION
# ==============================================================================
tab1, tab2, tab3 = st.tabs(["ဒေတာအသစ်တင်ရန်(CSV)", "ခန့်မှန်းချက်တွက်ရန်", "Graph များကြည့်ရန်"])

# ==============================================================================
# 📁 TAB 1: CSV BULK UPLOAD PANEL
# ==============================================================================
with tab1:
    st.subheader("📁 CSV File Bulk Upload")
    st.write("✨ **စခန်းအားလုံး၏ နေ့စဉ်ဒေတာပါဝင်သော CSV ဖိုင်ကို တင်လိုက်လျှင် Google Sheet သို့ပါ တိုက်ရိုက် Update ရောက်သွားမည်ဖြစ်သည်**")
    
    if 'ts_extended' in st.session_state and st.session_state.ts_extended is not None:
        df_now = st.session_state.ts_extended.copy()
    else:
        df_now = pd.DataFrame()

    if 'last_uploaded_file' not in st.session_state:
        st.session_state.last_uploaded_file = None

    uploaded_file = st.file_uploader("ကျေးစုပြု၍ ဒေတာဖိုင် (CSV) ကို ရွေးချယ်တင်ပေးပါ-", type=['csv'])
    
    if uploaded_file is not None:
        if st.session_state.last_uploaded_file != uploaded_file.name:
            st.session_state.upload_done = False
            st.session_state.last_uploaded_file = uploaded_file.name

        if not st.session_state.upload_done:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                
                if 'Date' not in uploaded_df.columns:
                    st.error("❌ ဖိုင်ထဲတွင် 'Date' Column အတိအကျ မပါဝင်ပါ။")
                else:
                    try:
                        uploaded_df['Date'] = pd.to_datetime(uploaded_df['Date'], dayfirst=True, errors='coerce')
                        uploaded_df = uploaded_df.dropna(subset=['Date'])
                        
                        if not df_now.empty and 'Date' in df_now.columns:
                            df_now['Date'] = pd.to_datetime(df_now['Date'], dayfirst=True, errors='coerce')
                            df_now = df_now.dropna(subset=['Date'])
                            
                            df_old = df_now[~df_now['Date'].isin(uploaded_df['Date'])].copy()
                            updated_master = pd.concat([df_old, uploaded_df], ignore_index=True)
                        else:
                            updated_master = uploaded_df.copy()
                        
                        updated_master = updated_master.sort_values('Date').reset_index(drop=True)
                        save_data_to_sheets_and_cloud(updated_master)
                        
                    except Exception as inner_err:
                        st.error(f"❌ နောက်ကွယ်မှ ဒေတာများကို ပေါင်းစပ်ရာတွင် အဆင်မပြေပါ (Data Type Error) - {inner_err}")
                            
            except Exception as e:
                st.error(f"❌ ဒေတာတင်ရာတွင် Error တက်နေပါသည်: {str(e)}")
        else:
            st.success("✅ လက်ရှိတင်ထားသော CSV ဖိုင်မှ ဒေတာများကို Cloud ပေါ်သို့ သိမ်းဆည်းပြီးပါပြီဗျာ။")
            
# ==============================================================================
# 🔮 TAB 2: FORECAST SYSTEM ENGINE
# ==============================================================================
with tab2:
    st.subheader("Multi-Station AI Forecasting")
    
    if st.button("🔄 Sync Data from Google Sheet", type="secondary"):
        with st.spinner("Google Sheet မှ ဒေတာအသစ်များကို ရယူနေပါသည်..."):
            df_sync = load_data_from_sheets(GOOGLE_SHEET_URL)
            if df_sync is not None:
                st.session_state.ts_extended = df_sync
                st.success("✅ ဒေတာများကို အောင်မြင်စွာ နောက်ဆုံးပေါ်ဖြစ်အောင် လုပ်ဆောင်ပြီးပါပြီ။")
                st.rerun()

    f_col1, f_col2 = st.columns(2)
    with f_col1:
        date_forecast = st.date_input("📅 Base Date:", datetime.date.today(), key="fc_dt")
    with f_col2:
        lt_forecast = st.selectbox("🔮 Lead Time (Days):", [1, 3], index=1)

    base_date = date_forecast.strftime('%Y-%m-%d')
    last_observed_wl = 400 

    if st.button("🚀 Run Forecast Engine", type="primary"):
        lt_val = int(lt_forecast)
        target_dt = pd.to_datetime(date_forecast)
        ts_df = st.session_state.ts_extended
        st.info(f"⏳ စခန်းအားလုံး၏ {lt_val}-Day Ahead Forecast ဇယားကို တွက်ချက်နေပါသည်...")
        
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
                
                current_window_df = available_past_data.tail(30).copy()
                
                if not current_window_df.empty:
                    actual_data_date_str = current_window_df.iloc[-1]['Date'].strftime('%Y-%m-%d')
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
            st.success("📊 Summary Forecast Table Generated Successfully!")
            st.dataframe(pd.DataFrame(st.session_state.summary_results), use_container_width=True)
            st.balloons()
            
    elif len(st.session_state.summary_results) > 0:
        st.markdown("---")
        st.subheader("📊 လက်ရှိ တွက်ချက်ထားသော Forecast ရလဒ်များ")
        st.dataframe(pd.DataFrame(st.session_state.summary_results), use_container_width=True)

# ==============================================================================
# 📊 TAB 3: SELECTIVE GRAPH VIEW PANEL (STABLE & CORRECTED)
# ==============================================================================
with tab3:
    st.subheader("📈 Interactive Forecast Hydrograph & Rainfall")
    
    if not st.session_state.get('GLOBAL_GRAPHS_DATA'):
        st.warning("⚠️ ကျေးဇူးပြု၍ '🔮 ခန့်မှန်းချက်တွက်ရန်' Tab တွင် Run Forecast Engine ကို အရင်နှိပ်ပေးပါရန်။")
    else:
        graph_station_selector = st.selectbox(
            "📍 Select Station (ကြည့်ရှုလိုသော စခန်းကို ရွေးချယ်ပါ):", 
            list(stations_list),
            key="graph_st_selector_plotly_final_fixed"
        )
        
        if st.button("📊 Show Forecast Graph", type="primary", key="btn_trigger_plotly_final_fixed"):
            st_name = graph_station_selector
            
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
                            line=dict(color='#1f77b4', width=3),
                            marker=dict(size=6)
                        ),
                        secondary_y=False
                    )
                    
                fig.add_trace(
                    go.Scatter(
                        x=extended_forecast_dates, y=extended_forecast_wl,
                        mode='lines+markers', name='AI Forecast WL',
                        line=dict(color='red', width=3, dash='dash'),
                        marker=dict(size=6, symbol='square')
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
                
                fig.update_layout(
                    title=dict(text=f"<b>📊 Hydrograph & Rainfall Trend: {st_name}</b>", x=0.5),
                    xaxis=dict(
                        type='date',
                        tickformat='%Y-%m-%d',
                        tickangle=-45,
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
                        autorange="reversed"  # မိုးရေချိန်တိုင်များကို အပေါ်မှအောက်သို့ ပြရန်
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    hovermode="x unified",
                    height=600
                )
                
                # 🚀 Streamlit ပေါ်တွင် Graph ပြသခြင်း
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error(f"❌ {st_name} စခန်းအတွက် Forecast Graph ဒေတာ မတွေ့ရှိရပါ။")
