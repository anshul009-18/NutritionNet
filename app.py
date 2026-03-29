import io
import json
import datetime
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b4
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable, Image as RLImage)

st.set_page_config(
    page_title="NutritionNet — Skin & Nail Health Scanner",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════════════════════════════
# STYLES — Light blue theme, premium typography, forced light mode
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,400;0,600;0,700;1,400&family=DM+Sans:wght@300;400;500;600&display=swap');

html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"],
[data-testid="block-container"],.stApp,.stMarkdown,
[data-testid="stMarkdownContainer"],.element-container {
  background-color:#F4F8FF!important; color:#0a2540!important;
  color-scheme:light only!important;
}
*{ font-family:'DM Sans',sans-serif; box-sizing:border-box; }
h1,h2,h3{ font-family:'Fraunces',serif; }
[data-testid="stSidebar"],[data-testid="collapsedControl"]{ display:none; }
.block-container{ padding:0!important; padding-top:0!important; max-width:100%!important; margin:0 auto; }
p,span,label,div{ color:#0a2540; }
[data-testid="stSpinner"] p{ color:#1a5fa8!important; }

/* ── Hero ── */
.hero{
  background:linear-gradient(135deg,#0a2540 0%,#1a5fa8 60%,#2e86de 100%);
  padding:52px 64px 48px; text-align:left; position:relative; overflow:hidden;
  width:100vw; margin-left:calc(50% - 50vw);
  display:flex; align-items:center; justify-content:space-between; gap:40px;
}
.hero::before{
  content:'';position:absolute;inset:0;
  background:radial-gradient(ellipse at 10% 90%,rgba(100,180,255,0.15) 0%,transparent 55%),
              radial-gradient(ellipse at 85% 15%,rgba(180,220,255,0.10) 0%,transparent 50%);
}
.hero-left{ position:relative; z-index:1; flex:1; }
.hero-right{position:relative;z-index:1;flex-shrink:0;width:170px;height:170px;display:flex;align-items:center;justify-content:center;}
.hero-ring-outer{position:absolute;width:170px;height:170px;border-radius:50%;border:1.5px solid rgba(255,255,255,0.2);animation:pulse-ring 2.4s ease-out infinite;}
.hero-ring-inner{position:absolute;width:126px;height:126px;border-radius:50%;background:rgba(255,255,255,0.07);border:1.5px solid rgba(255,255,255,0.28);}
.hero-icon-wrap{position:relative;z-index:2;font-size:58px;line-height:1;}
.hero-check{position:absolute;top:-4px;right:-4px;width:26px;height:26px;border-radius:50%;background:#10b981;border:2px solid #0a2540;display:flex;align-items:center;justify-content:center;font-size:13px;color:#fff;font-weight:700;z-index:3;}
@keyframes pulse-ring{0%{transform:scale(1);opacity:0.6}70%{transform:scale(1.18);opacity:0}100%{transform:scale(1.18);opacity:0}}
.hero-badge{
  display:inline-block; background:rgba(255,255,255,0.12);
  border:1px solid rgba(255,255,255,0.22); border-radius:100px;
  padding:5px 14px; font-size:11px; font-weight:600; letter-spacing:0.08em;
  color:rgba(255,255,255,0.85); margin-bottom:20px; text-transform:uppercase;
}
.hero h1{
  font-family:'Fraunces',serif; font-size:58px; font-weight:700;
  color:#ffffff; margin:0 0 18px; line-height:1.0; letter-spacing:-0.02em;
}
.hero h1 span{ color:#90cdf4; }
.hero-sub{
  font-size:15px; color:rgba(255,255,255,0.72); max-width:560px;
  margin:0 0 28px; line-height:1.65; font-weight:300;
}
.hero-pills{ display:flex; gap:8px; flex-wrap:wrap; }
.hero-pill{
  background:rgba(255,255,255,0.12); border:1px solid rgba(255,255,255,0.2);
  border-radius:100px; padding:5px 14px; font-size:12px; font-weight:500;
  color:rgba(255,255,255,0.85);
}

/* ── Upload section ── */
.upload-section{
  background:#ffffff;border-radius:16px;
  padding:20px 24px 20px;border:1px solid #dbeafe;
  box-shadow:0 4px 16px rgba(10,37,64,0.06);
}
.upload-label{
  font-family:'Fraunces',serif;font-size:18px;font-weight:600;
  color:#0a2540;margin-bottom:12px;display:block;
}


[data-testid="stFileUploader"],
[data-testid="stFileUploader"]>div,
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] section>div,
[data-testid="stFileUploaderDropzone"],
div[class*="uploadedFile"],div[class*="fileUploader"]{
  background:#f0f7ff!important;background-color:#f0f7ff!important;color-scheme:light!important;
}
[data-testid="stFileUploader"]{border:2px dashed #93c5fd!important;border-radius:14px!important;padding:0.5rem!important;}
[data-testid="stFileUploaderDropzone"]{border:none!important;border-radius:10px!important;padding:1.2rem!important;background:#f0f7ff!important;}
[data-testid="stFileUploaderDropzone"]:hover{background:#e0eeff!important;}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploaderDropzone"] span{color:#1a5fa8!important;font-weight:500!important;}
[data-testid="stFileUploader"] small,
[data-testid="stFileUploaderDropzone"] small{color:#60a5fa!important;}
[data-testid="stFileUploaderDropzone"] svg{fill:#93c5fd!important;}
[data-testid="stBaseButton-secondary"],
[data-testid="stFileUploaderDropzone"] button{
  background:#ffffff!important;color:#1a5fa8!important;
  border:1.5px solid #93c5fd!important;border-radius:10px!important;font-weight:500!important;
}

/* ── Page body ── */
.page-body{padding:24px 32px;background:#F4F8FF;}

/* ── Result card ── */
.rcard{
  background:#ffffff;border-radius:20px;overflow:hidden;
  border:1px solid #dbeafe;box-shadow:0 4px 24px rgba(10,37,64,0.08);
  animation:riseUp 0.5s cubic-bezier(.22,.68,0,1.2);margin-bottom:20px;
}
@keyframes riseUp{from{opacity:0;transform:translateY(20px) scale(.98)}to{opacity:1;transform:translateY(0) scale(1)}}
.rcard-top{padding:22px 26px 18px;}
.rcard-badge-row{display:flex;align-items:center;gap:10px;margin-bottom:12px;}
.rtype-pill{font-size:10px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;padding:3px 10px;border-radius:100px;}
.pill-nail{background:#dbeafe;color:#1e40af;}
.pill-skin{background:#d1fae5;color:#065f46;}
.pill-healthy{background:#dcfce7;color:#14532d;}
.rcard-title{font-family:'Fraunces',serif;font-size:22px;font-weight:700;color:#0a2540;margin:0 0 8px;line-height:1.2;}
.rcard-desc{font-size:14px;color:#4b5563;line-height:1.7;margin:0;}
.sec-divider{height:1px;background:#f0f7ff;margin:0 26px;}

/* ── Gauge ── */
.gauge-section{padding:18px 26px;}
.gauge-wrap{background:#f4f8ff;border-radius:14px;padding:18px 20px;border:1px solid #dbeafe;}
.gauge-title{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;color:#6b7280;margin-bottom:14px;}
.gauge-track{height:10px;background:#e2e8f0;border-radius:100px;overflow:hidden;}
.gauge-fill{height:100%;border-radius:100px;transition:width 1s cubic-bezier(.22,.68,0,1.2);}
.gauge-labels{display:flex;justify-content:space-between;font-size:11px;color:#9ca3af;margin-top:6px;}
.gauge-stats{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-top:14px;}
.gstat{text-align:center;background:#ffffff;border-radius:10px;padding:10px 8px;border:1px solid #dbeafe;}
.gstat-val{font-family:'Fraunces',serif;font-size:18px;font-weight:700;color:#0a2540;display:block;}
.gstat-lbl{font-size:10px;color:#6b7280;display:block;margin-top:2px;}

/* ── Bars ── */
.bars-section{padding:0 26px 18px;}
.section-title{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;color:#6b7280;margin-bottom:12px;}
.bar-row{margin-bottom:10px;}
.bar-meta{display:flex;justify-content:space-between;font-size:13px;color:#374151;margin-bottom:5px;}
.bar-meta span:last-child{font-weight:600;}
.bar-bg{height:8px;background:#e2e8f0;border-radius:100px;overflow:hidden;border:1px solid #dbeafe;}
.bar-fg{height:100%;border-radius:100px;}

/* ── Food grid ── */
.food-section{padding:0 26px 18px;}
.food-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;}
.food-card{background:#f4f8ff;border:1px solid #dbeafe;border-radius:14px;padding:14px 8px 12px;text-align:center;transition:transform .18s,box-shadow .18s;cursor:default;}
.food-card:hover{transform:translateY(-3px);box-shadow:0 6px 20px rgba(10,37,64,0.12);}
.food-emoji{font-size:28px;display:block;margin-bottom:6px;}
.food-name{font-size:12px;font-weight:600;color:#1a5fa8;margin-bottom:6px;}
.food-pill{display:inline-block;background:#e0eeff;border-radius:100px;padding:2px 9px;font-size:10px;color:#1a5fa8;font-weight:500;}
.food-need{font-size:10px;color:#9ca3af;margin-top:10px;font-style:italic;}

/* ── Tips ── */
.tips-section{padding:0 26px 18px;}
.tip-card{display:flex;gap:12px;padding:13px 14px;background:#f4f8ff;border:1px solid #dbeafe;border-radius:12px;margin-bottom:8px;}
.tip-idx{min-width:26px;height:26px;border-radius:50%;background:#dbeafe;color:#1e40af;font-size:12px;font-weight:700;display:flex;align-items:center;justify-content:center;flex-shrink:0;}
.tip-body{font-size:13px;color:#374151;line-height:1.6;}
.tip-body strong{color:#0a2540;}

/* ── Symptom checker ── */
.symptom-section{padding:0 26px 18px;}
.symptom-box{background:#fffbeb;border:1px solid #fde68a;border-radius:14px;padding:16px 18px;}
.symptom-title{font-size:13px;font-weight:600;color:#78350f;margin-bottom:10px;}
.symptom-row{display:flex;gap:7px;flex-wrap:wrap;}
.symptom-chip{background:#fef3c7;border:1px solid #fde68a;border-radius:100px;padding:4px 12px;font-size:11px;color:#92400e;}

/* ── Disclaimer ── */
.disclaimer-bar{background:#f4f8ff;border-top:1px solid #dbeafe;padding:14px 26px;text-align:center;font-size:11px;color:#9ca3af;line-height:1.5;}

/* ── Model status ── */
.model-footer{display:flex;justify-content:space-between;align-items:center;padding-top:10px;margin-top:10px;border-top:1px solid #f0f7ff;}

/* ── Buttons ── */
.stButton button{
  background:#0a2540!important;color:#ffffff!important;border:none!important;
  border-radius:12px!important;padding:12px 24px!important;font-weight:600!important;
  font-size:15px!important;width:100%!important;cursor:pointer!important;
}
.stButton button:hover{background:#1a5fa8!important;}
.stButton button *{color:#ffffff!important;background:transparent!important;border:none!important;font-weight:600!important;}
[data-testid="stDownloadButton"] button{
  background:#ffffff!important;color:#0a2540!important;
  border:1.5px solid #0a2540!important;border-radius:10px!important;
  font-size:14px!important;font-weight:600!important;
}
[data-testid="stDownloadButton"] button:hover{background:#f0f7ff!important;}

/* diet toggle handled via st.columns buttons */
[data-testid="stImage"] img{border-radius:14px;border:1px solid #dbeafe;max-height:280px;width:100%!important;object-fit:cover;display:block;}
[data-testid="stImage"] p{color:#9ca3af!important;font-size:12px!important;}
.stTabs [data-baseweb="tab-list"]{background:#e0eeff!important;border-radius:12px!important;padding:4px!important;gap:4px!important;display:flex!important;justify-content:center!important;width:fit-content!important;margin:0 auto!important;}
.stTabs [data-baseweb="tab"]{border-radius:9px!important;font-size:14px!important;font-weight:500!important;color:#6b7280!important;padding:8px 28px!important;letter-spacing:0.01em!important;}
.stTabs [aria-selected="true"]{background:#ffffff!important;color:#0a2540!important;font-weight:600!important;box-shadow:0 2px 8px rgba(10,37,64,0.12)!important;}
.stTabs [data-testid="stTabs"]{display:flex!important;flex-direction:column!important;align-items:center!important;}
.about-card{background:#ffffff;border-radius:18px;border:1px solid #dbeafe;padding:22px;margin-bottom:14px;}
.about-card h3{font-family:'Fraunces',serif;font-size:17px;color:#0a2540;margin:0 0 10px;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
NAIL_MODEL_PATH = "/Users/ajanshul02gmail.com/Mac/nail_disease_dataset/best_model.keras"
SKIN_MODEL_PATH = "/Users/ajanshul02gmail.com/Mac/nail_disease_dataset/best_vitamin_model.pth"
NAIL_IMG_SIZE   = 224
SKIN_IMG_SIZE   = 380
CONF_THRESH     = 55.0

NAIL_CLASSES = ["healthy", "onychomycosis", "psoriasis"]
SKIN_CLASSES = sorted([
    "Vitamin A deficiency", "Vitamin B-12 deficiency",
    "Vitamin C deficiency", "Vitamin D deficiency", "Zinc/Iron/Biotin deficiency",
])

NAIL_COLORS = {"healthy": "#10b981", "onychomycosis": "#f59e0b", "psoriasis": "#ef4444"}
SKIN_COLORS = {
    "Vitamin A deficiency": "#f97316", "Vitamin B-12 deficiency": "#ef4444",
    "Vitamin C deficiency": "#eab308", "Vitamin D deficiency": "#f97316",
    "Zinc/Iron/Biotin deficiency": "#8b5cf6",
}

SEVERITY_SCORE = {
    "healthy": 0, "onychomycosis": 55, "psoriasis": 72,
    "Vitamin A deficiency": 60, "Vitamin B-12 deficiency": 78,
    "Vitamin C deficiency": 55, "Vitamin D deficiency": 70,
    "Zinc/Iron/Biotin deficiency": 58,
}

def severity_label(s):
    if s < 30:  return "None",     "#10b981"
    if s < 55:  return "Mild",     "#f59e0b"
    if s < 75:  return "Moderate", "#f97316"
    return             "Severe",   "#ef4444"

def severity_gauge_color(s):
    if s < 30: return "#10b981"
    if s < 55: return "#f59e0b"
    if s < 75: return "#f97316"
    return "#ef4444"

# ══════════════════════════════════════════════════════════════════════════════
# DISEASE INFO  (nail + skin, merged from both files)
# ══════════════════════════════════════════════════════════════════════════════
NAIL_HEADER_MAP = {
    "psoriasis":     "Vitamin D deficiency detected",
    "onychomycosis": "Zinc deficiency detected",
}
NAIL_BAR_MAP = {
    "psoriasis":     "Vitamin D deficiency",
    "onychomycosis": "Zinc deficiency",
    "healthy":       "Healthy nail",
}

INFO = {
    "healthy": {
        "title": "Healthy nail", "sub": "No disease detected", "prevalence": "—",
        "model": "MobileNetV3", "pill": "pill-healthy", "pill_text": "💅 Nail scan",
        "desc": "Your nail appears healthy with no signs of disease. Maintain good nail hygiene and a balanced diet to keep nails strong.",
        "rich_foods": [("💧","Water","Hydration"),("🥚","Eggs","Biotin"),("🥦","Broccoli","Vitamin C"),("🐟","Fish","Omega-3"),("🌰","Nuts","Zinc"),("🍊","Citrus","Vitamin C")],
        "need": "Stay hydrated and eat biotin-rich foods",
        "tips": [],
    },
    "onychomycosis": {
        "title": "Zinc deficiency detected", "sub": "Onychomycosis — Fungal nail infection",
        "prevalence": "10% of population", "model": "MobileNetV3",
        "pill": "pill-nail", "pill_text": "💅 Nail scan",
        "desc": "A fungal infection linked to low zinc and protein levels that weaken nail structure. Causes thickening, discolouration, and brittleness. Very treatable with antifungals and zinc supplementation.",
        "rich_foods": [("🌰","Pumpkin seeds","Zinc"),("🫘","Chickpeas","Iron"),("🥚","Eggs","Protein"),("🧄","Garlic","Antifungal"),("🥛","Yoghurt","Probiotic"),("🍊","Oranges","Vitamin C")],
        "need": "Zinc: 8–11mg/day + antifungal diet",
        "tips": [
            ("🥜","Zinc sources","Eat pumpkin seeds, chickpeas, and cashews daily — low zinc weakens nail structure and invites fungal growth."),
            ("🚿","Keep nails dry","Trim nails short, wear breathable footwear, change socks daily. Never share nail tools."),
            ("💊","Antifungal treatment","Oral antifungal medication is usually required. Topical creams alone rarely penetrate deep enough."),
            ("🏥","Get zinc tested","Ask for a serum zinc blood test. Dermatologist can prescribe antifungal + zinc supplementation together."),
        ],
    },
    "psoriasis": {
        "title": "Vitamin D deficiency detected", "sub": "Nail Psoriasis — Autoimmune nail condition",
        "prevalence": "~50% of psoriasis patients", "model": "MobileNetV3",
        "pill": "pill-nail", "pill_text": "💅 Nail scan",
        "desc": "An autoimmune nail condition strongly linked to Vitamin D deficiency. Causes pitting, discolouration, and nail separation. Treatment focuses on Vitamin D restoration and reducing immune response.",
        "rich_foods": [("🐟","Salmon","Vitamin D"),("🥚","Egg yolk","Vitamin D"),("🍄","Mushrooms","Vitamin D"),("🥛","Fortified milk","Calcium"),("☀️","Sunlight","Vitamin D"),("🫐","Blueberries","Antioxidant")],
        "need": "Vitamin D: 600–800 IU/day + sunlight",
        "tips": [
            ("☀️","Sunlight daily","Get 15–20 min of midday sunlight on skin — Vitamin D from sunlight directly reduces psoriasis nail flares."),
            ("🐟","Fatty fish","Salmon, mackerel, sardines are the best natural Vitamin D sources. Aim for 2 servings per week."),
            ("🧘","Manage stress","Stress is the #1 psoriasis trigger. Practice 10–15 min daily meditation and ensure 7–8 hrs sleep."),
            ("🏥","Get D levels tested","Ask for a 25-hydroxyvitamin D blood test. A dermatologist may prescribe Vitamin D analogues or biologic therapy."),
        ],
    },
    "Vitamin A deficiency": {
        "title": "Vitamin A deficiency", "sub": "Retinol deficiency",
        "prevalence": "~29% globally", "model": "EfficientNet-B4",
        "pill": "pill-skin", "pill_text": "🩺 Skin scan",
        "desc": "Vitamin A deficiency causes dry rough skin, poor wound healing, and increased infection susceptibility. Fat-soluble — always eat with healthy fats for proper absorption.",
        "rich_foods": [("🥕","Carrots","β-carotene"),("🍠","Sweet potato","Vitamin A"),("🥦","Spinach","Iron + folate"),("🥚","Egg yolk","Retinol"),("🐟","Salmon","Vitamin D"),("🥛","Fortified milk","Calcium")],
        "need": "700–900 mcg RAE per day",
        "tips": [
            ("🥕","Eat orange foods","Carrots, sweet potatoes, pumpkin, and mango are rich in beta-carotene which converts into Vitamin A."),
            ("🥚","Animal sources","Liver, egg yolks, and dairy contain preformed Vitamin A (retinol) — the most bioavailable form."),
            ("🥦","Pair with fats","Always eat plant-based Vitamin A sources with olive oil or avocado — fat dramatically increases absorption."),
            ("🏥","Get tested first","Ask for a serum retinol blood test before supplementing — excess Vitamin A can be toxic at high doses."),
        ],
    },
    "Vitamin B-12 deficiency": {
        "title": "Vitamin B-12 deficiency", "sub": "Cobalamin deficiency",
        "prevalence": "~6% of adults", "model": "EfficientNet-B4",
        "pill": "pill-skin", "pill_text": "🩺 Skin scan",
        "desc": "Vitamin B-12 deficiency causes fatigue, pale skin, nerve tingling, and megaloblastic anaemia. Vegans and older adults are at significantly higher risk.",
        "rich_foods": [("🐟","Salmon","B-12"),("🥩","Beef","B-12"),("🥚","Eggs","B-12"),("🥛","Dairy","B-12"),("🦐","Shellfish","B-12"),("🌱","Fortified foods","B-12")],
        "need": "2.4 mcg per day",
        "tips": [
            ("🐟","Animal sources first","Salmon, tuna, beef, eggs, and dairy are the richest B-12 sources. Eat at least one daily."),
            ("🌱","Vegan supplement","If you avoid animal products, take B-12 supplements (500–1000 mcg/day) or eat fortified cereals."),
            ("🧠","Watch for nerve signs","Tingling or numbness in hands and feet is a serious symptom — seek medical attention urgently."),
            ("🏥","Blood test required","Serum B-12 below 200 pg/mL requires treatment. Injections may be needed if absorption is impaired."),
        ],
    },
    "Vitamin C deficiency": {
        "title": "Vitamin C deficiency", "sub": "Scurvy / ascorbic acid deficiency",
        "prevalence": "~7% (low-income)", "model": "EfficientNet-B4",
        "pill": "pill-skin", "pill_text": "🩺 Skin scan",
        "desc": "Vitamin C deficiency (scurvy) causes bleeding gums, slow-healing wounds, rough bumpy skin, and easy bruising. Resolves quickly with adequate dietary intake.",
        "rich_foods": [("🍊","Oranges","Vitamin C"),("🥝","Kiwi","Vitamin C"),("🍓","Strawberries","Vitamin C"),("🫑","Bell pepper","Vitamin C"),("🥦","Broccoli","Vitamin C"),("🍋","Lemon","Vitamin C")],
        "need": "65–90 mg per day",
        "tips": [
            ("🫑","Red bell peppers","Contain 3× more Vitamin C than oranges. Eat raw — cooking destroys up to 50% of Vitamin C."),
            ("🍊","Daily citrus","One orange or kiwi gives your full daily Vitamin C requirement. Fresh fruit beats juice every time."),
            ("🔥","Cook gently","Vitamin C is heat-sensitive. Steam vegetables briefly and eat fruits raw for maximum nutrient retention."),
            ("💊","Supplements work","500–1000mg Vitamin C supplements are safe. Deficiency typically resolves within 2–4 weeks."),
        ],
    },
    "Vitamin D deficiency": {
        "title": "Vitamin D deficiency", "sub": "The sunshine vitamin",
        "prevalence": "~42% of adults", "model": "EfficientNet-B4",
        "pill": "pill-skin", "pill_text": "🩺 Skin scan",
        "desc": "The most common nutritional deficiency worldwide — causes fatigue, bone pain, muscle weakness, and skin issues. Strongly linked to indoor lifestyles and low sunlight exposure.",
        "rich_foods": [("🐟","Fatty fish","Vitamin D"),("🥚","Egg yolk","Vitamin D"),("🍄","Mushrooms","Vitamin D"),("🥛","Fortified milk","Vitamin D"),("🧀","Cheese","Calcium"),("☀️","Sunlight","Vitamin D")],
        "need": "600–800 IU per day",
        "tips": [
            ("☀️","Sunlight is key","15–20 minutes of direct midday sun on bare arms daily is the most effective natural Vitamin D source."),
            ("🐟","Eat fatty fish","Salmon, mackerel, sardines are the best dietary Vitamin D sources. Aim for 2 servings per week."),
            ("💊","Supplement safely","1000–4000 IU Vitamin D3 daily is safe. Always take with a fat-containing meal for proper absorption."),
            ("🏥","Test your levels","A 25-hydroxyvitamin D test confirms deficiency. Below 20 ng/mL requires treatment."),
        ],
    },
    "Zinc/Iron/Biotin deficiency": {
        "title": "Zinc deficiency", "sub": "Zinc / Iron / Biotin mineral deficiency",
        "prevalence": "~2 billion iron deficient", "model": "EfficientNet-B4",
        "pill": "pill-skin", "pill_text": "🩺 Skin scan",
        "desc": "Deficiencies in zinc, iron, or biotin cause hair thinning, brittle nails, dry skin, and slow wound healing. All are essential for cell growth and repair.",
        "rich_foods": [("🦪","Oysters","Zinc"),("🥩","Red meat","Iron"),("🌰","Pumpkin seeds","Zinc"),("🥬","Spinach","Iron"),("🥚","Eggs","Biotin"),("🫘","Lentils","Iron")],
        "need": "Zn: 8–11mg | Fe: 8–18mg | Biotin: 30mcg",
        "tips": [
            ("🥩","Zinc-rich foods","Red meat, shellfish, pumpkin seeds, and legumes are the richest zinc sources. Eat daily."),
            ("🥬","Iron absorption tip","Pair iron-rich foods with Vitamin C — this doubles iron absorption. Avoid coffee/tea with meals."),
            ("🥚","Biotin daily","Egg yolk, almonds, sweet potato, and salmon are excellent biotin sources for hair and nail growth."),
            ("🏥","Full mineral panel","Ask for a CBC and mineral panel to identify exactly which deficiency before supplementing."),
        ],
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# DIET PLANS (Indian, affordable, with veg swaps)
# ══════════════════════════════════════════════════════════════════════════════
VEG_SWAPS = {
    "Chicken curry": "Paneer curry", "chicken curry": "paneer curry",
    "Fish curry": "Rajma curry",     "fish curry": "rajma curry",
    "Fish tikka": "Paneer tikka",    "Methi chicken": "Methi paneer",
    "methi chicken": "methi paneer", "Chicken / paneer": "Paneer",
    "Chicken / rajma": "Rajma",      "Chicken / soya": "Soya chunks",
    "Fish / paneer": "Paneer",       "Fish / soya": "Soya curry",
    "Grilled fish": "Grilled paneer","grilled fish": "grilled paneer",
    "Egg bhurji": "Paneer bhurji",   "egg bhurji": "paneer bhurji",
    "Egg curry": "Chana curry",      "egg curry": "chana curry",
    "Anda paratha": "Aloo paratha",  "anda paratha": "aloo paratha",
    "Anda bhurji": "Paneer bhurji",  "boiled eggs": "dahi + peanuts",
    "Boiled eggs": "Dahi + peanuts", "boiled egg": "dahi",
    "Boiled egg": "Dahi",
}

def apply_veg_swaps(text):
    for meat, sub in VEG_SWAPS.items():
        text = text.replace(meat, sub)
    return text

DIET_PLANS = {
    "onychomycosis": {
        "focus": "Zinc, Iron & Protein-rich Indian foods to strengthen nails and fight fungal infection",
        "days": [
            ("Mon", "Poha + pumpkin seeds + chai",        "Moong dal + roti + onion salad",         "Chicken curry / paneer curry + rice"),
            ("Tue", "Boiled eggs + banana + milk",        "Rajma + steamed rice + sliced cucumber", "Methi sabzi + roti + curd"),
            ("Wed", "Besan chilla + green chutney",       "Chana dal + jeera rice + pickle",        "Egg bhurji / tofu bhurji + roti"),
            ("Thu", "Upma + peanuts + nimbu paani",       "Palak dal + rice + papad",               "Chicken / soya keema + roti + salad"),
            ("Fri", "Idli + sambar + coconut chutney",    "Masoor dal + roti + tomato sabzi",       "Fish curry / paneer + rice + curd"),
            ("Sat", "Dalia + milk + jaggery",             "Chole + kulcha + onion rings",           "Egg curry / mixed veg curry + roti"),
            ("Sun", "Paratha + curd + pickle",            "Moong dal khichdi + ghee + papad",       "Chicken / soya chunks gravy + rice"),
        ]
    },
    "psoriasis": {
        "focus": "Anti-inflammatory Indian foods + Vitamin D & Zinc to reduce psoriasis nail flares",
        "days": [
            ("Mon", "Haldi doodh + poha + banana",        "Palak dal + roti + cucumber raita",      "Fish curry / paneer + brown rice"),
            ("Tue", "Moong dal chilla + green chutney",   "Rajma + rice + salad",                   "Methi chicken / tofu sabzi + roti"),
            ("Wed", "Dalia porridge + jaggery + milk",    "Chana dal + jeera rice + achar",         "Egg bhurji / paneer bhurji + roti"),
            ("Thu", "Idli + sambar + coconut chutney",    "Palak paneer / palak tofu + roti",       "Fish tikka / soya curry + rice + curd"),
            ("Fri", "Poha + peanuts + chai",              "Masoor dal + roti + tomato onion salad", "Chicken curry / rajma + rice"),
            ("Sat", "Besan chilla + curd + fruit",        "Khichdi + ghee + papad",                 "Egg curry / mixed veg + roti"),
            ("Sun", "Paratha + curd + haldi milk",        "Chole + rice + onion rings",             "Grilled fish / paneer tikka + salad"),
        ]
    },
    "Vitamin A deficiency": {
        "focus": "Beta-carotene rich Indian foods (gajar, papaya, aam) for Vitamin A",
        "days": [
            ("Mon", "Gajar juice + boiled eggs + roti",   "Pumpkin dal + roti + salad",             "Palak paneer / chicken + rice"),
            ("Tue", "Papaya + milk + poha",               "Gajar methi sabzi + roti + dal",         "Fish curry / tofu + rice + curd"),
            ("Wed", "Anda bhurji + gajar paratha",        "Sweet potato chaat + dal",               "Chicken / rajma + roti + salad"),
            ("Thu", "Dalia + milk + mango (seasonal)",    "Palak dal + rice + papad",               "Paneer or egg curry + roti"),
            ("Fri", "Besan chilla + tomato chutney",      "Gajar matar sabzi + roti + dal",         "Fish / soya curry + rice"),
            ("Sat", "Papaya + dahi + poha",               "Pumpkin khichdi + ghee",                 "Chicken / paneer + roti + salad"),
            ("Sun", "Paratha + curd + seasonal fruit",    "Palak chana dal + roti",                 "Egg bhurji / tofu + rice + pickle"),
        ]
    },
    "Vitamin B-12 deficiency": {
        "focus": "Eggs, dairy & fortified Indian foods to restore Vitamin B-12 levels",
        "days": [
            ("Mon", "Boiled eggs + milk + banana",        "Dal + roti + dahi",                      "Chicken curry / paneer + rice"),
            ("Tue", "Dahi + poha + chai",                 "Egg curry / chana + roti + salad",       "Fish curry / rajma + rice + curd"),
            ("Wed", "Anda paratha + dahi + fruit",        "Moong dal + rice + papad",               "Egg bhurji / tofu sabzi + roti"),
            ("Thu", "Dalia + milk + jaggery",             "Paneer bhurji / egg wrap + salad",       "Dal makhani / chicken + roti"),
            ("Fri", "Idli + sambar + boiled egg",         "Masoor dal + roti + curd",               "Fish / soya tikka + rice + salad"),
            ("Sat", "Besan chilla + dahi + fruit",        "Chole + kulcha + onion rings",           "Egg curry / mixed dal + roti"),
            ("Sun", "Paratha + eggs + chai",              "Khichdi + ghee + dahi",                  "Chicken / paneer curry + rice + salad"),
        ]
    },
    "Vitamin C deficiency": {
        "focus": "Amla, nimbu, tomato & guava — affordable Vitamin C sources in every Indian kitchen",
        "days": [
            ("Mon", "Amla juice + poha + chai",           "Tomato dal + roti + kachumber salad",    "Sabzi + roti + nimbu water"),
            ("Tue", "Guava + boiled eggs + milk",         "Palak dal + rice + raw onion",           "Chicken / rajma + roti + tomato salad"),
            ("Wed", "Nimbu poha + green chutney",         "Chana salad + roti + dahi",              "Stir-fried capsicum + paneer + rice"),
            ("Thu", "Orange juice + idli + sambar",       "Masoor dal + rice + raw tomato",         "Egg bhurji / soya + roti"),
            ("Fri", "Papaya + dalia + milk",              "Tomato shorba + roti + salad",           "Fish / paneer + rice + nimbu"),
            ("Sat", "Amla chutney + paratha + chai",      "Raw salad bowl + dal + roti",            "Chicken / chole + roti + curd"),
            ("Sun", "Guava + besan chilla + dahi",        "Khichdi + ghee + papad + nimbu",         "Mixed veg / egg curry + rice"),
        ]
    },
    "Vitamin D deficiency": {
        "focus": "Morning sunlight + eggs, fish, fortified milk — simple Indian Vitamin D sources",
        "days": [
            ("Mon", "Morning walk + boiled eggs + milk",  "Palak dal + roti + dahi",                "Fish curry / paneer + rice"),
            ("Tue", "Dahi + poha + banana",               "Egg curry / chana + roti + salad",       "Chicken / rajma + rice + curd"),
            ("Wed", "Besan chilla + fortified milk",      "Moong dal + rice + papad",               "Mushroom sabzi / egg bhurji + roti"),
            ("Thu", "Dalia + milk + jaggery + sunlight",  "Palak paneer / tofu + roti",             "Fish tikka / soya curry + rice"),
            ("Fri", "Boiled eggs + chai + fruit",         "Masoor dal + roti + salad",              "Chicken / mixed veg + roti + dahi"),
            ("Sat", "Fortified milk + paratha + curd",    "Chole + rice + onion rings",             "Fish / paneer curry + roti"),
            ("Sun", "Anda paratha + haldi doodh",         "Khichdi + ghee + dahi",                  "Grilled fish / egg + salad + roti"),
        ]
    },
    "Zinc/Iron/Biotin deficiency": {
        "focus": "Zinc & Iron-rich Indian foods — chana, rajma, eggs, pumpkin seeds for strong nails",
        "days": [
            ("Mon", "Poha + pumpkin seeds + chai",        "Rajma + rice + onion salad",             "Chicken / paneer + roti + curd"),
            ("Tue", "Boiled eggs + banana + milk",        "Chana dal + roti + spinach sabzi",       "Methi chicken / soya sabzi + rice"),
            ("Wed", "Besan chilla + green chutney",       "Masoor dal + jeera rice + papad",        "Egg bhurji / tofu + roti + salad"),
            ("Thu", "Dalia + milk + jaggery",             "Palak chole + roti + dahi",              "Fish curry / mixed dal + rice"),
            ("Fri", "Idli + sambar + pumpkin seeds",      "Moong dal + rice + tomato salad",        "Chicken / rajma + roti + curd"),
            ("Sat", "Anda paratha + curd + fruit",        "Kidney beans salad + roti + dal",        "Paneer / egg curry + rice"),
            ("Sun", "Upma + peanuts + chai",              "Khichdi + ghee + papad",                 "Chicken / soya keema + roti + salad"),
        ]
    },
    "healthy": {
        "focus": "Balanced everyday Indian diet to maintain nail & skin health",
        "days": [
            ("Mon", "Poha + chai + banana",               "Dal + roti + salad",                     "Sabzi + roti + curd + rice"),
            ("Tue", "Idli + sambar + coconut chutney",    "Rajma + rice + onion salad",             "Egg bhurji / paneer + roti"),
            ("Wed", "Dalia + milk + jaggery",             "Moong dal + jeera rice + papad",         "Fish / mixed veg curry + roti"),
            ("Thu", "Upma + peanuts + chai",              "Chana dal + roti + dahi",                "Chicken / tofu + rice + salad"),
            ("Fri", "Besan chilla + green chutney",       "Palak dal + roti + kachumber",           "Dal tadka + rice + curd"),
            ("Sat", "Paratha + curd + seasonal fruit",    "Chole + kulcha + onion rings",           "Egg curry / paneer + roti + salad"),
            ("Sun", "Poha + banana + haldi doodh",        "Khichdi + ghee + papad",                 "Dal makhani / chicken + roti + curd"),
        ]
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_nail_model():
    return tf.keras.models.load_model(NAIL_MODEL_PATH)

@st.cache_resource
def load_skin_model():
    m = efficientnet_b4(weights=None)
    inf = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.LayerNorm(inf), nn.Dropout(0.3),
        nn.Linear(inf, 512), nn.GELU(),
        nn.LayerNorm(512), nn.Dropout(0.15),
        nn.Linear(512, len(SKIN_CLASSES))
    )
    m.load_state_dict(torch.load(SKIN_MODEL_PATH, map_location="cpu"))
    m.eval()
    return m

SKIN_TF = transforms.Compose([
    transforms.Resize((SKIN_IMG_SIZE, SKIN_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION & ROUTING
# ══════════════════════════════════════════════════════════════════════════════
def predict_nail(model, img):
    a = np.expand_dims(np.array(img.convert("RGB").resize((NAIL_IMG_SIZE, NAIL_IMG_SIZE)), dtype=np.float32), 0)
    p = model.predict(a, verbose=0)[0]
    i = int(np.argmax(p))
    return NAIL_CLASSES[i], float(p[i]) * 100, {NAIL_CLASSES[j]: float(p[j]) * 100 for j in range(3)}

def predict_skin(model, img):
    with torch.no_grad():
        p = torch.softmax(model(SKIN_TF(img.convert("RGB")).unsqueeze(0)), dim=1)[0].numpy()
    i = int(p.argmax())
    return SKIN_CLASSES[i], float(p[i]) * 100, {SKIN_CLASSES[j]: float(p[j]) * 100 for j in range(5)}

def smart_route(image, nail_m, skin_m):
    nail_cls, nail_conf, nail_all = predict_nail(nail_m, image)
    skin_cls, skin_conf, skin_all = predict_skin(skin_m, image)
    # Both unconfident → unknown
    if nail_conf < 40.0 and skin_conf < 40.0:
        return "unknown", None, max(nail_conf, skin_conf), {}
    # Nail wins only if 20% more confident than skin
    if nail_conf > skin_conf * 1.2:
        return "nail", nail_cls, nail_conf, nail_all
    return "skin", skin_cls, skin_conf, skin_all

# ══════════════════════════════════════════════════════════════════════════════
# PDF REPORT
# ══════════════════════════════════════════════════════════════════════════════
def generate_pdf(image, pred, conf, all_conf, sev_lbl, is_veg=False):
    buf  = io.BytesIO()
    doc  = SimpleDocTemplate(buf, pagesize=A4,
                              leftMargin=2*cm, rightMargin=2*cm,
                              topMargin=2*cm, bottomMargin=2*cm)
    info = INFO[pred]
    SS   = getSampleStyleSheet()
    H1   = ParagraphStyle("H1",  parent=SS["Heading1"], fontSize=18, spaceAfter=4,  textColor=colors.HexColor("#0a2540"))
    H2   = ParagraphStyle("H2",  parent=SS["Heading2"], fontSize=13, spaceAfter=4,  textColor=colors.HexColor("#1a5fa8"), spaceBefore=14)
    BDY  = ParagraphStyle("BDY", parent=SS["Normal"],   fontSize=10, leading=15,    textColor=colors.HexColor("#374151"))
    SML  = ParagraphStyle("SML", parent=SS["Normal"],   fontSize=9,  leading=13,    textColor=colors.HexColor("#64748b"))
    story = []

    story.append(Paragraph("NutritionNet — Skin & Nail Health Scanner", H1))
    story.append(Paragraph(f"Report generated: {datetime.datetime.now().strftime('%d %B %Y, %I:%M %p')}", SML))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#dbeafe"), spaceAfter=12))

    ib = io.BytesIO()
    th = image.copy().convert("RGB")
    th.thumbnail((400, 400))
    th.save(ib, format="JPEG", quality=85)
    ib.seek(0)
    story.append(RLImage(ib, width=8*cm, height=8*cm, kind="proportional"))
    story.append(Spacer(1, 10))

    sev_color = {"Severe":"#dc2626","Moderate":"#ea580c","Mild":"#f59e0b"}.get(sev_lbl, "#10b981")
    tbl = Table([
        ["Diagnosis",  info["title"]],
        ["Condition",  info["sub"]],
        ["Confidence", f"{conf:.1f}%"],
        ["Severity",   sev_lbl if sev_lbl else "None (healthy)"],
    ], colWidths=[4*cm, 12*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0,0),(0,-1), colors.HexColor("#eff6ff")),
        ("FONTNAME",       (0,0),(0,-1), "Helvetica-Bold"),
        ("FONTSIZE",       (0,0),(-1,-1), 10),
        ("TEXTCOLOR",      (1,3),(1,3),  colors.HexColor(sev_color)),
        ("FONTNAME",       (1,3),(1,3),  "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0,0),(-1,-1),[colors.HexColor("#f8fafc"), colors.white]),
        ("GRID",           (0,0),(-1,-1), 0.5, colors.HexColor("#dbeafe")),
        ("PADDING",        (0,0),(-1,-1), 7),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 6))

    story.append(Paragraph("About this condition", H2))
    story.append(Paragraph(info["desc"], BDY))

    story.append(Paragraph("Confidence breakdown", H2))
    crow = [["Condition", "Confidence"]]
    for c, p in sorted(all_conf.items(), key=lambda x: -x[1])[:2]:
        crow.append([INFO[c]["title"], f"{p:.1f}%"])
    ct = Table(crow, colWidths=[12*cm, 4*cm])
    ct.setStyle(TableStyle([
        ("BACKGROUND",     (0,0),(-1,0),  colors.HexColor("#0a2540")),
        ("TEXTCOLOR",      (0,0),(-1,0),  colors.white),
        ("FONTNAME",       (0,0),(-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0,0),(-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1),(-1,-1), [colors.HexColor("#f8fafc"), colors.white]),
        ("GRID",           (0,0),(-1,-1), 0.5, colors.HexColor("#dbeafe")),
        ("PADDING",        (0,0),(-1,-1), 6),
        ("ALIGN",          (1,0),(1,-1),  "CENTER"),
    ]))
    story.append(ct)

    if info["tips"]:
        story.append(Paragraph("What you can do", H2))
        for i, (ic, t, tx) in enumerate(info["tips"], 1):
            story.append(Paragraph(f"<b>{i}. {t}:</b> {tx}", BDY))
            story.append(Spacer(1, 4))

    plan = DIET_PLANS.get(pred)
    if plan:
        story.append(Paragraph("7-Day Indian Diet Plan", H2))
        story.append(Paragraph(f"Focus: {plan['focus']}", SML))
        story.append(Spacer(1, 6))
        drows = [["Day", "Breakfast", "Lunch", "Dinner"]]
        for row in plan["days"]:
            d, b, l, dn = row
            if is_veg:
                b, l, dn = apply_veg_swaps(b), apply_veg_swaps(l), apply_veg_swaps(dn)
            drows.append([d, b, l, dn])
        dt = Table(drows, colWidths=[1.5*cm, 5*cm, 5*cm, 5.5*cm])
        dt.setStyle(TableStyle([
            ("BACKGROUND",     (0,0),(-1,0),  colors.HexColor("#0a2540")),
            ("TEXTCOLOR",      (0,0),(-1,0),  colors.white),
            ("FONTNAME",       (0,0),(-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",       (0,0),(-1,-1), 8),
            ("ROWBACKGROUNDS", (0,1),(-1,-1), [colors.HexColor("#f8fafc"), colors.white]),
            ("GRID",           (0,0),(-1,-1), 0.5, colors.HexColor("#dbeafe")),
            ("PADDING",        (0,0),(-1,-1), 5),
            ("VALIGN",         (0,0),(-1,-1), "TOP"),
            ("FONTNAME",       (0,1),(0,-1),  "Helvetica-Bold"),
        ]))
        story.append(dt)

    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#dbeafe")))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "For informational purposes only — not a substitute for professional medical advice. "
        "Always consult a qualified doctor or dermatologist.", SML))

    doc.build(story)
    buf.seek(0)
    return buf.read()

# ══════════════════════════════════════════════════════════════════════════════
# RESULT RENDERER
# ══════════════════════════════════════════════════════════════════════════════
def render_result(pred, conf, all_conf, img_type, image):
    info      = INFO[pred]
    color_map = NAIL_COLORS if img_type == "nail" else SKIN_COLORS
    sev       = SEVERITY_SCORE[pred]
    sev_lbl, sev_col = severity_label(sev)
    sev_gauge  = severity_gauge_color(sev)
    conf_int   = str(int(round(conf)))
    now_str    = datetime.datetime.now().strftime("%H:%M · %b %d")

    # Bar labels — nail shows nutrient name not disease name
    def get_bar_label(cls):
        if img_type == "nail":
            return NAIL_BAR_MAP.get(cls, INFO[cls]["title"])
        return INFO[cls]["title"]

    # Build bars HTML (top 2 only)
    bars_html = ""
    for cls, pct in sorted(all_conf.items(), key=lambda x: -x[1])[:2]:
        p_str = str(round(pct, 1))
        w_str = str(round(min(pct, 100), 1))
        col   = color_map.get(cls, "#1a5fa8")
        lbl   = get_bar_label(cls)
        bars_html += (
            '<div class="bar-row">'
            '<div class="bar-meta"><span>' + lbl + '</span>'
            '<span>' + p_str + '%</span></div>'
            '<div class="bar-bg"><div class="bar-fg" style="width:' + w_str + '%;background:' + col + '"></div></div>'
            '</div>'
        )

    # Food grid
    foods_html = ""
    for em, nm, nt in info["rich_foods"]:
        foods_html += (
            '<div class="food-card">'
            '<span class="food-emoji">' + em + '</span>'
            '<span class="food-name">' + nm + '</span>'
            + ('<span class="food-pill">' + nt + '</span>' if nt else '') +
            '</div>'
        )

    # Tips
    tips_block = ""
    if info["tips"]:
        rows = ""
        for i, (ic, t, tx) in enumerate(info["tips"], 1):
            rows += (
                '<div class="tip-card">'
                '<div class="tip-idx">' + str(i) + '</div>'
                '<div class="tip-body"><strong>' + ic + ' ' + t + ':</strong> ' + tx + '</div>'
                '</div>'
            )
        tips_block = (
            '<div class="sec-divider"></div>'
            '<div class="tips-section"><div class="section-title">What you can do</div>'
            + rows + '</div>'
        )

    # Symptom checker (skin only)
    symptom_block = ""
    if img_type == "skin" and conf >= CONF_THRESH:
        symptom_block = (
            '<div class="sec-divider"></div>'
            '<div class="symptom-section"><div class="symptom-box">'
            '<div class="symptom-title">🔍 Also check — B-vitamin symptoms not visible from skin alone</div>'
            '<div class="symptom-row">'
            '<span class="symptom-chip">Tingling hands → B-12</span>'
            '<span class="symptom-chip">Mouth sores → B2/B3</span>'
            '<span class="symptom-chip">Fatigue → B-12</span>'
            '<span class="symptom-chip">Hair thinning → Biotin</span>'
            '<span class="symptom-chip">Pale skin → B-12/Iron</span>'
            '</div>'
            '<p style="font-size:11px;color:#92400e;margin:10px 0 0">'
            'Ask your doctor for a complete vitamin panel blood test if you have these symptoms.</p>'
            '</div></div>'
        )

    # Low confidence note
    low_note = ""
    if conf < CONF_THRESH and pred != "healthy":
        low_note = (
            '<p style="font-size:12px;color:#6b7280;margin:8px 0 0;padding:8px 12px;'
            'background:#f8fafc;border-radius:8px;border-left:3px solid #dbeafe">'
            'Result based on ' + conf_int + '% confidence — try a closer, well-lit photo for best accuracy.</p>'
        )

    html = (
        '<div class="rcard">'
        '<div class="rcard-top">'
        '<div style="text-align:right;font-size:11px;color:#9ca3af;margin-bottom:8px">' + now_str + '</div>'
        '<div class="rcard-title">' + info["title"] + '</div>'
        '<div style="font-size:13px;color:#6b7280;margin-bottom:10px">' + info["sub"] + ' &nbsp;·&nbsp; ' + info["prevalence"] + '</div>'
        '<p class="rcard-desc">' + info["desc"] + '</p>'
        + low_note +
        '</div>'
        '<div class="sec-divider"></div>'

        '<div class="gauge-section"><div class="gauge-wrap">'
        '<div class="gauge-title">Severity assessment</div>'
        '<div class="gauge-track"><div class="gauge-fill" style="width:' + str(sev) + '%;background:' + sev_gauge + '"></div></div>'
        '<div class="gauge-labels"><span>None</span><span>Mild</span><span>Moderate</span><span>Severe</span></div>'
        '<div class="gauge-stats">'
        '<div class="gstat"><span class="gstat-val">' + conf_int + '%</span><span class="gstat-lbl">AI confidence</span></div>'
        '<div class="gstat"><span class="gstat-val" style="color:' + sev_col + '">' + sev_lbl + '</span><span class="gstat-lbl">Severity level</span></div>'
        ''
        '</div></div></div>'
        '<div class="sec-divider"></div>'

        '<div class="bars-section"><div class="section-title">Confidence breakdown</div>'
        + bars_html +
        '</div>'
        '<div class="sec-divider"></div>'

        '<div class="food-section"><div class="section-title">Top foods to eat</div>'
        '<div class="food-grid">' + foods_html + '</div>'
        '<div class="food-need">' + info["need"] + '</div>'
        '</div>'

        + tips_block + symptom_block +

        '<div class="disclaimer-bar">⚕️ For informational purposes only — not a substitute for professional medical advice.<br>'
        'Always consult a qualified doctor or dermatologist for proper diagnosis and treatment.</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

    # 7-day diet plan
    st.markdown("""
<div style="display:flex;align-items:center;gap:14px;background:#ffffff;
            border:1px solid #dbeafe;border-radius:14px;padding:14px 18px;
            margin:1.2rem 0 0.8rem;box-shadow:0 2px 8px rgba(10,37,64,0.05)">
  <span style="font-size:32px;flex-shrink:0">🍱</span>
  <div>
    <div style="font-family:Fraunces,serif;font-size:16px;font-weight:700;color:#0a2540;margin-bottom:2px">7-Day Indian Diet Plan</div>
    <div style="font-size:12px;color:#6b7280">Personalised for your condition · veg &amp; non-veg options</div>
  </div>
</div>
""", unsafe_allow_html=True)
    st.markdown("**What is your food preference?**")
    diet_key = f"diet_{pred}"
    current  = st.session_state.get(diet_key)

    # Segmented control via two plain columns buttons
    # Selected styles injected via CSS targeting the specific button keys
    nv_active = current == "non-veg"
    vg_active = current == "veg"
    st.markdown(f"""
<style>
div[data-testid="column"]:nth-of-type(1) [data-testid="stBaseButton-secondary"] button,
button[kind="secondary"][data-testid="baseButton-secondary"]{{}}
div:has(> button[data-testid="baseButton-secondary"][aria-label="🥩 Non-Veg"]) button{{
  {'background:#0a2540!important;color:#fff!important;border:none!important;' if nv_active else ''}
}}
</style>
""", unsafe_allow_html=True)

    _c1, _c2, _c3 = st.columns([1, 1, 2])
    with _c1:
        if nv_active:
            st.markdown('<style>#nv-btn button{background:#0a2540!important;color:#fff!important;border:none!important;}</style><span id="nv-btn">', unsafe_allow_html=True)
        if st.button("🥩 Non-Veg", key=f"nv_{pred}", use_container_width=True):
            st.session_state[diet_key] = "non-veg"
            st.rerun()
        if nv_active:
            st.markdown('</span>', unsafe_allow_html=True)
    with _c2:
        if vg_active:
            st.markdown('<style>#vg-btn button{background:#0a2540!important;color:#fff!important;border:none!important;}</style><span id="vg-btn">', unsafe_allow_html=True)
        if st.button("🥗 Vegetarian", key=f"vg_{pred}", use_container_width=True):
            st.session_state[diet_key] = "veg"
            st.rerun()
        if vg_active:
            st.markdown('</span>', unsafe_allow_html=True)

    if current:
        is_veg = current == "veg"
        plan   = DIET_PLANS.get(pred)
        if plan:
            pref_lbl = "🥗 Vegetarian" if is_veg else "🥩 Non-Vegetarian"
            st.markdown(
                f'<p style="font-size:12px;color:#6b7280;margin:6px 0 10px">'
                f'Focus: <strong>{plan["focus"]}</strong> &nbsp;·&nbsp; {pref_lbl}</p>',
                unsafe_allow_html=True
            )
            rows_html = ""
            today_day = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][__import__("datetime").datetime.now().weekday()]
            for ridx, (day, b, l, dn) in enumerate(plan["days"]):
                if is_veg:
                    b, l, dn = apply_veg_swaps(b), apply_veg_swaps(l), apply_veg_swaps(dn)
                is_today  = (day == today_day)
                row_bg    = "background:#eff6ff;border-left:3px solid #1a5fa8;" if is_today else ("background:#f8fafc;" if ridx%2==0 else "background:#ffffff;")
                day_label = "<strong style='color:#1a5fa8'>Today</strong>" if is_today else "<span style='color:#0a2540;font-weight:600'>" + day + "</span>"
                cell_col  = "#1a5fa8" if is_today else "#374151"
                rows_html += (
                    '<tr style="border-bottom:1px solid #f0f7ff;' + row_bg + '">'
                    '<td style="padding:9px 10px;white-space:nowrap;width:60px">' + day_label + '</td>'
                    '<td style="padding:9px 10px;color:' + cell_col + ';font-size:12px">' + b + '</td>'
                    '<td style="padding:9px 10px;color:' + cell_col + ';font-size:12px">' + l + '</td>'
                    '<td style="padding:9px 10px;color:' + cell_col + ';font-size:12px">' + dn + '</td>'
                    '</tr>'
                )
            st.markdown(
                '<div style="overflow-x:auto;border-radius:10px;border:1px solid #dbeafe;margin-bottom:1rem">'
                '<table style="width:100%;border-collapse:collapse">'
                '<thead><tr style="background:#0a2540;color:white">'
                '<th style="padding:9px 10px;text-align:left;font-size:12px;font-weight:600;white-space:nowrap">Day</th>'
                '<th style="padding:9px 10px;text-align:left;font-size:12px;font-weight:600">Breakfast</th>'
                '<th style="padding:9px 10px;text-align:left;font-size:12px;font-weight:600">Lunch</th>'
                '<th style="padding:9px 10px;text-align:left;font-size:12px;font-weight:600">Dinner</th>'
                '</tr></thead>'
                '<tbody>' + rows_html + '</tbody>'
                '</table></div>',
                unsafe_allow_html=True
            )

    # PDF download — fused card with inline download link
    is_veg_pdf = st.session_state.get(diet_key) == "veg"
    with st.spinner("Generating PDF…"):
        pdf_bytes = generate_pdf(image, pred, conf, all_conf, sev_lbl, is_veg_pdf)
    fname = f"NutritionNet_{pred.replace(' ','_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"

    import base64 as _b64pdf
    pdf_b64 = _b64pdf.b64encode(pdf_bytes).decode()
    st.markdown(f"""
<div style="background:linear-gradient(135deg,#0a2540 0%,#1a5fa8 100%);
            border-radius:14px;padding:18px 22px;margin-top:1rem;
            display:flex;align-items:center;justify-content:space-between;gap:16px">
  <div>
    <div style="font-family:Fraunces,serif;font-size:16px;font-weight:600;
                color:#ffffff;margin-bottom:3px">Your health report is ready</div>
    <div style="font-size:12px;color:rgba(255,255,255,0.6)">Diagnosis · diet · tips · PDF</div>
  </div>
  <a href="data:application/pdf;base64,{pdf_b64}"
     download="{fname}"
     style="background:#ffffff;color:#0a2540;border-radius:10px;
            padding:10px 22px;font-size:13px;font-weight:700;
            text-decoration:none;white-space:nowrap;flex-shrink:0;
            display:inline-flex;align-items:center;gap:6px">
    ⬇ PDF
  </a>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════════════════════════════
nail_model = skin_model = None
nail_err   = skin_err   = None
try:
    nail_model = load_nail_model()
except Exception as e:
    nail_err = str(e)
try:
    skin_model = load_skin_model()
except Exception as e:
    skin_err = str(e)

# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-left">
    <div class="hero-badge">AI Health Scanner</div>
    <h1>Nutrition<span>Net</span></h1>
    <p class="hero-sub">Nail or skin image → instant AI diagnosis → 7-day personalised Indian diet plan → PDF report</p>
    <div class="hero-pills">
      <span class="hero-pill">🧴 Vitamin deficiencies</span>
      <span class="hero-pill">🍱 Indian diet plan</span>
      <span class="hero-pill">📄 PDF report</span>
    </div>
  </div>
  <div class="hero-right"><div class="hero-ring-outer"></div><div class="hero-ring-inner"></div><div class="hero-icon-wrap">🔬<div class="hero-check">✓</div></div></div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["🔬 Analyse", "ℹ️ About"])

# ── TAB 1 ────────────────────────────────────────────────────────────────────
with tab1:
    nail_ok = nail_model is not None
    skin_ok = skin_model is not None

    if not nail_ok or not skin_ok:
        st.error("One or both models failed to load. Check the model paths.")
        st.stop()

    # Two-column layout: left = upload, right = results
    col_left, col_right = st.columns([1, 1.6], gap="large")

    with col_left:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown('<span class="upload-label">Upload your image</span>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Drop your nail or skin image here or click to browse",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed"
        )

        if uploaded:
            # show image in fixed-height box
            img_obj  = Image.open(uploaded).convert("RGB")
            buf_prev = io.BytesIO()
            img_obj.save(buf_prev, format="JPEG", quality=85)
            import base64 as _b64
            b64_str = _b64.b64encode(buf_prev.getvalue()).decode()
            st.markdown(
                f'<div style="width:100%;height:260px;border-radius:12px;overflow:hidden;'
                f'border:1px solid #dbeafe;margin-bottom:10px">'
                f'<img src="data:image/jpeg;base64,{b64_str}" '
                f'style="width:100%;height:100%;object-fit:cover;display:block"></div>',
                unsafe_allow_html=True
            )
            run = st.button("Analyse now →", use_container_width=True)

        nail_dot2 = "#10b981" if nail_ok else "#ef4444"
        skin_dot2 = "#10b981" if skin_ok else "#ef4444"

    with col_right:
        # Run analysis and cache result so diet radio doesn't wipe it
        if uploaded and "run" in dir() and run:
            image = Image.open(uploaded)
            with st.spinner("🔍 Running both AI models on your image…"):
                img_type, pred, conf, all_conf = smart_route(image, nail_model, skin_model)
            st.session_state["last_result"] = {
                "img_type": img_type, "pred": pred,
                "conf": conf, "all_conf": all_conf,
            }
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            st.session_state["last_image_bytes"] = buf.getvalue()

        # Always show cached result (persists across radio reruns)
        if "last_result" in st.session_state and uploaded:
            r = st.session_state["last_result"]
            img_bytes = st.session_state.get("last_image_bytes")
            cached_image = Image.open(io.BytesIO(img_bytes)) if img_bytes else Image.open(uploaded)

            st.markdown('<div class="page-body" style="padding:0">', unsafe_allow_html=True)
            if r["img_type"] == "unknown":
                st.markdown(
                    '<div style="background:#fefce8;border:1px solid #fde68a;border-radius:12px;'
                    'padding:16px 20px;font-size:14px;color:#713f12;line-height:1.7">'
                    '<strong>Image not recognised.</strong><br><br>'
                    'Please upload a <strong>close-up of a single nail</strong> or a '
                    '<strong>close-up of affected skin</strong>. '
                    'Make sure the area fills most of the frame in good lighting.'
                    '</div>',
                    unsafe_allow_html=True
                )
            else:
                render_result(r["pred"], r["conf"], r["all_conf"], r["img_type"], cached_image)
            st.markdown('</div>', unsafe_allow_html=True)
        elif not uploaded:
            st.markdown('''
<div style="background:#ffffff;border-radius:20px;border:1px solid #dbeafe;
            padding:28px 28px;box-shadow:0 4px 24px rgba(10,37,64,0.06);">
  <p style="font-family:Fraunces,serif;font-size:18px;font-weight:600;
             color:#0a2540;margin:0 0 18px">What you'll get</p>
  <div style="display:flex;flex-direction:column;gap:14px">
    <div style="display:flex;align-items:center;gap:12px">
      <span style="font-size:22px;width:32px;text-align:center">📊</span>
      <span style="font-size:14px;color:#374151">AI confidence breakdown</span>
    </div>
    <div style="display:flex;align-items:center;gap:12px">
      <span style="font-size:22px;width:32px;text-align:center">🍱</span>
      <span style="font-size:14px;color:#374151">7-day Indian diet plan (veg & non-veg)</span>
    </div>
    <div style="display:flex;align-items:center;gap:12px">
      <span style="font-size:22px;width:32px;text-align:center">📄</span>
      <span style="font-size:14px;color:#374151">Downloadable PDF health report</span>
    </div>
    <div style="display:flex;align-items:center;gap:12px">
      <span style="font-size:22px;width:32px;text-align:center">🥗</span>
      <span style="font-size:14px;color:#374151">Top foods to eat for your condition</span>
    </div>
  </div>
  <div style="margin-top:22px;padding-top:16px;border-top:1px solid #f0f7ff;
              font-size:11px;color:#94a3b8;text-align:center">
    Supports nail diseases · vitamin deficiencies · skin conditions
  </div>
</div>
''', unsafe_allow_html=True)

# ── TAB 2 ────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div style="padding:24px 0">', unsafe_allow_html=True)
    st.markdown("""
<div class="about-card">
  <h3>🔬 How NutritionNet works</h3>
  <p style="font-size:14px;color:#4b5563;line-height:1.7">
    Upload any nail or skin close-up. NutritionNet runs both AI models simultaneously and picks
    the one that's more confident — with the nail model requiring 20% higher confidence to
    win (skin is the broader, safer default). Results include a severity gauge, confidence
    breakdown, food recommendations, tips, 7-day Indian diet plan, and a downloadable PDF report.
  </p>
</div>
<div class="about-card">
  <h3>🥗 7-Day Indian Diet Plan</h3>
  <p style="font-size:14px;color:#4b5563;line-height:1.7">
    Every detected condition comes with a personalised 7-day Indian diet plan —
    affordable, widely available foods targeting the specific deficient nutrient.
    Choose Non-Vegetarian or Vegetarian — all meat/fish items are automatically swapped
    with Indian vegetarian equivalents (paneer, rajma, chana, soya chunks, tofu).
  </p>
</div>
<div class="about-card">
  <h3>⚕️ Medical disclaimer</h3>
  <p style="font-size:14px;color:#4b5563;line-height:1.7">
    NutritionNet is for educational and informational purposes only. It does not constitute medical
    advice, diagnosis, or treatment. AI confidence scores are not clinical accuracy measures.
    Always consult a qualified dermatologist or doctor for any health concern.
  </p>
</div>
""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)