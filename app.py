import os
import time
import base64
import random
from io import BytesIO

import numpy as np
import requests
import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import plotly.graph_objects as go
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torchvision import datasets

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='PawsClassifier',
    page_icon='🐾',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown('''
<style>
    [data-testid="stAppViewContainer"] { background: #0e1117; }
    [data-testid="stSidebar"] { background: #161b27; border-right: 1px solid #e94560; }
    h1, h2, h3 { color: #f0f0f0; }
    .breed-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #e94560;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 24px rgba(233,69,96,0.25);
        color: white;
        margin: 12px 0;
    }
    .breed-card h2 { color: #e94560; margin: 0 0 4px 0; font-size: 1.6em; }
    .breed-card .subtitle { color: #888; margin: 0 0 16px 0; font-size: 0.9em; }
    .info-grid {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 10px;
    }
    .info-box {
        background: #0f3460;
        border-radius: 10px;
        padding: 12px 8px;
        text-align: center;
        font-size: 0.85em;
        line-height: 1.5;
    }
    .stat-card {
        background: #161b27;
        border: 1px solid #2a2f3e;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        margin: 4px 0;
    }
    .stat-card .value { font-size: 1.8em; font-weight: bold; color: #e94560; }
    .stat-card .label { font-size: 0.8em; color: #888; margin-top: 4px; }
    .correct { color: #00cc44; font-weight: bold; }
    .wrong { color: #ff4444; font-weight: bold; }
    .result-row {
        background: #161b27;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 4px 0;
        border-left: 3px solid #2a2f3e;
        font-size: 0.88em;
    }
    .section-header {
        color: #e94560;
        font-size: 1.1em;
        font-weight: bold;
        margin: 16px 0 8px 0;
        border-bottom: 1px solid #2a2f3e;
        padding-bottom: 6px;
    }
    .paw-divider { text-align: center; color: #e94560; letter-spacing: 8px; margin: 12px 0; }
    .warning-box {
        background: #1a1200;
        border: 1px solid #ffaa00;
        border-radius: 8px;
        padding: 12px;
        color: #ffaa00;
        font-size: 0.85em;
    }
</style>
''', unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'best_model.pth'

CLASS_NAMES = [
    'Abyssinian', 'American Bulldog', 'American Pit Bull Terrier',
    'Basset Hound', 'Beagle', 'Bengal', 'Birman', 'Bombay', 'Boxer',
    'British Shorthair', 'Chihuahua', 'Egyptian Mau', 'English Cocker Spaniel',
    'English Setter', 'German Shorthaired', 'Great Pyrenees', 'Havanese',
    'Japanese Chin', 'Keeshond', 'Leonberger', 'Maine Coon',
    'Miniature Pinscher', 'Newfoundland', 'Persian', 'Pomeranian', 'Pug',
    'Ragdoll', 'Russian Blue', 'Saint Bernard', 'Samoyed', 'Scottish Terrier',
    'Shiba Inu', 'Siamese', 'Sphynx', 'Staffordshire Bull Terrier',
    'Wheaten Terrier', 'Yorkshire Terrier'
]

BREED_PROFILES = {
    'Abyssinian': {'origin': '🌍 Ethiopia', 'weight': '⚖️ 3–5 kg', 'temperament': '❤️ Curious, active, playful', 'type': 'cat'},
    'American Bulldog': {'origin': '🌍 USA', 'weight': '⚖️ 27–54 kg', 'temperament': '❤️ Friendly, assertive, loyal', 'type': 'dog'},
    'American Pit Bull Terrier': {'origin': '🌍 USA/UK', 'weight': '⚖️ 14–27 kg', 'temperament': '❤️ Loyal, playful, obedient', 'type': 'dog'},
    'Basset Hound': {'origin': '🌍 France', 'weight': '⚖️ 20–29 kg', 'temperament': '❤️ Gentle, tenacious, friendly', 'type': 'dog'},
    'Beagle': {'origin': '🌍 England', 'weight': '⚖️ 9–11 kg', 'temperament': '❤️ Curious, merry, friendly', 'type': 'dog'},
    'Bengal': {'origin': '🌍 USA', 'weight': '⚖️ 3.6–7 kg', 'temperament': '❤️ Alert, agile, playful', 'type': 'cat'},
    'Birman': {'origin': '🌍 Myanmar', 'weight': '⚖️ 3.5–7 kg', 'temperament': '❤️ Gentle, calm, social', 'type': 'cat'},
    'Bombay': {'origin': '🌍 USA', 'weight': '⚖️ 3.5–5 kg', 'temperament': '❤️ Affectionate, playful, curious', 'type': 'cat'},
    'Boxer': {'origin': '🌍 Germany', 'weight': '⚖️ 25–32 kg', 'temperament': '❤️ Playful, devoted, fearless', 'type': 'dog'},
    'British Shorthair': {'origin': '🌍 UK', 'weight': '⚖️ 4–8 kg', 'temperament': '❤️ Calm, patient, independent', 'type': 'cat'},
    'Chihuahua': {'origin': '🌍 Mexico', 'weight': '⚖️ 1.5–3 kg', 'temperament': '❤️ Alert, spirited, loyal', 'type': 'dog'},
    'Egyptian Mau': {'origin': '🌍 Egypt', 'weight': '⚖️ 3–5 kg', 'temperament': '❤️ Active, loyal, playful', 'type': 'cat'},
    'English Cocker Spaniel': {'origin': '🌍 England', 'weight': '⚖️ 12–15 kg', 'temperament': '❤️ Cheerful, trainable, gentle', 'type': 'dog'},
    'English Setter': {'origin': '🌍 England', 'weight': '⚖️ 20–36 kg', 'temperament': '❤️ Gentle, friendly, active', 'type': 'dog'},
    'German Shorthaired': {'origin': '🌍 Germany', 'weight': '⚖️ 20–32 kg', 'temperament': '❤️ Intelligent, bold, affectionate', 'type': 'dog'},
    'Great Pyrenees': {'origin': '🌍 France/Spain', 'weight': '⚖️ 36–54 kg', 'temperament': '❤️ Patient, calm, strong-willed', 'type': 'dog'},
    'Havanese': {'origin': '🌍 Cuba', 'weight': '⚖️ 3–6 kg', 'temperament': '❤️ Cheerful, gentle, social', 'type': 'dog'},
    'Japanese Chin': {'origin': '🌍 Japan', 'weight': '⚖️ 1.4–4.5 kg', 'temperament': '❤️ Charming, noble, loving', 'type': 'dog'},
    'Keeshond': {'origin': '🌍 Netherlands', 'weight': '⚖️ 14–18 kg', 'temperament': '❤️ Outgoing, friendly, agile', 'type': 'dog'},
    'Leonberger': {'origin': '🌍 Germany', 'weight': '⚖️ 41–77 kg', 'temperament': '❤️ Gentle, friendly, loyal', 'type': 'dog'},
    'Maine Coon': {'origin': '🌍 USA', 'weight': '⚖️ 4–9 kg', 'temperament': '❤️ Gentle, sociable, playful', 'type': 'cat'},
    'Miniature Pinscher': {'origin': '🌍 Germany', 'weight': '⚖️ 3.6–4.5 kg', 'temperament': '❤️ Alert, fearless, energetic', 'type': 'dog'},
    'Newfoundland': {'origin': '🌍 Canada', 'weight': '⚖️ 45–70 kg', 'temperament': '❤️ Gentle, trainable, sweet', 'type': 'dog'},
    'Persian': {'origin': '🌍 Iran', 'weight': '⚖️ 3–6 kg', 'temperament': '❤️ Quiet, gentle, affectionate', 'type': 'cat'},
    'Pomeranian': {'origin': '🌍 Germany', 'weight': '⚖️ 1.4–3.2 kg', 'temperament': '❤️ Bold, lively, curious', 'type': 'dog'},
    'Pug': {'origin': '🌍 China', 'weight': '⚖️ 6–9 kg', 'temperament': '❤️ Charming, playful, loving', 'type': 'dog'},
    'Ragdoll': {'origin': '🌍 USA', 'weight': '⚖️ 4.5–9 kg', 'temperament': '❤️ Gentle, relaxed, affectionate', 'type': 'cat'},
    'Russian Blue': {'origin': '🌍 Russia', 'weight': '⚖️ 3.5–7 kg', 'temperament': '❤️ Gentle, reserved, intelligent', 'type': 'cat'},
    'Saint Bernard': {'origin': '🌍 Switzerland', 'weight': '⚖️ 64–120 kg', 'temperament': '❤️ Gentle, friendly, patient', 'type': 'dog'},
    'Samoyed': {'origin': '🌍 Russia', 'weight': '⚖️ 16–30 kg', 'temperament': '❤️ Gentle, adaptable, friendly', 'type': 'dog'},
    'Scottish Terrier': {'origin': '🌍 Scotland', 'weight': '⚖️ 8–10 kg', 'temperament': '❤️ Feisty, alert, independent', 'type': 'dog'},
    'Shiba Inu': {'origin': '🌍 Japan', 'weight': '⚖️ 8–11 kg', 'temperament': '❤️ Alert, confident, spirited', 'type': 'dog'},
    'Siamese': {'origin': '🌍 Thailand', 'weight': '⚖️ 3.5–5.5 kg', 'temperament': '❤️ Vocal, social, intelligent', 'type': 'cat'},
    'Sphynx': {'origin': '🌍 Canada', 'weight': '⚖️ 3.5–7 kg', 'temperament': '❤️ Energetic, loyal, curious', 'type': 'cat'},
    'Staffordshire Bull Terrier': {'origin': '🌍 England', 'weight': '⚖️ 11–17 kg', 'temperament': '❤️ Brave, tenacious, affectionate', 'type': 'dog'},
    'Wheaten Terrier': {'origin': '🌍 Ireland', 'weight': '⚖️ 14–20 kg', 'temperament': '❤️ Playful, confident, happy', 'type': 'dog'},
    'Yorkshire Terrier': {'origin': '🌍 England', 'weight': '⚖️ 2–3 kg', 'temperament': '❤️ Bold, confident, intelligent', 'type': 'dog'},
}

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    m = models.resnet18(weights=None)
    m.fc = torch.nn.Linear(512, 37)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    m.to(DEVICE)
    m.eval()
    return m

@st.cache_resource
def load_test_dataset():
    full = datasets.OxfordIIITPet(root='data', download=False, transform=TRANSFORM)
    indices = list(range(len(full)))
    _, temp = train_test_split(indices, test_size=0.2, random_state=42)
    _, test_idx = train_test_split(temp, test_size=0.5, random_state=42)
    return Subset(full, test_idx)

@st.cache_resource
def get_misclassifications(_model, _dataset, n=6):
    wrong = []
    for i in range(len(_dataset)):
        img_tensor, label = _dataset[i]
        with torch.no_grad():
            out = _model(img_tensor.unsqueeze(0).to(DEVICE))
            pred = out.argmax(1).item()
        if pred != label:
            wrong.append((img_tensor, label, pred))
        if len(wrong) >= n:
            break
    return wrong

# ── Helpers ───────────────────────────────────────────────────────────────────
def predict(model, img):
    tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)[0]
    top5_probs, top5_idx = probs.topk(5)
    return [(CLASS_NAMES[idx.item()], prob.item()) for idx, prob in zip(top5_idx, top5_probs)]

def tensor_to_pil(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (tensor * std + mean).clamp(0, 1)
    arr = (img.permute(1, 2, 0).numpy() * 255).astype('uint8')
    return Image.fromarray(arr)

def pil_to_b64(img):
    buf = BytesIO()
    img.save(buf, format='JPEG')
    return base64.b64encode(buf.getvalue()).decode()

def confidence_gauge(confidence):
    if confidence >= 0.75:
        bar_color = '#00cc44'
    elif confidence >= 0.50:
        bar_color = '#ffaa00'
    else:
        bar_color = '#ff4444'

    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=round(confidence * 100, 1),
        number={'suffix': '%', 'font': {'size': 32, 'color': bar_color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#444'},
            'bar': {'color': bar_color, 'thickness': 0.28},
            'bgcolor': '#1a1a2e',
            'borderwidth': 1,
            'bordercolor': '#333',
            'steps': [
                {'range': [0, 50], 'color': '#2a0a0a'},
                {'range': [50, 75], 'color': '#2a1a00'},
                {'range': [75, 100], 'color': '#0a2a0a'},
            ],
            'threshold': {
                'line': {'color': bar_color, 'width': 4},
                'thickness': 0.8,
                'value': confidence * 100
            }
        },
        title={'text': 'Model Confidence', 'font': {'size': 14, 'color': '#aaa'}}
    ))
    fig.update_layout(
        height=220,
        margin=dict(t=40, b=0, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}
    )
    return fig

def top5_bar(predictions):
    breeds = [p[0] for p in predictions][::-1]
    confs = [p[1] * 100 for p in predictions][::-1]
    colors = ['#e94560' if i == 4 else '#0f3460' for i in range(5)]

    fig = go.Figure(go.Bar(
        x=confs,
        y=breeds,
        orientation='h',
        marker_color=colors,
        text=[f'{c:.1f}%' for c in confs],
        textposition='outside',
        textfont={'color': 'white', 'size': 11}
    ))
    fig.update_layout(
        height=220,
        margin=dict(t=8, b=8, l=4, r=50),
        xaxis=dict(range=[0, 115], showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, tickfont={'size': 11, 'color': 'white'}),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}
    )
    return fig

def render_breed_card(name, confidence):
    profile = BREED_PROFILES.get(name, {})
    icon = '🐱' if profile.get('type') == 'cat' else '🐶'
    st.markdown(f'''
    <div class="breed-card">
        <h2>{icon} {name}</h2>
        <p class="subtitle">Identified with {confidence:.1%} confidence</p>
        <div class="info-grid">
            <div class="info-box">{profile.get("origin", "N/A")}</div>
            <div class="info-box">{profile.get("weight", "N/A")}</div>
            <div class="info-box">{profile.get("temperament", "N/A")}</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

def ask_gemini(img: Image.Image) -> str:
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        return 'API key not set'
    client = genai.Client(api_key=api_key)
    prompt = (
        'You are a pet breed classifier. Given this image, identify the breed. '
        'You must respond with ONLY one of these exact class names, nothing else: '
        + ', '.join(CLASS_NAMES)
    )
    response = client.models.generate_content(
        model='gemini-2.5-flash-lite',
        contents=[
            types.Part.from_bytes(
                data=base64.b64decode(pil_to_b64(img)),
                mime_type='image/jpeg'
            ),
            prompt
        ]
    )
    return response.text.strip()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('''
    <div style="text-align:center; padding: 16px 0;">
        <div style="font-size:3em;">🐾</div>
        <h2 style="color:#e94560; margin:4px 0;">PawsClassifier</h2>
        <p style="color:#888; font-size:0.8em;">Oxford-IIIT Pets · ResNet18</p>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('---')
    st.markdown('<div class="section-header">🤖 Model Info</div>', unsafe_allow_html=True)

    for label, val in [
        ('Architecture', 'ResNet18'),
        ('Strategy', 'Transfer Learning'),
        ('Dataset', 'Oxford-IIIT Pets'),
        ('Classes', '37 breeds'),
        ('Device', str(DEVICE).upper()),
    ]:
        st.markdown(f'''
        <div style="display:flex; justify-content:space-between; padding:4px 0;
                    border-bottom:1px solid #1e2330; font-size:0.82em;">
            <span style="color:#888;">{label}</span>
            <span style="color:#f0f0f0; font-weight:500;">{val}</span>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown('<div class="section-header" style="margin-top:16px;">📊 Test Performance</div>', unsafe_allow_html=True)

    for label, val in [
        ('Accuracy', '88.2%'),
        ('Macro F1', '0.881'),
        ('Epochs', '21 (early stop)'),
        ('Optimizer', 'Adam · lr=0.001'),
    ]:
        st.markdown(f'''
        <div style="display:flex; justify-content:space-between; padding:4px 0;
                    border-bottom:1px solid #1e2330; font-size:0.82em;">
            <span style="color:#888;">{label}</span>
            <span style="color:#e94560; font-weight:600;">{val}</span>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown('---')
    st.markdown('<p style="color:#555; font-size:0.75em; text-align:center;">Fanshawe College · Deep Learning with Pytorch</p>', unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('''
<div style="
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    border: 1px solid #e94560;
    box-shadow: 0 4px 32px rgba(233,69,96,0.2);
">
    <h1 style="margin:0; color:#f0f0f0; font-size:2.2em;">
        🐾 PawsClassifier
    </h1>
    <p style="margin:8px 0 0 0; color:#aaa; font-size:1em;">
        Fine-tuned ResNet18 · 37 pet breeds · Oxford-IIIT Pets dataset
    </p>
</div>
''', unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
model = load_model()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    '🔍 Classify',
    '🏆 Model vs Gemini',
    '🤔 Where It Fails',
    '📊 Full Results'
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Classify
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    col_input, col_results = st.columns([1, 1], gap='large')

    with col_input:
        st.markdown('### 📸 Input Image')
        method = st.radio('Source', ['🌐 Image URL', '📁 Upload file'], horizontal=True, label_visibility='collapsed')

        pil_img = None

        if method == '🌐 Image URL':
            url = st.text_input('Paste image URL', placeholder='https://example.com/dog.jpg')
            if url:
                try:
                    r = requests.get(url, timeout=8)
                    pil_img = Image.open(BytesIO(r.content)).convert('RGB')
                    st.image(pil_img, caption='Preview', use_container_width=True)
                except Exception:
                    st.error('Could not load image from URL. Check the link and try again.')
        else:
            uploaded = st.file_uploader('Drop image here', type=['jpg', 'jpeg', 'png'])
            if uploaded:
                pil_img = Image.open(uploaded).convert('RGB')
                st.image(pil_img, use_container_width=True)

        classify_btn = st.button(
            '🔍 Classify Breed',
            disabled=pil_img is None,
            use_container_width=True,
            type='primary'
        )

    with col_results:
        st.markdown('### 🎯 Results')

        if classify_btn and pil_img is not None:
            with st.spinner('🐾 Sniffing the image...'):
                predictions = predict(model, pil_img)

            top_breed, top_conf = predictions[0]

            # Gauge
            st.plotly_chart(confidence_gauge(top_conf), use_container_width=True)

            # Breed card
            render_breed_card(top_breed, top_conf)

            # Top 5 bar chart
            st.markdown('<div class="section-header">Top 5 Predictions</div>', unsafe_allow_html=True)
            st.plotly_chart(top5_bar(predictions), use_container_width=True)
        else:
            st.markdown('''
            <div style="
                border: 2px dashed #2a2f3e;
                border-radius: 16px;
                padding: 60px 20px;
                text-align: center;
                color: #555;
                margin-top: 40px;
            ">
                <div style="font-size: 3em; margin-bottom: 12px;">🐾</div>
                <p style="font-size: 1em;">Paste a URL or upload an image<br>to identify the breed</p>
            </div>
            ''', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Model vs Gemini
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('### 🏆 ResNet18 (fine-tuned) vs Gemini 2.5 Flash-Lite')
    st.markdown('''
    <p style="color:#aaa; font-size:0.9em;">
        A task-specific model trained on ~3,000 images vs a general-purpose vision LLM.
        Each image is sent to both models and predictions are compared side-by-side.
    </p>
    ''', unsafe_allow_html=True)

    st.markdown('''
    <div class="warning-box">
        ⚠️ Gemini API free tier: <strong>20 requests/day per project</strong>.
        Use 3–5 samples to preserve quota. Results are cached during this session.
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('')

    api_key_set = bool(os.getenv('GOOGLE_API_KEY'))
    if not api_key_set:
        st.error('GOOGLE_API_KEY not found in .env — Gemini column will be unavailable.')

    col_ctrl1, col_ctrl2 = st.columns([1, 3])
    with col_ctrl1:
        n_samples = st.selectbox('Batch size', [3, 4, 5], index=0)

    run_btn = st.button('🐾 Run Comparison', type='primary', use_container_width=False)

    if run_btn:
        test_data = load_test_dataset()
        total = len(test_data)
        chosen = random.sample(range(total), n_samples)

        results = []
        model_correct = 0
        gemini_correct = 0

        progress = st.progress(0, text='🐾 Fetching images...')
        status = st.empty()

        for i, idx in enumerate(chosen):
            img_tensor, label = test_data[idx]
            real = CLASS_NAMES[label]
            pil = tensor_to_pil(img_tensor)

            # ResNet prediction
            preds = predict(model, pil)
            model_pred, model_conf = preds[0]
            model_ok = model_pred == real
            if model_ok:
                model_correct += 1

            # Gemini prediction
            gemini_pred = 'N/A'
            gemini_ok = False
            if api_key_set:
                try:
                    status.markdown(f'<p style="color:#aaa; font-size:0.85em;">🌐 Asking Gemini about image {i+1}/{n_samples}...</p>', unsafe_allow_html=True)
                    gemini_pred = ask_gemini(pil)
                    gemini_ok = gemini_pred == real
                    if gemini_ok:
                        gemini_correct += 1
                    if i < n_samples - 1:
                        time.sleep(13)
                except Exception as e:
                    gemini_pred = f'Error: {str(e)[:40]}'

            results.append({
                'img': pil,
                'real': real,
                'model_pred': model_pred,
                'model_conf': model_conf,
                'model_ok': model_ok,
                'gemini_pred': gemini_pred,
                'gemini_ok': gemini_ok
            })

            progress.progress((i + 1) / n_samples, text=f'🐾 Processed {i+1}/{n_samples}')

        progress.empty()
        status.empty()

        st.session_state['batch_results'] = results
        st.session_state['batch_n'] = n_samples
        st.session_state['model_correct'] = model_correct
        st.session_state['gemini_correct'] = gemini_correct

    if 'batch_results' in st.session_state:
        results = st.session_state['batch_results']
        n = st.session_state['batch_n']
        model_correct = st.session_state['model_correct']
        gemini_correct = st.session_state['gemini_correct']

        st.markdown('---')

        # Summary metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f'''
            <div class="stat-card">
                <div class="value">{model_correct}/{n}</div>
                <div class="label">🤖 ResNet18 correct</div>
            </div>
            ''', unsafe_allow_html=True)
        with m2:
            st.markdown(f'''
            <div class="stat-card">
                <div class="value">{gemini_correct}/{n}</div>
                <div class="label">✨ Gemini correct</div>
            </div>
            ''', unsafe_allow_html=True)
        with m3:
            winner = 'Tie 🤝' if model_correct == gemini_correct else ('ResNet18 🏆' if model_correct > gemini_correct else 'Gemini 🏆')
            st.markdown(f'''
            <div class="stat-card">
                <div class="value" style="font-size:1.2em;">{winner}</div>
                <div class="label">Round winner</div>
            </div>
            ''', unsafe_allow_html=True)

        st.markdown('<div class="paw-divider">🐾 🐾 🐾</div>', unsafe_allow_html=True)

        # Per-image results
        for r in results:
            img_col, info_col = st.columns([1, 3], gap='medium')
            with img_col:
                st.image(r['img'], use_container_width=True)
            with info_col:
                st.markdown(f'**Real breed:** `{r["real"]}`')

                m_icon = '✅' if r['model_ok'] else '❌'
                g_icon = '✅' if r['gemini_ok'] else '❌'

                st.markdown(f'''
                <div class="result-row">
                    {m_icon} <strong>ResNet18:</strong> {r["model_pred"]}
                    &nbsp;&nbsp;<span style="color:#888; font-size:0.85em;">({r["model_conf"]:.1%} confidence)</span>
                </div>
                <div class="result-row">
                    {g_icon} <strong>Gemini 2.5:</strong> {r["gemini_pred"]}
                </div>
                ''', unsafe_allow_html=True)

            st.markdown('<div style="border-bottom:1px solid #1e2330; margin:12px 0;"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Where It Fails
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('### 🤔 Where Does the Model Get Confused?')
    st.markdown('''
    <p style="color:#aaa; font-size:0.9em;">
        These are real test set errors — cases where visually similar breeds fooled the model.
        Understanding failure modes is as important as knowing accuracy.
    </p>
    ''', unsafe_allow_html=True)

    test_data_fails = load_test_dataset()
    misses = get_misclassifications(model, test_data_fails, n=6)

    cols = st.columns(3)
    for i, (tensor, true_label, pred_label) in enumerate(misses):
        pil = tensor_to_pil(tensor)
        true_name = CLASS_NAMES[true_label]
        pred_name = CLASS_NAMES[pred_label]
        true_icon = '🐱' if BREED_PROFILES.get(true_name, {}).get('type') == 'cat' else '🐶'

        with cols[i % 3]:
            st.image(pil, use_container_width=True)
            st.markdown(f'''
            <div style="
                background:#161b27; border-radius:10px; padding:12px;
                border-left: 3px solid #e94560; margin-bottom:16px;
                font-size:0.82em;
            ">
                <div style="color:#888; margin-bottom:4px;">True label</div>
                <div style="color:#00cc44; font-weight:bold;">{true_icon} {true_name}</div>
                <div style="color:#888; margin: 8px 0 4px 0;">Model predicted</div>
                <div style="color:#ff4444; font-weight:bold;">❌ {pred_name}</div>
            </div>
            ''', unsafe_allow_html=True)

    st.markdown('---')
    st.markdown('''
    <div style="background:#161b27; border-radius:12px; padding:20px; border:1px solid #2a2f3e;">
        <h4 style="color:#e94560; margin-top:0;">💡 Why does this happen?</h4>
        <p style="color:#aaa; font-size:0.9em; margin:0;">
            Fine-grained classification is hard even for humans. Visually similar breeds
            (e.g. Great Pyrenees vs Leonberger, Russian Blue vs British Shorthair) share coat
            color, body shape, and facial structure. The model was trained on ~2,900 images
            with frozen backbone weights — only the final classification layer was updated.
            Unfreezing deeper layers or training with more data could reduce these errors.
        </p>
    </div>
    ''', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Full Results
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('### 📊 Full Test Set Results')

    preds_path = 'all_preds.npy'
    labels_path = 'all_labels.npy'

    if not os.path.exists(preds_path) or not os.path.exists(labels_path):
        st.warning('''
        Saved prediction arrays not found.
        Run this cell in your notebook first:

        ```python
        import numpy as np
        np.save("all_preds.npy", np.array(all_preds))
        np.save("all_labels.npy", np.array(all_labels))
        ```
        ''')
    else:
        all_preds_arr = np.load(preds_path)
        all_labels_arr = np.load(labels_path)

        # Summary metrics
        accuracy = (all_preds_arr == all_labels_arr).mean()
        r = classification_report(all_labels_arr, all_preds_arr, output_dict=True)
        macro_f1 = r['macro avg']['f1-score']
        macro_prec = r['macro avg']['precision']
        macro_rec = r['macro avg']['recall']

        m1, m2, m3, m4 = st.columns(4)
        for col, label, val in zip(
            [m1, m2, m3, m4],
            ['Test Accuracy', 'Macro F1', 'Macro Precision', 'Macro Recall'],
            [f'{accuracy:.1%}', f'{macro_f1:.3f}', f'{macro_prec:.3f}', f'{macro_rec:.3f}']
        ):
            with col:
                st.markdown(f'''
                <div class="stat-card">
                    <div class="value">{val}</div>
                    <div class="label">{label}</div>
                </div>
                ''', unsafe_allow_html=True)

        st.markdown('---')

        # Confusion matrix heatmap
        st.markdown('#### Confusion Matrix')
        from sklearn.metrics import confusion_matrix as sk_cm
        cm = sk_cm(all_labels_arr, all_preds_arr)

        # Build annotation text: number + 🐾 for high-error cells
        diag = cm.diagonal()
        annotations = []
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                val = cm[i, j]
                text = str(val) if val > 0 else ''
                if i != j and val >= 2:
                    text = f'{val} 🐾'
                annotations.append(text)
        annotations_2d = np.array(annotations).reshape(len(CLASS_NAMES), len(CLASS_NAMES)).tolist()

        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=CLASS_NAMES,
            y=CLASS_NAMES,
            colorscale=[
                [0.0, '#0e1117'],
                [0.3, '#0f3460'],
                [0.7, '#e94560'],
                [1.0, '#ffaa00']
            ],
            text=annotations_2d,
            texttemplate='%{text}',
            textfont={'size': 9},
            showscale=True,
            hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
        ))
        fig_cm.update_layout(
            height=750,
            xaxis=dict(tickangle=90, tickfont={'size': 9}, side='bottom'),
            yaxis=dict(tickfont={'size': 9}, autorange='reversed'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'},
            margin=dict(t=20, b=120, l=120, r=20)
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        # Per-class F1 bar chart
        st.markdown('#### Per-class F1 Score')
        class_f1 = [r[str(i)]['f1-score'] for i in range(37)]
        colors_f1 = ['#00cc44' if f >= 0.85 else '#ffaa00' if f >= 0.65 else '#ff4444' for f in class_f1]

        fig_f1 = go.Figure(go.Bar(
            x=CLASS_NAMES,
            y=class_f1,
            marker_color=colors_f1,
            text=[f'{f:.2f}' for f in class_f1],
            textposition='outside',
            textfont={'size': 9, 'color': 'white'}
        ))
        fig_f1.add_hline(y=0.85, line_dash='dash', line_color='#00cc44', opacity=0.5, annotation_text='0.85')
        fig_f1.add_hline(y=0.65, line_dash='dash', line_color='#ffaa00', opacity=0.5, annotation_text='0.65')
        fig_f1.update_layout(
            height=380,
            xaxis=dict(tickangle=45, tickfont={'size': 8}),
            yaxis=dict(range=[0, 1.15]),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'},
            margin=dict(t=20, b=100, l=40, r=20)
        )
        st.plotly_chart(fig_f1, use_container_width=True)
