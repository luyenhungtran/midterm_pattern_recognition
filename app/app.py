"""MNIST Lab — đọc kết quả notebook 01–09 và checkpoint đã huấn luyện.

Chạy từ thư mục gốc: python -m streamlit run app/app.py
"""
from pathlib import Path
from time import perf_counter
from io import BytesIO
from html import escape
from collections import Counter
import base64
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from joblib import load
from PIL import Image, ImageOps, UnidentifiedImageError

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent if APP_DIR.name == 'app' else APP_DIR
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / 'results'
CLASSES = np.arange(10)
COLORS = ['#6366f1', '#0d9488', '#d97706', '#db5d85', '#2563eb', '#7c3aed']
CORE_METHODS = ['Gaussian NB', 'Softmax', 'MLP', 'Stacking AdaBoost']
SPECS = {
    'Gaussian NB': dict(folder='gaussian', model='manual', metrics='metrics_all_splits.csv', artifact='manual_parameters.npz', cm='confusion_manual_', notebook='02',
        tag='Xác suất · NumPy', feature='Ước lượng mean, variance và prior theo lớp.', limit='Giả định độc lập có điều kiện; confidence có thể cao khi sai.'),
    'Softmax': dict(folder='softmax', model='Softmax tự viết', metrics='metrics.csv', artifact='manual_weights.npz', cm='confusion_', notebook='03',
        tag='Tuyến tính · NumPy', feature='Ánh xạ 784 pixel thành 10 logit.', limit='Ranh giới tuyến tính; hạn chế biểu diễn tương tác pixel.'),
    'MLP': dict(folder='mlp', model='manual', metrics='metrics_all_splits.csv', artifact='manual_model.joblib', cm='confusion_manual_', notebook='04',
        tag='Phi tuyến · NumPy', feature='Feedforward và backpropagation thủ công.', limit='Phụ thuộc kiến trúc và quá trình tối ưu.'),
    'Stacking AdaBoost': dict(folder='stacking_adaboost', model='Stacking AdaBoost', metrics='metrics.csv', artifact='classifier.joblib', cm='Stacking_AdaBoost_confusion_', notebook='06',
        tag='OOF · Ensemble', feature='Ghép xác suất base models thành 30 đặc trưng cho AdaBoost.', limit='OOF/refit tốn chi phí; tăng accuracy cần đọc cùng chi phí.'),
    'MLP260 · 50 epoch': dict(folder='cnn_mlp', model='MLP260', metrics='metrics.csv', artifact='MLP260.pt', notebook='08',
        tag='MLP · PyTorch', feature='784–260–10; thí nghiệm so sánh kiến trúc với CNN.', limit='Khác MLP tự viết ở notebook 04.'),
    'CNN · 50 epoch': dict(folder='cnn_mlp', model='CNN_FC128', metrics='metrics.csv', artifact='CNN_FC128.pt', notebook='08',
        tag='CNN · PyTorch', feature='Conv16–Pool–Conv32–Pool–FC128–10.', limit='Checkpoint chọn theo validation CE; không phải snapshot ensemble.'),
}
for model in ['MLP260', 'CNN_FC128']:
    for mode, label in [('none', 'không aug'), ('all', 'có aug')]:
        name = f'{model} · {label} · 15 epoch'
        SPECS[name] = dict(folder='cnn_mlp_augmentation', model=model, mode=mode, metrics='metrics.csv',
            artifact=f'{model}_{mode}.pt', notebook='09', tag='Augmentation · PyTorch',
            feature='So sánh không augmentation và xoay/dịch/nhiễu ở cùng 15 epoch.',
            limit='Cần phân biệt clean accuracy và robustness trên ảnh biến đổi.')
CSS = """
<style>
:root{--ink:#182239;--muted:#6b778e;--primary:#6366f1;--line:#e5eaf3}
.stApp{background:#f6f8fc;color:var(--ink)}
[data-testid="stHeader"]{background:rgba(246,248,252,.94)}
.block-container{max-width:1440px;padding-top:2.2rem;padding-bottom:3rem}
h1,h2,h3{color:var(--ink);letter-spacing:-.035em} h1{font-weight:780!important}
[data-testid="stSidebar"]{background:#121c32;min-width:224px;max-width:250px}
[data-testid="stSidebar"] .stMarkdown p,[data-testid="stSidebar"] label{color:#d7def0}
[data-testid="stSidebar"] [data-testid="stButton"] button{width:100%;justify-content:flex-start;background:transparent;color:#cdd6ee;border:1px solid #2e3a54;border-radius:12px;min-height:48px;font-weight:600}
[data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"]{background:#6366f1;border-color:#777afa;color:white;box-shadow:0 5px 20px #0002}
[data-testid="stSidebar"] hr{border-color:#2a3650}
.brand{font-size:25px;font-weight:800;color:white;margin:8px 0 0;letter-spacing:-1px}.brand span{color:#a5b4fc}
.brand-sub{color:#91a0bf;font-size:11px;letter-spacing:2px;margin-bottom:30px}.nav-label{font-size:10px;color:#8393b4;letter-spacing:2px;margin:16px 0 12px}
.eyebrow{color:#6366f1;font-size:11px;font-weight:750;letter-spacing:2px;margin-bottom:9px}
.hero h1{font-size:39px;line-height:1.15;margin:0 0 9px}.hero p{color:#6b778e;max-width:800px;margin-bottom:22px;font-size:15px}
.pill{display:inline-block;background:#eceefe;color:#5559cc;font-size:11px;padding:5px 10px;border-radius:30px;margin:0 5px 8px 0;font-weight:650}
[data-testid="stVerticalBlockBorderWrapper"]>div{border-radius:18px!important}
[data-testid="stMetric"]{background:white;border:1px solid var(--line);border-radius:16px;padding:15px 19px}
[data-testid="stMetricValue"]{font-weight:750;color:#29344f}
[data-testid="stButton"] button,[data-testid="stDownloadButton"] button{border-radius:10px;min-height:42px}
[data-testid="stButton"] button[kind="primary"]{background:#6366f1;border-color:#6366f1}
.demo-heading{text-align:center}
.st-key-model_preview summary{justify-content:center}
.st-key-model_preview summary [data-testid="stMarkdownContainer"],
.st-key-model_preview [data-testid="stCaptionContainer"]{text-align:center}
.pred-card{background:white;border:1px solid #e5eaf3;border-radius:16px;padding:17px 15px;min-height:172px;margin-bottom:12px}
.pred-name{font-size:12px;font-weight:700;color:#4b5872;min-height:31px;line-height:16px}
.pred-digit{font-size:48px;font-weight:800;line-height:1.2;color:#202b46;letter-spacing:-2px}
.pred-meta{font-size:11px;color:#71809a;margin-top:5px}.confidence-track{height:5px;background:#edf0f7;border-radius:9px;margin:11px 0 5px}.confidence-fill{height:5px;border-radius:9px}
.empty-state{border:1px dashed #cdd5e8;border-radius:18px;background:#f9fbff;text-align:center;padding:35px 20px;color:#8490a7;margin:8px 0 18px}.empty-symbol{font-size:48px;line-height:1.2;color:#c6cee0;font-weight:750;margin-bottom:10px}
.detail-label{font-size:11px;font-weight:700;letter-spacing:1px;color:#75829b;margin-top:10px}
.stTabs [data-baseweb="tab-list"]{gap:16px}.stTabs [data-baseweb="tab"]{font-weight:600}
@media(max-width:850px){.hero h1{font-size:30px}.block-container{padding-top:1.4rem}.pred-card{padding:12px}.pred-digit{font-size:38px}}
.st-key-predict button,
.st-key-predict button p {
    color: #fff !important;
    font-weight: 700 !important;
}
</style>
"""
CANVAS_HTML = """
<div class="pad">
  <div class="canvas-wrap"><canvas id="digit" width="320" height="320" aria-label="Bảng viết chữ số"></canvas></div>
  <div class="tools"><button id="undo" type="button">↶ Hoàn tác</button><button id="clear" type="button">Xóa bảng</button>
    <label>Nét <input id="brush" type="range" min="8" max="30" value="18" aria-label="Độ dày nét" /><span id="size">18</span></label></div>
</div>
"""
CANVAS_CSS = """
.pad{font-family:system-ui,sans-serif;color:#4b5872}.canvas-wrap{max-width:320px;margin:auto}
canvas{width:100%;display:block;aspect-ratio:1;border:1px solid #dce3ef;border-radius:14px;touch-action:none;cursor:crosshair;background:#000;box-sizing:border-box}
.tools{display:flex;gap:8px;align-items:center;justify-content:center;flex-wrap:wrap;margin-top:12px}
button{font-family:inherit;border:1px solid #dce3ef;border-radius:8px;padding:8px 11px;color:#43516a;background:white;cursor:pointer}button:hover{background:#eef0ff}
label{display:flex;align-items:center;gap:5px;font-size:12px}input{width:67px;accent-color:#6366f1}.hint{text-align:center;font-size:11px;color:#8490a7;margin:12px 0 0}
"""
CANVAS_JS = r"""
export default function({parentElement, data, setStateValue}) {
 const canvas=parentElement.querySelector('#digit'), ctx=canvas.getContext('2d');
 const brush=parentElement.querySelector('#brush'), size=parentElement.querySelector('#size');
 const payload=data?.payload ?? null;
 let history=payload?.history ? [...payload.history] : [];
 brush.value=payload?.brush ?? 18; size.textContent=brush.value;
 let drawing=false, ready=true;
 const blank=()=>{ctx.fillStyle='#000000';ctx.fillRect(0,0,320,320)};
 const restore=(png)=>{blank(); if(png){ready=false;const im=new Image();im.onload=()=>{ctx.drawImage(im,0,0);ready=true};im.src=png}};
 if(canvas.dataset.png !== (payload?.png ?? '')){restore(payload?.png);canvas.dataset.png=payload?.png ?? ''}
 if(!canvas.dataset.initialized){restore(payload?.png);canvas.dataset.initialized='1'}
 const send=(png)=>{canvas.dataset.png=png ?? '';setStateValue('payload',{png,history:history.slice(-10),brush:Number(brush.value)})};
 const point=e=>{const r=canvas.getBoundingClientRect();return [(e.clientX-r.left)*320/r.width,(e.clientY-r.top)*320/r.height]};
 brush.oninput=()=>{size.textContent=brush.value};
 canvas.onpointerdown=e=>{if(!ready)return;e.preventDefault();canvas.setPointerCapture(e.pointerId);history.push(canvas.toDataURL('image/png'));drawing=true;
   const [x,y]=point(e);ctx.fillStyle='#ffffff';ctx.beginPath();ctx.arc(x,y,Number(brush.value)/2,0,Math.PI*2);ctx.fill();ctx.beginPath();ctx.moveTo(x,y);
   ctx.strokeStyle='#ffffff';ctx.lineWidth=Number(brush.value);ctx.lineCap='round';ctx.lineJoin='round'};
 canvas.onpointermove=e=>{if(!drawing)return;e.preventDefault();const [x,y]=point(e);ctx.lineTo(x,y);ctx.stroke()};
 const finish=e=>{if(!drawing)return;drawing=false;ctx.closePath();send(canvas.toDataURL('image/png'))};
 canvas.onpointerup=finish;canvas.onpointercancel=finish;
 parentElement.querySelector('#clear').onclick=()=>{drawing=false;history=[];blank();send(null)};
 parentElement.querySelector('#undo').onclick=()=>{if(history.length){const png=history.pop();restore(png);send(png)}};
}
"""


def artifact_path(method):
    spec = SPECS[method]
    return RESULTS / spec['folder'] / spec['artifact']


def fingerprint(paths):
    return tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths if p.exists())


def load_cnn(path):
    import torch
    from torch import nn

    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    name = checkpoint.get('model_name', checkpoint.get('model'))
    if name == 'MLP260':
        model = nn.Sequential(nn.Flatten(), nn.Linear(784, 260), nn.ReLU(), nn.Linear(260, 10))
    else:
        model = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10),
        )
    model.load_state_dict(checkpoint['state_dict'])
    return model.eval()



@st.cache_resource(show_spinner=False)
def load_models(stamp):
    models, issues = {}, {}
    for name, spec in SPECS.items():
        path = artifact_path(name)
        try:
            if path.suffix == '.pt':
                model = load_cnn(path)
            elif path.suffix == '.npz':
                with np.load(path, allow_pickle=False) as values:
                    model = {key: values[key].copy() for key in values.files}
            else:
                model = load(path)
            models[name] = model
        except Exception as exc:
            issues[name] = f"Không nạp được mô hình từ notebook {spec['notebook']}: {exc}"
    return models, issues



@st.cache_data(show_spinner=False)
def read_table(path, modified):
    return pd.read_csv(path)


def table_at(path):
    return read_table(str(path), path.stat().st_mtime_ns) if path.exists() else None


def get_config(method):
    path = RESULTS / SPECS[method]['folder'] / 'config.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def read_results():
    frames, issues = [], []
    for method, spec in SPECS.items():
        directory = RESULTS / spec['folder']
        try:
            frame = table_at(directory / spec['metrics'])
            frame = frame.loc[frame.model == spec['model']].copy()
            if spec['notebook'] == '09':
                frame = frame.loc[frame['mode'] == spec['mode']]
                runs = table_at(directory / 'runs.csv')
                frame = frame.merge(runs, on=['model', 'mode'], how='left')
            frame['split'] = frame['split'].str.strip().str.capitalize()
            frame['method'] = method
            frames.append(frame)
        except Exception as exc:
            issues.append(f"{method}: không đọc được kết quả notebook {spec['notebook']} ({exc})")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), issues



def prepare_digit_image(image, polarity='Tự động', center=True, threshold=20):
    image = ImageOps.exif_transpose(image)
    if image.width * image.height > 16_000_000:
        raise ValueError('Ảnh quá lớn. Hãy chọn ảnh nhỏ hơn 16 megapixel.')
    rgba = image.convert('RGBA')
    a = np.asarray(rgba)
    background = 255
    if polarity == 'Nét trắng':
        background = 0
    elif polarity == 'Tự động' and (a[:, :, 3] < 255).any():
        opaque = a[:, :, :3][a[:, :, 3] > 128]
        background = 0 if len(opaque) and opaque.mean() > 127 else 255
    surface = Image.new('RGBA', rgba.size, (background, background, background, 255))
    gray = Image.alpha_composite(surface, rgba).convert('L')
    pixels = np.asarray(gray)
    border = np.concatenate([pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]])
    invert = polarity == 'Nét đen' or (polarity == 'Tự động' and np.median(border) > 127)
    if invert:
        gray = ImageOps.invert(gray)
    ink = np.asarray(gray)
    ys, xs = np.nonzero(ink > threshold)
    if len(xs) < 3:
        raise ValueError('Chưa thấy nét chữ. Viết một chữ số hoặc chọn lại màu nền/nét.')
    if center:
        cropped = gray.crop((int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1))
        scale = 20 / max(cropped.size)
        cropped = cropped.resize((max(1, round(cropped.width*scale)), max(1, round(cropped.height*scale))), Image.Resampling.LANCZOS)
        processed = Image.new('L', (28, 28), 0)
        processed.paste(cropped, ((28-cropped.width)//2, (28-cropped.height)//2))
        weights = np.asarray(processed, dtype=float)
        yy, xx = np.indices((28, 28))
        dx = round(13.5 - (xx*weights).sum()/weights.sum())
        dy = round(13.5 - (yy*weights).sum()/weights.sum())
        processed = processed.transform((28, 28), Image.Transform.AFFINE, (1, 0, -dx, 0, 1, -dy), fillcolor=0)
    else:
        if gray.size != (28, 28):
            raise ValueError('Chế độ giữ nguyên cần ảnh 28×28. Bật căn nét chữ cho ảnh khác kích thước.')
        processed = gray
    X = np.asarray(processed, dtype=np.float32).reshape(1, 784) / 255.0
    return X, processed


def align_proba(model, X, classes=CLASSES):
    scores = model.predict_proba(X)
    order = [int(np.flatnonzero(np.asarray(model.classes_) == label)[0]) for label in classes]
    return scores[:, order]


def infer_one(name, model, X):
    details, votes = [], []
    if name == 'Gaussian NB':
        logit = np.log(model['priors']) - .5 * (np.log(2*np.pi*model['variances']) + (X.astype(float)-model['means'])**2/model['variances']).sum(axis=1)
        proba = np.exp(logit-logit.max()); proba /= proba.sum()
        classes = np.asarray(model['classes'])
        prediction = int(classes[proba.argmax()])
        proba = proba[[int(np.flatnonzero(classes == k)[0]) for k in CLASSES]]
    elif name == 'Softmax':
        logits = X @ model['W'] + model['b']
        proba = np.exp(logits-logits.max(axis=1, keepdims=True))[0]; proba /= proba.sum()
        prediction = int(proba.argmax())
    elif name == 'MLP':
        proba = align_proba(model, X)[0]
        prediction = int(model.predict(X)[0])
    elif SPECS[name]['artifact'].endswith('.pt'):
        import torch

        image = torch.as_tensor(X, dtype=torch.float32).reshape(-1, 1, 28, 28)
        with torch.inference_mode():
            proba = model(image).softmax(dim=1)[0].numpy()
        prediction = int(proba.argmax())
    else:
        classes = np.asarray(model['classes'])
        features = []
        for base_name in model['model_order']:
            base = model['base_models'][base_name]
            scores = align_proba(base, X, classes)[0]
            predicted = int(classes[scores.argmax()])
            details.append(dict(model=base_name, prediction=predicted,
                confidence=float(scores[np.flatnonzero(classes == predicted)[0]])))
            features.append(scores.reshape(1, -1))
        probabilities = np.concatenate(features, axis=0)
        voting_scores = [('Soft voting', probabilities.mean(axis=0))]
        weights = model.get('voting_weights')
        if weights is not None:
            voting_scores.append(('Weighted soft voting', np.average(probabilities, axis=0, weights=weights)))
        for label, scores in voting_scores:
            best = int(scores.argmax())
            votes.append(dict(model=label, prediction=int(classes[best]), confidence=float(scores[best])))
        Z = np.concatenate(features, axis=1)
        proba = align_proba(model['meta_model'], Z)[0]
        prediction = int(proba.argmax())
    if proba.shape != (10,) or not np.isfinite(proba).all() or (proba < 0).any() or not np.isclose(proba.sum(), 1, atol=1e-5):
        raise ValueError('Mô hình trả điểm xác suất không hợp lệ.')
    others = np.argsort(-proba, kind='stable')
    second = int(next(k for k in others if k != prediction))
    return dict(model=name, prediction=prediction, confidence=float(proba[prediction]), second=second,
                margin=float(proba[prediction]-proba[second]), scores=proba.tolist(), base_details=details, voting_details=votes)


def style_figure(fig, height=340):
    fig.update_layout(template='plotly_white', height=height, margin=dict(l=15, r=15, t=40, b=20),
        font=dict(family='Arial, sans-serif', color='#4b5872'), paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)', legend=dict(orientation='h', y=1.13, x=0))
    return fig


def plot(fig, key=None):
    st.plotly_chart(style_figure(fig), width='stretch', key=key, config={'displayModeBar': False})


def score_frame(frame):
    cols = {'method': 'Mô hình', 'model': 'Mô hình', 'split': 'Tập', 'accuracy': 'Accuracy (%)',
            'precision_macro': 'Precision macro (%)', 'recall_macro': 'Recall macro (%)', 'f1_macro': 'F1 macro (%)',
            'fit_seconds': 'Fit (s)', 'predict_seconds': 'Predict (s)', 'errors': 'Số ảnh sai'}
    columns = [c for c in frame.columns if c in cols and (c != 'model' or 'method' not in frame)]
    work = frame[columns].copy()
    for col in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']:
        if col in work:
            work[col] = (work[col]*100).round(2)
    return work.rename(columns=cols).round(4)


def csv_download(frame, name, label='Tải bảng CSV'):
    st.download_button(label, frame.to_csv(index=False).encode('utf-8-sig'), file_name=name, mime='text/csv', key='download_'+name)


def hero(eyebrow, title, description):
    st.markdown(f'<div class="hero"><div class="eyebrow">{escape(eyebrow)}</div><h1>{escape(title)}</h1><p>{escape(description)}</p></div>', unsafe_allow_html=True)


def render_predictions(records, available):
    by_name = {r['model']: r for r in records}
    names = list(by_name) or CORE_METHODS
    for start in range(0, len(names), 3):
        for col, name in zip(st.columns(3), names[start:start+3]):
            color = COLORS[list(SPECS).index(name) % len(COLORS)]
            r = by_name.get(name)
            digit = str(r['prediction']) if r else '—'
            pct = r['confidence'] * 100 if r else 0
            text = f"Confidence {pct:.1f}%" if r else ('Chờ nhận dạng' if name in available else 'Chưa có mô hình')
            with col:
                st.markdown(f'<div class="pred-card" style="border-top:3px solid {color}"><div class="pred-name">{escape(name)}</div><div class="pred-digit">{digit}</div><div class="confidence-track"><div class="confidence-fill" style="background:{color};width:{pct:.2f}%"></div></div><div class="pred-meta">{text}</div></div>', unsafe_allow_html=True)



def render_demo():
    st.markdown('<div class="hero"><h1 style="text-align: center; color: #6366f1;">INTERACTION DEMO</h1></div>', unsafe_allow_html=True)

    paths = [artifact_path(m) for m in SPECS]
    model_stamp = fingerprint(paths)
    models, issues = load_models(model_stamp)
    left, right = st.columns([.95, 1.65], gap='large')
    X = processed = None
    current_signature = None
    with left, st.container(border=True):
        
        with st.container(
            horizontal=True,
            horizontal_alignment='center',
            vertical_alignment='center',
            gap='medium',
        ):
            st.markdown(
                '<span style="font-weight: 780; font-size: 20px;">Chữ số</span>',
                unsafe_allow_html=True,
                width='content',
            )
            mode = st.radio(
                'Nguồn ảnh',
                ['Bảng viết', 'Tải ảnh'],
                horizontal=True,
                label_visibility='collapsed',
                key='input_mode',
                width='content',
            )
        original, polarity, center = None, 'Nét trắng', True
        if mode == 'Bảng viết':
            state = st.session_state.get('white_ink_canvas', {})
            payload = state.get('payload', st.session_state.get('saved_white_ink_canvas'))
            result = digit_canvas(data={'payload': payload}, default={'payload': payload}, key='white_ink_canvas', on_payload_change=lambda: None)
            st.session_state['saved_white_ink_canvas'] = result.payload
            if result.payload and result.payload.get('png'):
                raw = base64.b64decode(result.payload['png'].split(',', 1)[1])
                original = Image.open(BytesIO(raw)).convert('RGB')
        else:
            uploaded = st.file_uploader('Ảnh một chữ số', type=['png', 'jpg', 'jpeg', 'webp'], key='digit_upload')
            polarity = st.selectbox('Màu nét chữ', ['Tự động', 'Nét đen', 'Nét trắng'])
            center = st.checkbox('Cắt và căn nét chữ về dạng MNIST', value=True)
            if not center:
                st.caption('Giữ đúng pixel ảnh 28×28 đã ở dạng MNIST; không resize/căn lại.')
            if uploaded is not None:
                try:
                    original = Image.open(uploaded)
                    st.image(ImageOps.exif_transpose(original), width=220)
                except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
                    st.error(f'Không đọc được ảnh: {exc}')
        if original is not None:
            try:
                X, processed = prepare_digit_image(original, polarity, center)
                current_signature = hashlib.sha256(processed.tobytes()).hexdigest()
            except ValueError as exc:
                st.caption(str(exc))
        with st.expander('Xem ảnh mô hình nhận', expanded=True, key='model_preview'):
            if processed is not None:
                with st.container(horizontal=True, horizontal_alignment='center'):
                    st.image(processed.resize((168, 168), Image.Resampling.NEAREST), width=168)
                buffer = BytesIO(); processed.save(buffer, format='PNG')
                with st.container(horizontal=True, horizontal_alignment='center'):
                    st.download_button('Tải ảnh 28×28', buffer.getvalue(), 'digit_28x28.png', 'image/png', key='download_digit')
            else:
                st.caption('Ảnh sau tiền xử lý sẽ xuất hiện tại đây.')
        chosen = st.multiselect('Mô hình dự đoán', list(models), default=[name for name in CORE_METHODS + ['CNN · 50 epoch'] if name in models], key='demo_models')
        predict = st.button('Nhận dạng chữ số', type='primary', width='stretch', disabled=X is None or not chosen, icon=':material/auto_awesome:', key='predict')
        if predict:
            records, failures = [], []
            with st.spinner('Các mô hình đang xử lý…'):
                for name in chosen:
                    model = models[name]
                    try:
                        start = perf_counter(); record = infer_one(name, model, X)
                        record['inference_ms'] = (perf_counter()-start)*1000
                        records.append(record)
                    except Exception as exc:
                        failures.append(f'{name}: {exc}')
            st.session_state['prediction'] = dict(signature=current_signature, stamp=model_stamp, records=records, failures=failures, selected=chosen)
        with st.expander('Tiền xử lý và mô hình'):
            st.write('Ảnh tự viết được crop theo nét, giữ tỷ lệ cạnh dài 20 pixel, đặt trên nền 28×28 và căn theo trọng tâm nét. Không dùng mean/std vì các notebook hiện tại chỉ chia 255.')
            st.write('Viết một chữ số, tránh nhiều ký tự và nền ảnh phức tạp. Có thể chọn lại màu nét nếu tự động nhận sai nền.')
            for name, message in issues.items():
                st.caption(f'{name} — {message}')
    with right:
        st.markdown('<h3 class="demo-heading">Quyết định của các mô hình</h3>', unsafe_allow_html=True)
        result = st.session_state.get('prediction')
        is_current = result is not None and current_signature is not None and result['signature'] == current_signature and result['stamp'] == model_stamp and result.get('selected') == chosen
        records = result['records'] if is_current else []
        render_predictions(records, models)
        if not records:
            st.markdown('<div class="empty-state"><div class="empty-symbol">0 · 1 · 2 · … · 9</div>Nhập một chữ số rồi bấm <b>Nhận dạng chữ số</b>.<br>Kết quả sẽ được so sánh trên cùng một ảnh.</div>', unsafe_allow_html=True)
        else:
            votes = Counter(r['prediction'] for r in records)
            digit, count = votes.most_common(1)[0]
            if count == len(records):
                st.success(f'{count}/{len(records)} mô hình đã chạy cùng dự đoán chữ số {digit}.')
            else:
                st.info('Các mô hình chưa đồng thuận. Xem phân bố 10 lớp bên dưới để so sánh.')
            if result['failures']:
                for error in result['failures']: st.warning(error)
            frame = pd.DataFrame([{'Mô hình': r['model'], 'Dự đoán': r['prediction'], 'Confidence (%)': r['confidence']*100,
                'Lựa chọn 2': r['second'], 'Chênh lệch (điểm %)': r['margin']*100, 'Suy luận (ms)': r['inference_ms']} for r in records])
            st.dataframe(frame.round(3), hide_index=True, width='stretch')
            selected = st.selectbox('Phân bố xác suất 10 lớp', list(by_name := {r['model']: r for r in records}), key='demo_score_model')
            selected_record = by_name[selected]
            scores = selected_record['scores']
            fig = go.Figure(go.Bar(x=list(range(10)), y=np.array(scores)*100,
                marker_color=['#6366f1' if k == selected_record['prediction'] else '#dfe3f6' for k in range(10)],
                text=[f'{v:.1%}' for v in scores], textposition='outside'))
            fig.update_layout(xaxis=dict(title='Chữ số', dtick=1), yaxis=dict(title='Xác suất mô hình (%)', range=[0, 112]))
            plot(fig, 'demo_scores')
            if selected_record['base_details']:
                with st.expander('Quyết định bên trong ensemble'):
                    base_table = pd.DataFrame(selected_record['base_details'])
                    base_table['confidence'] *= 100
                    st.dataframe(base_table.rename(columns={'model':'Base model','prediction':'Dự đoán','confidence':'Confidence (%)'}).round(2), hide_index=True)
                    votes = pd.DataFrame(selected_record.get('voting_details', []))
                    if not votes.empty:
                        votes['confidence'] *= 100
                        st.write('Soft voting từ cùng các xác suất mô hình nền')
                        st.dataframe(votes.rename(columns={'model':'Phương án','prediction':'Dự đoán','confidence':'Confidence (%)'}).round(2), hide_index=True)
                    st.caption('Base MLP trong stacking có thể khác MLP tự viết đang hiển thị ở trang Demo. Base learners và meta-model đều chọn argmax xác suất như notebook 06.')
            payload = dict(input_sha256=current_signature, input_shape=[28,28], preprocessing=dict(polarity=polarity, centered=center, scale='1/255'), records=records)
            st.download_button('Tải kết quả ảnh này · JSON', json.dumps(payload, ensure_ascii=False, indent=2), 'prediction.json', 'application/json')
        st.caption('Confidence là điểm xác suất mô hình gán cho chữ số dự đoán. Các mô hình có thể có mức hiệu chuẩn khác nhau. Không dùng confidence để chọn mô hình thắng: Gaussian NB có thể rất tự tin khi sai, còn xác suất SAMME có thể khá phẳng.')
        st.caption('Thời gian demo tính cả dự đoán và xác suất cho một ảnh, không gồm nạp model/tiền xử lý.')


def display_confusion(method):
    spec = SPECS[method]
    directory = RESULTS / spec['folder']
    c1, c2 = st.columns(2)
    splits = ['Test', 'Validation', 'Train'] if method in CORE_METHODS else ['Test']
    split = c1.selectbox('Tập dữ liệu', splits, key='cm_split')
    normalized = c2.toggle('Chuẩn hóa theo nhãn thật', value=True, key='cm_norm')
    if method in CORE_METHODS:
        raw = table_at(directory / f"{spec['cm']}{split.lower()}.csv")
        cm = raw.iloc[:, 1:].to_numpy(dtype=float)
    else:
        if spec['notebook'] == '08':
            predictions = table_at(directory / 'test_predictions.csv')
            truth, pred = predictions.y_true.to_numpy(), predictions[spec['model']].to_numpy()
        else:
            with np.load(directory / 'test_predictions.npz', allow_pickle=False) as predictions:
                truth, pred = predictions['y_true'], predictions[f"{spec['model']}_{spec['mode']}"]
        cm = np.bincount(truth.astype(int) * 10 + pred.astype(int), minlength=100).reshape(10, 10).astype(float)
    row_sum = cm.sum(axis=1, keepdims=True)
    shown = cm / row_sum * 100 if normalized else cm
    text = np.array([[f'{v:.1f}%' if normalized else f'{v:.0f}' for v in row] for row in shown])
    fig = go.Figure(go.Heatmap(z=shown, x=list(range(10)), y=list(range(10)), colorscale='Blues', zmin=0,
        zmax=100 if normalized else None, text=text, texttemplate='%{text}', colorbar=dict(title='%' if normalized else 'Ảnh')))
    fig.update_layout(xaxis=dict(title='Nhãn dự đoán', dtick=1), yaxis=dict(title='Nhãn thật', autorange='reversed', dtick=1))
    st.plotly_chart(style_figure(fig, 460), width='stretch', key='confusion_heatmap')
    if split == 'Test':
        recall = np.diag(cm) / row_sum[:, 0]
        st.caption(f'Lớp có recall thấp nhất: {int(recall.argmin())} ({recall.min():.1%}). Không dùng test để tuning lại.')



def show_survey(method):
    spec, cfg = SPECS[method], get_config(method)
    directory = RESULTS / spec['folder']
    if method == 'Gaussian NB':
        data = table_at(directory / 'smoothing_search.csv')
        data = data.loc[data.model == 'manual']
        melted = data.melt(id_vars='vs', value_vars=['accuracy', 'f1_macro'], var_name='Độ đo', value_name='score')
        fig = px.line(melted, x='vs', y='score', color='Độ đo', markers=True, log_x=True)
        fig.update_layout(xaxis_title='var_smoothing', yaxis_title='Điểm validation')
        plot(fig, 'nb_survey')
        st.caption(f"Cấu hình cuối: var_smoothing={cfg['final_manual_var_smoothing']}. Sklearn cuối dùng {cfg['final_sklearn_var_smoothing']}; hai cấu hình không đồng nhất.")
    elif method == 'Softmax':
        history = table_at(directory / 'history.csv')
        fig = go.Figure()
        for column, label in [('train_loss', 'Train CE'), ('val_loss', 'Validation CE')]:
            fig.add_scatter(x=history.epoch, y=history[column], name=label)
        fig.add_vline(x=cfg['best_epoch'], line_dash='dash')
        fig.update_layout(xaxis_title='Epoch', yaxis_title='Cross-entropy')
        plot(fig, 'softmax_loss')
        st.image(str(directory / 'weight_maps.png'), caption='Trọng số theo chữ số: đóng góp vào logit.')
        for filename in ['learning_rate.csv', 'l2.csv']:
            with st.expander(f'Khảo sát {filename[:-4]}'):
                st.dataframe(table_at(directory / filename).round(5), hide_index=True)
    elif method == 'MLP':
        history = table_at(directory / 'final_loss_manual.csv')
        fig = px.line(history, x='epoch', y='loss')
        fig.update_layout(xaxis_title='Epoch', yaxis_title='Train CE + L2')
        plot(fig, 'mlp_loss')
        st.caption('Kiến trúc cuối: ' + ' → '.join(map(str, [784, *cfg['final_manual']['hidden_units'], 10])))
        filename = st.selectbox('Khảo sát', ['learning_rate_search.csv', 'architecture_search.csv', 'epochs_search.csv', 'l2_search.csv', 'batch_size_search.csv'])
        st.dataframe(table_at(directory / filename).round(5), hide_index=True, width='stretch')
    elif spec['notebook'] in ['08', '09']:
        filename = f"history_{spec['model']}.csv" if spec['notebook'] == '08' else f"history_{spec['model']}_{spec['mode']}.csv"
        st.dataframe(table_at(directory / filename).round(5), hide_index=True, width='stretch')
        st.image(str(directory / 'learning_curves.png'), caption='Các đường học trong cùng thí nghiệm.')
        if spec['notebook'] == '09':
            robust = table_at(directory / 'robustness.csv')
            robust = robust.loc[robust.model == spec['model']]
            for kind, label in [('shift', 'Dịch chuyển (pixel)'), ('rotation', 'Xoay (độ)')]:
                fig = px.line(robust.loc[robust.kind == kind], x='level', y='accuracy', color='mode', markers=True)
                fig.update_layout(xaxis_title=label, yaxis_title='Test accuracy')
                plot(fig, 'robust_' + kind)
    else:
        data = table_at(directory / 'hyperparameters.csv')
        parameter = st.selectbox('Siêu tham số meta-learner', ['learning_rate', 'max_depth', 'n_estimators'], key='meta_parameter')
        fixed = {'learning_rate': (data.max_depth == 2) & (data.n_estimators == 100),
                 'max_depth': (data.learning_rate == .5) & (data.n_estimators == 100),
                 'n_estimators': (data.learning_rate == .5) & (data.max_depth == 2)}
        work = data.loc[fixed[parameter]].sort_values(parameter)
        melted = work.melt(id_vars=parameter, value_vars=['accuracy', 'f1_macro'], var_name='Độ đo', value_name='score')
        fig = px.line(melted, x=parameter, y='score', color='Độ đo', markers=True)
        fig.update_yaxes(title='Điểm validation')
        plot(fig, 'meta_survey')
        st.caption('Khảo sát từng yếu tố quanh baseline, không phải toàn bộ tổ hợp tham số.')
    with st.expander('Cấu hình đã lưu'):
        st.json(cfg, expanded=False)



def render_results():
    hero('EXPERIMENT RESULTS', 'Đằng sau một dự đoán.', 'Khám phá chất lượng, chi phí và những đặc điểm của các mô hình trên MNIST.')
    data, issues = read_results()
    if issues:
        with st.expander(f'Tình trạng dữ liệu · {len(issues)} mục cần bổ sung', expanded=data.empty):
            for issue in issues: st.info(issue)
            st.caption('Chạy notebook 02–04 và 05 → 06. Notebook 07 tổng hợp; notebook 08 so sánh CNN–MLP; notebook 09 khảo sát augmentation.')
    if data.empty:
        st.markdown('<div class="empty-state"><div class="empty-symbol">∑</div>Chưa có kết quả thực nghiệm.<br>Ứng dụng chỉ hiển thị số liệu đã chạy, không tạo điểm minh họa.</div>', unsafe_allow_html=True)
        return
    test = data[data.split == 'Test'].copy()
    if 'errors' not in test.columns:
        test['errors'] = np.nan

    missing = test['errors'].isna()
    test.loc[missing, 'errors'] = (
        (1 - test.loc[missing, 'accuracy']) * 10_000
    ).round().astype(int)
    for col, method in zip(st.columns(4), CORE_METHODS):
        row = test[test.method == method]
        with col:
            st.metric(method, f'{row.iloc[0].accuracy:.2%}' if len(row) else '—', help='Accuracy trên MNIST test; không phải confidence của ảnh demo.')
    overview, detail, ensemble, extensions, eda = st.tabs(['Bốn phương pháp', 'Theo mô hình', 'Ensemble', 'CNN & augmentation', 'Dữ liệu & EDA'])
    with overview:
        st.subheader('Chất lượng theo tập dữ liệu')
        l, r = st.columns([1,2])
        options = {'Accuracy':'accuracy','Precision macro':'precision_macro','Recall macro':'recall_macro','F1 macro':'f1_macro'}
        metric_label = l.selectbox('Độ đo', list(options), key='metric')
        splits = r.multiselect('Tập dữ liệu', ['Train','Validation','Test'], default=['Train','Validation','Test'], key='splits')
        chart = data[data.split.isin(splits) & data.method.isin(CORE_METHODS)].copy()
        chart['score'] = chart[options[metric_label]]*100
        fig = px.bar(chart, x='method', y='score', color='split', barmode='group',
            category_orders={'method':list(SPECS),'split':['Train','Validation','Test']}, color_discrete_sequence=COLORS)
        fig.update_layout(xaxis_title='', yaxis=dict(title=f'{metric_label} (%)', range=[0,105]))
        plot(fig, 'overview_quality')
        st.dataframe(score_frame(test.loc[test.method.isin(CORE_METHODS)].drop(columns='model')), hide_index=True, width='stretch')
        csv_download(data, 'metrics_all_splits.csv')
        st.caption('Train của stacking là in-sample sau refit; validation đã dùng chọn cấu hình. Không dùng bảng test để chọn lại mô hình.')
        st.subheader('Chi phí tính toán')
        if {'fit_seconds','predict_seconds'}.issubset(test.columns):
            costs = test.loc[test.method.isin(CORE_METHODS), ['method','fit_seconds','predict_seconds']].copy()
            costs = costs.melt(id_vars='method', var_name='Phép đo', value_name='Giây')
            costs = costs[costs['Giây'] > 0]
            fig = px.bar(costs, x='method', y='Giây', color='Phép đo', barmode='group', log_y=True, color_discrete_sequence=COLORS)
            fig.update_layout(xaxis_title=''); plot(fig, 'overview_timing')
            st.caption('Fit là cấu hình cuối. Stacking gồm OOF, refit base models và fit meta; không chỉ AdaBoost. Khi dùng cache, kiểm tra config notebook 06. Đây là các phép đo một lần, không kiểm soát hoàn toàn phần cứng/tải hệ thống.')
        comparison_dir = RESULTS / 'comparison'
        manifest = json.loads((comparison_dir / 'comparison_config.json').read_text(encoding='utf-8'))
        exported = [comparison_dir / name for name in manifest['output_tables'] if name != 'search_timing.csv']
        if exported:
            with st.expander('Bảng tổng hợp từ notebook 07'):
                selected_csv = st.selectbox('Bảng', sorted(p.name for p in exported), key='comparison_export')
                selected_path = comparison_dir / selected_csv
                st.dataframe(table_at(selected_path), hide_index=True, width='stretch')
                st.download_button('Tải file đã chọn', selected_path.read_bytes(), selected_csv, 'text/csv')
                st.caption('Chạy lại notebook 07 sau khi cập nhật các thí nghiệm để bảng tổng hợp phản ánh kết quả mới.')
        for filename, caption in [('scores_by_split.png', 'So sánh độ đo theo tập'), ('recall_by_digit.png', 'Recall theo chữ số')]:
            path = comparison_dir / filename
            if path.exists():
                with st.expander(caption):
                    st.image(str(path), width='stretch')
    with detail:
        method = st.selectbox('Mô hình cần khám phá', list(SPECS), key='detail_model')
        spec = SPECS[method]
        st.markdown(f'<span class="pill">{escape(spec["tag"])}</span>', unsafe_allow_html=True)
        st.write(spec['feature'])
        st.caption(spec['limit'])
        cfg = get_config(method)
        facts = {
            'Gaussian NB': [('var_smoothing', cfg.get('final_manual_var_smoothing', '—')), ('Đặc trưng', 784), ('Cách học', 'Thống kê lớp')],
            'Softmax': [('Learning rate', cfg.get('learning_rate', '—')), ('L2', cfg.get('l2', '—')), ('Checkpoint', cfg.get('best_epoch', '—'))],
            'MLP': [('Epoch', cfg.get('final_manual', {}).get('epochs', '—')), ('Batch size', cfg.get('final_manual', {}).get('batch_size', '—')), ('Learning rate', cfg.get('final_manual', {}).get('learning_rate', '—'))],
            'Stacking AdaBoost': [('OOF folds', cfg.get('folds', '—')), ('Số cây meta', cfg.get('actual_estimators', cfg.get('n_estimators', '—'))), ('Độ sâu cây', cfg.get('max_depth', '—'))],
        }
        model_rows = data.loc[(data.method == method) & (data.split == 'Test')]
        best_epoch = model_rows.iloc[0].get('best_epoch', '—') if len(model_rows) else '—'
        cnn_facts = [('Epoch tối đa', cfg.get('epochs', '—')), ('Checkpoint epoch', best_epoch),
                     ('Thí nghiệm', '08 · Kiến trúc' if spec['notebook'] == '08' else '09 · Augmentation')]
        for col, (label, value) in zip(st.columns(3), facts.get(method, cnn_facts)):
            col.metric(label, str(value))
        st.subheader('Khảo sát và đặc trưng')
        show_survey(method)
        st.subheader('Confusion matrix')
        display_confusion(method)
    with ensemble:
        cfg = get_config('Stacking AdaBoost')
        order = cfg.get('model_order', [])
        st.subheader('Từ xác suất đến quyết định cuối')
        if order:
            st.markdown(' '.join(f'<span class="pill">{escape(str(name))}</span>' for name in order), unsafe_allow_html=True)
            c1,c2,c3 = st.columns(3)
            c1.metric('Base models', len(order)); c2.metric('OOF folds', cfg.get('folds', '—')); c3.metric('Đặc trưng meta', len(order)*10)
        full = table_at(RESULTS / 'stacking_adaboost' / 'metrics.csv')
        if full is not None:
            selected = full[full.split == 'Test'].copy()
            st.dataframe(score_frame(selected), hide_index=True, width='stretch')
            if order and {'Soft voting', 'Stacking AdaBoost'}.issubset(set(selected.model)):
                by_name = selected.set_index('model')
                validation = full.loc[(full.split == 'Validation') & full.model.isin(order)].sort_values(
                    ['accuracy', 'f1_macro', 'model'], ascending=[False, False, True])
                baseline = cfg.get('best_single_model')
                if baseline is None and not validation.empty:
                    baseline = validation.iloc[0]['model']
                references = [name for name in [baseline, 'Soft voting', 'Weighted soft voting'] if name in by_name.index]
                stacking = by_name.loc['Stacking AdaBoost', 'accuracy']
                if references:
                    for col, name in zip(st.columns(len(references)), references):
                        col.metric(f'Stacking − {name}', f'{(stacking - by_name.loc[name, "accuracy"]) * 100:+.2f} điểm %')
                if baseline:
                    st.caption(f'Model đơn tham chiếu chọn trên validation: {baseline}.')

            errors = table_at(RESULTS / 'stacking_adaboost' / 'error_analysis_test.csv')
            if errors is not None:
                st.write('Số ảnh stacking sửa đúng và làm sai thêm trên test')
                st.dataframe(errors, hide_index=True, width='stretch')
            weights = cfg.get('voting_weights')
            weight_order = cfg.get('voting_weight_model_order', order)
            if weights is not None and len(weights) == len(weight_order):
                st.write('Trọng số soft voting đã chọn trên validation')
                st.dataframe(pd.DataFrame({'Model': weight_order, 'Trọng số': weights}), hide_index=True)
        else:
            st.info('Chạy xong notebook 06 để xem so sánh base models / voting / stacking.')
        selection_path = RESULTS / 'select_base_models'
        with st.expander('Vì sao chọn bộ ba này?'):
            selection = table_at(selection_path / 'single_pair_triple_comparison.csv')
            if selection is not None:
                st.dataframe(selection.round(5), hide_index=True, width='stretch')
            triples = table_at(selection_path / 'triples.csv')
            if triples is not None:
                st.dataframe(triples.head(10).round(5), hide_index=True, width='stretch')
                st.caption('Chọn theo soft voting accuracy validation, F1 macro phá hòa. Overlap là số ảnh cùng sai / số lỗi ít nhất, không phải tỷ lệ nhãn dự đoán trùng nhau.')
            else:
                st.caption('Chạy notebook 05 để có bảng lựa chọn bộ ba.')
            overlap = selection_path / 'error_overlap.png'
            if overlap.exists(): st.image(str(overlap), caption='Độ trùng lỗi giữa các cặp mô hình')
        st.caption('OOF dùng để fit meta-model. Chọn bộ ba bằng voting không bảo đảm bộ ba đó tốt nhất cho stacking; điểm cao hơn cần được đọc cùng chi phí và giới hạn một split/seed.')
    with extensions:
        st.subheader('CNN và MLP có số tham số gần nhau · notebook 08')
        part08 = test.loc[test.method.isin(['MLP260 · 50 epoch', 'CNN · 50 epoch'])]
        st.dataframe(score_frame(part08), hide_index=True, width='stretch')
        st.caption('50 epoch, checkpoint chọn theo validation CE. CNN có 206.922 tham số, MLP260 có 206.710 tham số; cùng số tham số không đồng nghĩa cùng chi phí tính toán.')
        st.subheader('Augmentation · notebook 09')
        part09 = test.loc[test.method.isin([name for name, spec in SPECS.items() if spec['notebook'] == '09'])]
        st.dataframe(score_frame(part09), hide_index=True, width='stretch')
        st.caption('Cùng 15 epoch: none là không augmentation, all là xoay/dịch/nhiễu. Không gộp với thí nghiệm 50 epoch.')
        robust = table_at(RESULTS / 'cnn_mlp_augmentation' / 'robustness.csv')
        for kind, label in [('shift', 'Dịch chuyển (pixel)'), ('rotation', 'Xoay (độ)')]:
            fig = px.line(robust.loc[robust.kind == kind], x='level', y='accuracy', color='model', line_dash='mode', markers=True)
            fig.update_layout(xaxis_title=label, yaxis_title='Test accuracy')
            plot(fig, 'extension_' + kind)
        st.caption('Augmentation có thể giảm điểm clean test nhưng tăng độ bền trên ảnh biến đổi; chỉ diễn giải trong phạm vi đã khảo sát.')
    with eda:
        st.subheader('Dữ liệu đầu vào')
        st.write('MNIST gồm 70.000 ảnh xám 28×28, nhãn 0–9. Đồ án dùng 50.000 train, 10.000 validation và 10.000 test; pixel chia 255, không chuẩn hóa mean/std trong các notebook hiện tại.')
        figures = [('class_distribution_train.png','Phân phối nhãn train'), ('pixel_intensity_std.png','Cường độ pixel'),
                   ('class_mean_distances.png','Khoảng cách ảnh trung bình'), ('class_mean_images.png','Ảnh trung bình theo lớp')]
        columns = st.columns(2)
        for i,(filename, caption) in enumerate(figures):
            path = RESULTS / 'eda' / filename
            with columns[i%2]:
                st.image(str(path), caption=caption, width='stretch')


def main():
    global digit_canvas
    st.set_page_config(page_title='MNIST Lab · Demo & Result', page_icon='✍️', layout='wide', initial_sidebar_state='expanded')
    st.markdown(CSS, unsafe_allow_html=True)
    digit_canvas = st.components.v2.component('mnist_digit_canvas', html=CANVAS_HTML, css=CANVAS_CSS, js=CANVAS_JS)
    st.session_state.setdefault('page', 'Demo')
    with st.sidebar:
        st.markdown('<div class="brand"><span>mnist</span></div><div class="brand-sub">PATTERN RECOGNITION</div>', unsafe_allow_html=True)
        for page, icon in [('Demo', ':material/draw:'), ('Result', ':material/bar_chart:')]:
            if st.button(page, icon=icon, type='primary' if st.session_state.page == page else 'secondary', width='stretch', key='nav_'+page):
                st.session_state.page = page
                st.rerun()
        st.divider()
        count = sum(artifact_path(name).exists() for name in SPECS)
        st.markdown(
            f'<p style="text-align: center;"><strong>{count}/{len(SPECS)} phương án có file</strong></p>',
            unsafe_allow_html=True,
        )
        if st.button('Tải lại dữ liệu', icon=':material/refresh:', width='stretch', key='refresh'):
            load_models.clear(); read_table.clear()
            st.session_state.pop('prediction', None)
            st.rerun()
        with st.expander('Các file cần có'):
            for name, spec in SPECS.items(): st.caption(f"{name}: results/{spec['folder']}/{spec['artifact']}")
            st.caption('MLP cần src/models/mlp.py. CNN cần PyTorch. Dùng phiên bản thư viện phù hợp với môi trường huấn luyện.')
    if st.session_state.page == 'Demo': render_demo()
    else: render_results()
    st.divider()


if __name__ == '__main__':
    main()
