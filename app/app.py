import streamlit as st, traceback

# まず必ず作っておく（NameError回避）
_SEM_OK: bool = False
_SEM_ERR = None  # (err_name, err_msg, traceback_text)

# ---- semopy 互換インポート（旧/新API両対応）----
try:
    from semopy import ModelMeans, Optimizer
    from semopy.inspector import inspect
    try:
        from semopy.report import gather_statistics          # 旧API
    except ImportError:
        from semopy.inspector import inspect as gather_statistics  # 新APIをエイリアス
    _SEM_OK = True
except Exception as e:
    _SEM_OK = False
    _SEM_ERR = (type(e).__name__, str(e), "".join(traceback.format_exc()))
# ---------------------------------------------------------


import importlib.util, sys, subprocess
if importlib.util.find_spec("semopy") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "semopy==2.3.11"])
import streamlit as st
import pandas as pd
from io import StringIO, BytesIO
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score  # 決定係数計算用
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from statsmodels.miscmodels.ordinal_model import OrderedModel
import numpy as np
from PIL import Image
import textwrap
import matplotlib.pyplot as plt
import io
import base64
from semopy import Model



logo = Image.open("app/LOGO.png")
st.sidebar.image(logo, use_column_width=True)

st.markdown("""
<style>

/* =====================================
      カード全体のデザイン（背景＝緑、文字＝白）
===================================== */
.card {
    background-color: #2e7d32;       /* MORIPIEグリーン */
    padding: 2rem;
    margin-top: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    line-height: 1.6;
    color: #ffffff !important;

    /* 🔥 カードをほぼ全幅に広げる */
    width: 100%;
    max-width: 1800px; /* 1500〜2000が黄金比 */
    margin-left: auto;
    margin-right: auto;
}

/* カード内すべての文字を白 */
.card, .card * {
    color: #ffffff !important;
}

/* h2 タイトル */
.card h2 {
    margin-top: 0;
    font-weight: 600;
    color: #ffffff !important;
}

/* h3 見出し */
.card h3 {
    margin-top: 1.4rem;
    margin-bottom: 0.5rem;
    font-weight: 500;
    color: #e8f5e9 !important;
}

/* リスト */
.card ul {
    padding-left: 1.4rem;
}
.card li {
    margin-bottom: 0.4rem;
    color: #ffffff !important;
}

/* 太字 */
.card b {
    color: #ffffff !important;
}

/* h4 も白に */
.card h4, .card h4 * {
    color: #ffffff !important;
}

/* code（例: コードブロック） */
.card code {
    background-color: #ffffff22 !important;
    color: #000000 !important;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.9rem;
}

/* テーブル */
.card table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 1rem;
    background-color: rgba(255,255,255,0.1);
}
.card th, .card td {
    border: 1px solid rgba(255,255,255,0.3);
    padding: 0.5rem 0.8rem;
    color: #ffffff !important;
}
.card th {
    font-weight: bold;
    background-color: rgba(255,255,255,0.2);
}

/* 🔥 ページ自体の最大幅（重要） */
main .block-container {
    max-width: 2500px;
    padding-left: 2rem;
    padding-right: 2rem;
}

</style>

""", unsafe_allow_html=True)



st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@300;400;500;700&display=swap');

/* 全体のフォントに適用 */
html, body, div, span, input, textarea, button, p, h1, h2, h3, h4, h5, h6 {
    font-family: 'Noto Sans JP', sans-serif !important;
}

/* Streamlit 内部クラスも上書き */
[class^="css"], [class*="css"] {
    font-family: 'Noto Sans JP', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

def show_card(content_html: str):
    html = textwrap.dedent(content_html)
    st.markdown(
        f"""<div class="card">{html}</div>""",
        unsafe_allow_html=True
    )

st.markdown("""
<style>
/* ページ全体の背景を黒に */
html, body, .stApp {
    background-color: #000000 !important;
}

/* サイドバーの背景も黒に */
section[data-testid="stSidebar"] {
    background-color: #000000 !important;
}

/* サイドバー内部（要素の背景）も黒に */
section[data-testid="stSidebar"] .css-1d391kg, 
section[data-testid="stSidebar"] .css-1v3fvcr {
    background-color: #000000 !important;
}

/* Streamlit のツールバー（右上の Share, GitHub など）*/
header[data-testid="stHeader"] {
    background-color: #000000 !important;
}

/* ヘッダー内部の背景（余白部分も黒に） */
header[data-testid="stHeader"] div {
    background-color: #000000 !important;
}

/* ツールバーコンテナ */
div[data-testid="stToolbar"] {
    background-color: #000000 !important;
}

/* アイコンの色も緑 or 白にしたい場合： */
div[data-testid="stToolbar"] * {
    color: #ffffff !important;
}

/* カードとの余白を確保 */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
            
</style>
""", unsafe_allow_html=True)

st.markdown("""

<style>

/* ============================================
   0. Streamlit 全体は白文字（基本）
   ============================================ */
.stApp * {
    color: #ffffff !important;
}

/* ============================================
   1. 白背景コンポーネントの黒文字ルール
   ============================================ */

/* FileUploader */
[data-testid="stFileUploader"],
[data-testid="stFileUploader"] *,
[data-testid="stFileUploaderDropzoneInstructions"] {
    color: #000 !important;
}

/* TextInput / TextArea */
input, input *, textarea, textarea * {
    color: #000 !important;
    -webkit-text-fill-color: #000 !important;
}

/* TextArea コンテナそのもの */
.stTextArea, .stTextArea * {
    color: #000 !important;
}

/* Buttons（白背景なら黒文字） */
button, button * {
    color: #000 !important;
}

/* プレースホルダー */
textarea::placeholder {
    color: #555 !important;
}

/* ============================================
   2. ヘッダー（ログアウト等）は黒文字
   ============================================ */
header[data-testid="stHeader"] *,
[data-testid="stToolbar"] * {
    color: #000 !important;
}

/* ============================================
   3. h1（大見出し）は白文字固定
   ============================================ */
h1, h1 * {
    color: #fff !important;
}

/* Markdown の h1 */
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h1 * {
    color: #fff !important;
}

/* ログイン画面 */
.element-container h1, .element-container h1 * {
    color: #fff !important;
}
.element-container input, .element-container textarea {
    color: #000 !important;
}

/* ============================================
   4. Markdown の code / pre ブロック を黒文字に強制
   （←今回見えなかった根本原因）
   ============================================ */
code,
code *,
pre,
pre *,
.stMarkdown code,
.stMarkdown pre {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

/* code ブロックの背景が白なので文字が読める */
pre, code {
    background: #f5f5f5 !important;
}

/* ============================================
   5. Sidebar は白文字
   ============================================ */
[data-testid="stSidebar"],
[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* ============================================
   6. Spinner（インジケータ）は緑色
   ============================================ */
svg[role="img"],
div[role="status"] svg,
[data-testid="stStatusWidget"] svg {
    color: #00ff88 !important;
    stroke: #00ff88 !important;
    stroke-width: 2px !important;
}
/* =========================================================
   ★ ボタン内部のテキストを必ず黒にする（最優先）
   ========================================================= */
button *, button p, button div, button span {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important; /* Safari 対策 */
}

</style>



""", unsafe_allow_html=True)

def latex_to_png_base64(latex_str):
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.patch.set_facecolor("none")
    plt.axis("off")

    # LaTeX を描画
    plt.text(0.5, 0.5, f"${latex_str}$", fontsize=22, ha="center", va="center")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")

try:
    from causalimpact import CausalImpact
    _CAUSALIMPACT_OK = True
except Exception:
    _CAUSALIMPACT_OK = False


# ユーザーデータの作成（実際には安全な方法で保存する必要があります）
user_data = {
    'yomiko_mcc':'admin4035',
    'user1': 'password1',
    'user2': 'password2',
    # 他のユーザー情報
}

def func_fit(x, a, b, K):
    y = K / (1 + (a * x ** b))
    return y

def convert_df(df):
    return df.to_csv().encode('utf-8')

def download(df):
    df = convert_df(df)
    st.download_button(
        label="Download data as CSV",
        data=df,
        file_name='output.csv',
        mime='text/csv',
    )

# Excelデータ作成関数
def create_excel_file():
    output = BytesIO()  # メモリ上にバイナリデータを格納
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        fin_data.to_excel(writer, sheet_name='program×brand', index=True)
        allocated_program_data.to_excel(writer, sheet_name='allocated_program_data', index=True)
        view_track.to_excel(writer, sheet_name='view_track', index=True)
        fin_view_rate_list.to_excel(writer, sheet_name='fin_view_rate_list', index=True)
        allocated_brand_data.to_excel(writer, sheet_name='allocated_brand_cost', index=True)
    output.seek(0)  # ファイルポインタを先頭に戻す
    return output

def login():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        return True

    st.title("ログイン")
    username = st.text_input("ユーザー名")
    password = st.text_input("パスワード", type='password')

    if st.button("ログイン"):
        if username in user_data and user_data[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username  # ユーザー名をセッション状態に保存
            st.rerun()  # ページをリロードしてメインページに移動
        else:
            st.error("ユーザー名またはパスワードが正しくありません。")
    return False


def tab_PCA():
    # ======= カードUI：説明ブロック =======
    show_card(
"""
<h2>主成分分析（PCA）</h2>

<h3>目的</h3>
<ul>
    <li>多数の説明変数に潜む共通因子を抽出し、次元圧縮して全体構造を把握する。</li>
</ul>

<h3>使用ケース</h3>
<ul>
    <li><b>多変量の要約</b>：媒体接触や属性が多いときに、少数の指標（主成分）へ要約。</li>
    <li><b>可視化</b>：2次元に圧縮してクラスタ傾向・外れ値を把握。</li>
    <li><b>前処理</b>：回帰やクラスタリング前に多重共線性を緩和。</li>
</ul>

<h3>inputデータ</h3>
<ul>
    <li>1列目：<b>ID（y）</b></li>
    <li>2列目以降：<b>説明変数（X）</b>（数値列）</li>
    <li>※Excel/CSV対応。Excelは <b>A_入力</b> シートがあれば優先、無ければ先頭シートを読み込み。</li>
</ul>

<h3>アウトプット説明</h3>
<ul>
    <li><b>固有値・寄与率・累積寄与率</b>：どの主成分がどれだけ分散を説明するか。</li>
    <li><b>成分負荷量（loadings）</b>：各変数が主成分へどれだけ寄与するか。</li>
    <li><b>スコア（scores）</b>：各サンプルの主成分空間上の座標。</li>
    <li><b>スクリープロット</b> と <b>バイプロット（PC1×PC2）</b> を表示。</li>
    <li><b>CSVダウンロード</b>：成分負荷量・スコアを保存可能。</li>
</ul>
"""
    )
    
     # ここで Python 側でダウンロードボタンを表示
    with open("app/主成分OR因子分析.xlsx", "rb") as f:
        logistic_file = f.read()

    st.download_button(
        label="📥 入力シートをダウンロード",
        data=logistic_file,
        file_name="主成分OR因子分析.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # === ファイル入力 ===
    up = st.file_uploader("PCA用ファイル（CSV / XLSX）をアップロードしてください", type=["csv", "xlsx"], key="pca_file")
    if up is None:
        return

    # === 読み込み ===
    try:
        if up.name.lower().endswith(".xlsx"):
            bytes_data = up.read()
            xls = pd.ExcelFile(BytesIO(bytes_data))
            sheet = "A_入力" if "A_入力" in xls.sheet_names else xls.sheet_names[0]
            df = pd.read_excel(BytesIO(bytes_data), sheet_name=sheet)
        else:
            try:
                df = pd.read_csv(up)
            except UnicodeDecodeError:
                up.seek(0)
                df = pd.read_csv(up, encoding="shift-jis")
    except Exception as e:
        st.error(f"読み込みエラー: {e}")
        return

    if df.shape[1] < 2:
        st.error("少なくとも2列（1列目=ID、2列目以降=評価項目・イメージ項目）が必要です。")
        return

    st.write("データプレビュー：")
    st.dataframe(df.head())

    # === y / X 分割（1列目=ID, 2列目以降=説明変数） ===
    y = df.iloc[:, 0]
    X_raw = df.iloc[:, 1:].copy()

    # 数値列のみ利用（非数値は除外）
    X_num = X_raw.select_dtypes(include=[np.number])
    dropped = [c for c in X_raw.columns if c not in X_num.columns]
    if dropped:
        st.warning(f"数値でない列を除外しました: {', '.join(map(str, dropped))}")

    # 欠損値処理
    na_opt = st.radio("欠損値の扱い", ["行ごとに削除（推奨）", "列平均で補完"], index=0, horizontal=True)
    if na_opt == "行ごとに削除（推奨）":
        data = pd.concat([y, X_num], axis=1).dropna()
        y = data.iloc[:, 0]
        X_num = data.iloc[:, 1:]
    else:
        X_num = X_num.fillna(X_num.mean())

    if X_num.shape[1] == 0 or X_num.shape[0] < 2:
        st.error("有効な数値データが不足しています。")
        return

    # スケーリング（平均0, 分散1）
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_num)

    # === 成分数の指定方法 ===
    st.subheader("成分数の指定")
    mode = st.radio("選択", ["個数を指定", "累積寄与率で自動"], index=1, horizontal=True)

    if mode == "個数を指定":
        k_max = min(X_num.shape[1], 20)
        n_components = st.slider("主成分の個数", min_value=1, max_value=k_max, value=min(2, k_max), step=1)
        pca = PCA(n_components=n_components, random_state=0)
    else:
        thr = st.slider("累積寄与率（例：0.80〜0.99）", min_value=0.50, max_value=0.99, value=0.90, step=0.01)
        pca = PCA(n_components=thr, random_state=0)

    # === PCA 実行 ===
    try:
        scores = pca.fit_transform(X_std)
    except Exception as e:
        st.error(f"PCA実行エラー: {e}")
        return

    comps = pca.components_                  # 形状: [n_components, n_features]
    expvar = pca.explained_variance_ratio_   # 各成分の寄与率
    cumexp = np.cumsum(expvar)

    # === テーブル類 ===
    pc_names = [f"PC{i+1}" for i in range(len(expvar))]
    loadings = pd.DataFrame(comps.T, index=X_num.columns, columns=pc_names)
    loadings_abs = loadings.abs().sort_values(pc_names[0], ascending=False)

    scores_df = pd.DataFrame(scores, columns=pc_names, index=X_num.index)
    scores_df.insert(0, y.name if hasattr(y, "name") and y.name is not None else "target", y.loc[scores_df.index].values)

    exp_table = pd.DataFrame({
        "PC": pc_names,
        "explained_variance_ratio": expvar,
        "cumulative_ratio": cumexp
    })

    st.subheader("寄与率")
    st.dataframe(exp_table)

    st.subheader("成分負荷量（loadings）")
    st.caption("※数値の絶対値が大きいほど、その変数が該当主成分に強く寄与")
    st.dataframe(loadings_abs)

    st.subheader("スコア（各サンプルのPC座標）")
    st.dataframe(scores_df.head())

    # === ダウンロード ===
    st.download_button(
        "成分負荷量CSVをダウンロード",
        data=loadings.to_csv(index=True).encode("utf-8"),
        file_name="pca_loadings.csv",
        mime="text/csv"
    )
    st.download_button(
        "スコアCSVをダウンロード",
        data=scores_df.to_csv(index=True).encode("utf-8"),
        file_name="pca_scores.csv",
        mime="text/csv"
    )

    # === スクリープロット ===
    st.subheader("スクリープロット（寄与率）")
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(range(1, len(expvar) + 1), expvar, marker='o', label='Explained variance ratio')
    ax1.plot(range(1, len(cumexp) + 1), cumexp, marker='o', linestyle='--', label='Cumulative')
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Ratio")
    ax1.set_xticks(range(1, len(expvar) + 1))
    ax1.legend()
    st.pyplot(fig1)

    # === バイプロット（PC1×PC2） ===
    if len(pc_names) >= 2:
        st.subheader("バイプロット（PC1 × PC2）")
        fig2, ax2 = plt.subplots(figsize=(6, 6))

        # スコア散布
        ax2.scatter(scores_df["PC1"], scores_df["PC2"], alpha=0.6)
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        ax2.axhline(0, linewidth=0.5)
        ax2.axvline(0, linewidth=0.5)

        # 矢印（変数ベクトル）：成分負荷量を可視化
        # スケーリング（見やすさ調整）
        arrow_scale = 1.0
        load2 = loadings[["PC1", "PC2"]].values * arrow_scale

        for i, var in enumerate(X_num.columns):
            ax2.arrow(0, 0, load2[i, 0], load2[i, 1], head_width=0.02, length_includes_head=True)
            ax2.text(load2[i, 0]*1.07, load2[i, 1]*1.07, var, fontsize=9)

        ax2.set_title("Biplot")
        st.pyplot(fig2)
    else:
        st.info("PCが1つのため、バイプロットは表示しません。")



def tab_Logistic():

    show_card(
    """
    <h2>Logistic回帰</h2>

    <h3>目的</h3>
    <ul>
        <li>ある特定の事象が起きる確率を分析し、結果を予測する。</li>
    </ul>

    <h3>使用ケース</h3>
    <ul>
        <li>調査結果の個票データ解析: 説明変数として各メディアの接触有無（0,1データ）、目的変数として認知などのKPI有無（0,1データ）を使用して、各メディアの接触がKPIに与える影響を定量化する。GoogleトレンドやDS.INSIGHTなどからKWボリュームの過去傾向を分析し、季節性や長期トレンドを確認。</li>
        <li>CV起点でのCP評価: IDベースに、CPごとにFQしたかどうかを説明変数として（0,1データ）、ある指定期間内にCVしたかどうかを目的変数としたときに（0,1データ）、過去蓄積効果があったのか確認する。</li>
    </ul>

    <h3>inputデータ</h3>
    <ul>
        <li>1列目：ID</li>
        <li>2列目：目的変数（0or1）</li>
        <li>3列目以降：説明変数（数値列）</li>
    </ul>

    <h3>アウトプット説明</h3>
    <ul>
        <li><b>★importance</b>: 説明変数（各メディア接触有無）が目的変数（KPI）に与える貢献度をはかるための指標。</li>
        <li><b>odds</b>: オッズ比。importanceと大小関係は基本同じ。1より大きいならKPIに対して＋に働く、1より低いなら－に働く。</li>
        <li><b>P>|z|</b>：P値。有意水準0.05を下回ればその説明変数は有意な偏回帰係数であることが言える。</li>
        <li>inputデータの目的変数と説明変数の入力位置に注意。</li>
    </ul>
    """
    )

    # ここで Python 側でダウンロードボタンを表示
    with open("app/Logistic.xlsx", "rb") as f:
        logistic_file = f.read()

    st.download_button(
        label="📥 入力シートをダウンロード",
        data=logistic_file,
        file_name="Logistic.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    uploaded_file = st.file_uploader("ファイルをアップロードしてください", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            st.write("アップロードされたファイルの中身:")
            if uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                bytes_data = uploaded_file.read()
                df = pd.read_excel(BytesIO(bytes_data))
            else:
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                df = pd.read_csv(stringio, encoding="shift-jis")

            # === 読み込みはそのまま（df ができている前提） ===
            st.write(df)

            # 1列目=目的変数、2列目以降=説明変数（※0始まりに注意）
            y = df.iloc[:, 0]
            X = df.iloc[:, 1:].copy()

            # 数値化（文字が混ざっていたら NaN→除外/補完）
            X = X.apply(pd.to_numeric, errors='coerce')
            drop_na_opt = st.radio("欠損の扱い", ["行削除（推奨）", "列平均で補完"], index=0, horizontal=True)
            if drop_na_opt == "行削除（推奨）":
                data = pd.concat([y, X], axis=1).dropna()
                y = data.iloc[:, 0]
                X = data.iloc[:, 1:]
            else:
                X = X.fillna(X.mean())
                ok_idx = y.notna()
                y = y[ok_idx]
                X = X.loc[ok_idx]

            # 目的変数は0/1に揃える（すでに0/1ならそのまま）
            try:
                y = pd.to_numeric(y, errors='raise')
            except Exception:
                y = y.map({True: 1, False: 0})
            y = (y > 0).astype(int)  # 0/1に正規化

            # 列名（特徴量名）を後で使うので保持
            name_list = list(X.columns)

            # 定数項を付与
            import statsmodels.api as sm
            X_const = sm.add_constant(X, has_constant='add')

            # === ロジスティック回帰（GLM, Binomial）: フォーミュラを使わない ===
            logistic = sm.GLM(y, X_const, family=sm.families.Binomial()).fit()

            # 重要度の算出用に「1個だけ1、他0」の行列（定数項=1）を作る
            import numpy as np
            num = len(name_list)
            eye = np.zeros((num, num))
            np.fill_diagonal(eye, 1)

            df_dict = pd.DataFrame(eye, columns=name_list)
            df_dict.insert(0, 'const', 1.0)  # 定数項

            # 予測値（each feature = 1, others = 0 の時の確率）
            pred = logistic.predict(df_dict)

            # オッズ比とp値
            import numpy as np
            media_list = []
            odds_list = []
            p_values_list = []
            for i, col in enumerate(name_list):
                media_list.append(col)
                coef = logistic.params.get(col, np.nan)
                odds_list.append(np.exp(coef) if pd.notna(coef) else np.nan)
                p_values_list.append(logistic.pvalues.get(col, np.nan))

            df_odds = pd.DataFrame({
                "media": media_list,
                "importance": pred,   # 「その変数だけ1」のときの予測確率
                "odds": odds_list,
                "p_values": p_values_list
            })

            st.write(df_odds.head())
            download(df_odds)



        except Exception as e:
            st.error(f"ファイルを読み込む際にエラーが発生しました: {e}")

def tab_LogisticNum():

    show_card(
    """
    <h2>順序Logistic回帰</h2>

    <h3>目的</h3>
    <ul>
        <li>段階的（順序あり）な目的変数を、説明変数で説明・予測する。</li>
    </ul>

    <h3>使用ケース</h3>
    <ul>
        <li>満足度1〜5、評価A/B/C などの <b>順序ありカテゴリ</b> を扱いたいとき。</li>
    </ul>

    <h3>inputデータ</h3>
    <ul>
        <li>1列目：ID</li>
        <li>2列目：目的変数（順序カテゴリ or 数値/ラベル）</li>
        <li>3列目以降：説明変数（数値列）</li>
    </ul>
    """
    )

    # ここで Python 側でダウンロードボタンを表示
    with open("app/順序Logistic.xlsx", "rb") as f:
        logistic_file = f.read()

    st.download_button(
        label="📥 入力シートをダウンロード",
        data=logistic_file,
        file_name="順序Logistic.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


    up = st.file_uploader("ファイル（CSV / XLSX）をアップロード", type=["csv", "xlsx"], key="ordlogit_file")
    if up is None:
        return

    # --- 読み込み ---
    try:
        if up.name.lower().endswith(".xlsx"):
            bytes_data = up.read()
            xls = pd.ExcelFile(BytesIO(bytes_data))
            sheet = "A_入力" if "A_入力" in xls.sheet_names else xls.sheet_names[0]
            df = pd.read_excel(BytesIO(bytes_data), sheet_name=sheet)
        else:
            try:
                df = pd.read_csv(up)
            except UnicodeDecodeError:
                up.seek(0)
                df = pd.read_csv(up, encoding="shift-jis")
    except Exception as e:
        st.error(f"読み込みエラー: {e}")
        return

    if df.shape[1] < 3:
        st.error("少なくとも3列（1列目=ID、2列目=目的、3列目以降=説明変数）が必要です。")
        return

    st.write("データプレビュー：")
    st.dataframe(df.head())

    # --- y / X ---
    y_raw = df.iloc[:, 1]
    X = df.iloc[:, 2:].copy()
    X = X.apply(pd.to_numeric, errors='coerce')  # 非数値→NaN

    # 欠損処理
    na_opt = st.radio("欠損値の扱い", ["行ごとに削除（推奨）", "列平均で補完"], index=0, horizontal=True)
    if na_opt == "行ごとに削除（推奨）":
        data = pd.concat([y_raw, X], axis=1).dropna()
        y_raw = data.iloc[:, 0]
        X = data.iloc[:, 1:]
    else:
        X = X.fillna(X.mean())
        ok = y_raw.notna()
        y_raw = y_raw[ok]
        X = X.loc[ok]

    # 目的の順序（自動推定）
    uniq = pd.Index(pd.Series(y_raw).dropna().unique())
    try:
        uniq_sorted = pd.Index(sorted(pd.to_numeric(uniq, errors="raise")))
    except Exception:
        uniq_sorted = pd.Index(sorted(uniq.astype(str)))

    st.subheader("目的変数の順序")
    st.caption("※自動（昇順）を推奨。必要なら逆順に切り替え。")
    reverse = st.checkbox("順序を逆転する", value=False)
    categories = list(uniq_sorted[::-1] if reverse else uniq_sorted)

    # カテゴリ型（順序あり）へ
    cat_type = pd.api.types.CategoricalDtype(categories=categories, ordered=True)
    y = y_raw.astype(cat_type)

    # 標準化オプション
    do_std = st.checkbox("説明変数を標準化（平均0, 分散1）", value=True)
    if do_std:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_std = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    else:
        X_std = X

    # --- モデル学習（行列指定なので列名に記号があってもOK） ---
    try:
        model = OrderedModel(y, X_std, distr="logit")
        res = model.fit(method="bfgs", disp=False)
    except Exception as e:
        st.error(f"学習エラー: {e}")
        return

    # statsmodels の版差を吸収してカテゴリ名を取る
    def get_categories_safe():
        try:
            return list(res.model.endog.categories)   # 新しめ
        except Exception:
            pass
        try:
            return list(y.cat.categories)             # 手元のyから
        except Exception:
            pass
        try:
            k = res.predict(X_std.iloc[:1], which="prob").shape[1]
        except Exception:
            k = 2
        return [str(i) for i in range(k)]

    cats = [str(c) for c in get_categories_safe()]

    st.subheader("推定結果サマリ")
    st.text(res.summary().as_text())

    # 係数とp値（cut点は後で）
    coef = res.params.reindex(X_std.columns, fill_value=np.nan)
    pvals = res.pvalues.reindex(X_std.columns, fill_value=np.nan)
    odds = np.exp(coef)
    coef_df = pd.DataFrame({
        "coef": coef,
        "odds_ratio(単位増加)": odds,
        "p_value": pvals
    }).sort_values("p_value")
    st.subheader("係数・オッズ比・p値（説明変数）")
    st.dataframe(coef_df)

    # === 段階別の“寄与”（Δ確率） ===
    st.subheader("段階ごとの寄与（変数を動かしたときのΔ予測確率）")

    X_base = X_std.mean().to_frame().T  # 他変数は平均（標準化ONなら0）

    def probs_at(dfrow):
        p = res.predict(dfrow, which="prob")
        # ndarray or DataFrame -> 1D ベクトル
        if hasattr(p, "values"):
            p = p.values
        return np.ravel(p)

    base_p = probs_at(X_base)

    rows = []
    for col in X_std.columns:
        x1 = X_base.copy()
        unique_vals = pd.unique(X[col].dropna())
        if set(unique_vals).issubset({0, 1}):
            # ダミー: 0→1
            x0 = X_base.copy()
            x0[col] = 0.0
            x1[col] = 1.0
            p0 = probs_at(x0)
            p1 = probs_at(x1)
            dp = p1 - p0
            step_desc = "0→1"
        else:
            # 連続: +1標準化単位（非標準化なら +1σ）
            step = 1.0 if do_std else X[col].std(ddof=0)
            x1[col] = X_base[col].iloc[0] + step
            p1 = probs_at(x1)
            dp = p1 - base_p
            step_desc = f"+{('1σ' if not do_std else '1(標準化単位)')}"

        for c, d in zip(cats, dp):
            rows.append({"variable": col, "category": str(c), "delta_prob": float(d), "change": step_desc})

    effect_df = pd.DataFrame(rows).sort_values(["variable", "category"])
    st.dataframe(effect_df)

    st.subheader("Δ予測確率（ピボット表示）")
    pivot_df = effect_df.pivot(index="variable", columns="category", values="delta_prob").fillna(0.0)
    st.dataframe(pivot_df.style.format("{:+.3f}"))

    # cut点（カテゴリ間のしきい値）
    cut_df = res.params.drop(index=X_std.columns, errors="ignore").to_frame(name="threshold")
    st.subheader("しきい値（カテゴリ間のcut）")
    st.dataframe(cut_df)

    # ===== 予測確率（全行） =====
    proba = res.predict(X_std, which="prob")   # ndarray or DataFrame
    # 確実に float の numpy 配列へ
    proba = np.asarray(proba, dtype=float)

    # モデルのカテゴリ順で列を付与
    prob = pd.DataFrame(proba,
                        columns=[f"P({c})" for c in cats],
                        index=X_std.index)

    # 予測カテゴリ
    pred_class = prob.idxmax(axis=1).str.replace("P(", "", regex=False).str.replace(")", "", regex=False)

    out = pd.concat([
        y_raw.reset_index(drop=True).rename("y_true"),
        pred_class.reset_index(drop=True).rename("y_pred"),
        prob.reset_index(drop=True)
    ], axis=1)

    st.subheader("予測結果（上位表示）")
    st.dataframe(out.head().style.format({col: "{:.3f}" for col in prob.columns}))

    # 一致率
    acc = (out["y_true"].astype(str) == out["y_pred"].astype(str)).mean()
    st.write(f"**Accuracy（単純一致率）:** {acc:.3f}")


    # ===== 効果プロット（選択変数 vs 予測確率） =====
    if len(X_std.columns) >= 1:
        st.subheader("効果プロット（選択変数 vs 予測確率）")
        target_var = st.selectbox("変数を選択", list(X_std.columns))
        ngrid = 50
        x_min, x_max = X_std[target_var].min(), X_std[target_var].max()
        grid = np.linspace(x_min, x_max, ngrid)

        X_base = X_std.mean().to_frame().T
        X_plot = pd.DataFrame(np.repeat(X_base.values, ngrid, axis=0), columns=X_std.columns)
        X_plot[target_var] = grid

        proba_plot = res.predict(X_plot, which="prob")
        proba_plot = np.asarray(proba_plot, dtype=float)

        p_plot = pd.DataFrame(proba_plot, columns=[str(c) for c in cats])

        fig, ax = plt.subplots(figsize=(7, 4))
        for c in p_plot.columns:
            ax.plot(grid, p_plot[c].values, label=c)
        ax.set_xlabel(f"{target_var}（標準化後）" if do_std else target_var)
        ax.set_ylabel("予測確率")
        ax.legend(title="カテゴリ")
        st.pyplot(fig)

def tab_MultipleRegression():
    show_card(
    """
    <h2>重回帰（自動変数選択）</h2>

    <h3>目的</h3>
    <ul>
        <li>多数の説明変数の中から <b>最適な組み合わせを自動で探索</b> し、最も予測精度が高い回帰モデルを構築する。</li>
        <li>人手では困難な <b>変数選択・モデル選択（feature selection）</b> を CV（クロスバリデーション）や情報量基準（AIC/BIC）で自動化する。</li>
    </ul>

    <h3>使用ケース</h3>
    <ul>
        <li>多くの説明変数の中から <b>どれが本当に効いているのか</b> を知りたい</li>
        <li>多重共線性が疑われ、<b>変数を最適に減らしたい</b></li>
        <li>CV（汎化性能）を見ながら <b>過学習しないモデル</b> を作りたい</li>
        <li>広告・媒体別の <b>影響度シェア（寄与）の算出</b> を行いたい</li>
        <li>売上や指標の <b>ドライバー分析（Driver Analysis）</b> をしたい</li>
    </ul>

    <h3>inputデータ</h3>
    <ul>
        <li><b>1列目：日付（y）</b>（日にち、週など）</li>
        <li><b>2列目：目的変数（y）</b>（売上、認知得点、CV数など）</li>
        <li><b>3列目以降：説明変数（x1, x2, ...）</b>（媒体費用、接触指標、属性など）</li>
        <li>CSV / Excel（<code>A_入力</code> シートがあれば優先）</li>
        <li>数値列のみ自動抽出し、非数値列は除外</li>
        <li>欠損値処理は以下から選択：</li>
        <ul>
            <li>行ごと削除（推奨）</li>
            <li>列平均補完</li>
        </ul>
    </ul>

    <h3>アウトプット説明</h3>
    <ul>
        <li><b>選択された最適モデル（使用された説明変数）</b></li>
        <li><b>係数（元スケールに戻して出力）</b></li>
        <li>標準化あり/なしを選択可能</li>
        <li><b>モデル評価指標</b></li>
        <ul>
            <li>CV-R²</li>
            <li>CV-RMSE</li>
            <li>AIC / BIC</li>
            <li>調整R²</li>
        </ul>
        <li><b>寄与分解（Contribution Table）</b></li>
        <ul>
            <li>変数ごとの寄与量（impact）</li>
            <li>平均寄与シェア（どの変数が重要か）</li>
        </ul>
        <li><b>CSVダウンロード</b></li>
        <ul>
            <li>係数表</li>
            <li>予測値・寄与分解表</li>
        </ul>
    </ul>
    """
    )

    # ここで Python 側でダウンロードボタンを表示
    with open("app/重回帰分析.xlsx", "rb") as f:
        logistic_file = f.read()

    st.download_button(
        label="📥 入力シートをダウンロード",
        data=logistic_file,
        file_name="重回帰分析.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    up = st.file_uploader("CSV / XLSX をアップロード", type=["csv", "xlsx"], key="regsel_file")
    if up is None:
        return

    # --- 読み込み ---
    try:
        if up.name.lower().endswith(".xlsx"):
            bytes_data = up.read()
            xls = pd.ExcelFile(BytesIO(bytes_data))
            sheet = "A_入力" if "A_入力" in xls.sheet_names else xls.sheet_names[0]
            df = pd.read_excel(BytesIO(bytes_data), sheet_name=sheet)
        else:
            try:
                df = pd.read_csv(up)
            except UnicodeDecodeError:
                up.seek(0)
                df = pd.read_csv(up, encoding="shift-jis")
    except Exception as e:
        st.error(f"読み込みエラー: {e}")
        return

    if df.shape[1] < 3:
        st.error("少なくとも3列（目的+説明変数）が必要です。")
        return

    st.write("プレビュー：")
    st.dataframe(df.head())

    # --- 列の整理 ---
    y = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    X_raw = df.iloc[:, 2:].copy()
    X_num = X_raw.apply(pd.to_numeric, errors="coerce")
    dropped = [c for c in X_raw.columns if c not in X_num.columns]
    if dropped:
        st.warning("数値化できない列を除外: " + ", ".join(map(str, dropped)))

    # 欠損処理
    na_opt = st.radio("欠損の扱い", ["行削除（推奨）", "列平均で補完"], index=0, horizontal=True)
    if na_opt == "行削除（推奨）":
        data = pd.concat([y, X_num], axis=1).dropna()
        y = data.iloc[:, 1].values.astype(float)
        X = data.iloc[:, 2:].copy()
    else:
        X = X_num.fillna(X_num.mean())
        ok = y.notna()
        y = y[ok].values.astype(float)
        X = X.loc[ok]

    feature_names = list(X.columns)
    p = len(feature_names)
    if p == 0:
        st.error("有効な説明変数がありません。")
        return

    # オプション
    st.subheader("探索設定")
    col1, col2, col3 = st.columns(3)

    with col1:
        criterion = st.selectbox(
            "最適化基準",
            ["CV-RMSE(最小)", "CV-R2(最大)", "AIC(最小)", "BIC(最小)", "調整R2(最大)"],
            index=0,
            help="""
            ▼最適化基準
            CV-RMSE：予測誤差が最小となるモデルを選択（推奨）
            CV-R2：説明力が最大となるモデルを選択
            AIC/BIC：モデルの複雑さにペナルティを与え、シンプルなモデルを選択
            調整R2：説明変数の数を考慮したR2（分かりやすい指標）
            """
        )

    with col2:
        kfold = st.number_input(
            "CV 分割数",
            min_value=3, max_value=10, value=5,
            help="""
            ▼CV分割数
            データを何分割して交差検証を行うかの指定。
            5～10 が一般的で、値が大きいほど汎化性能が安定します。
            """
        )

    with col3:
        max_vars = st.number_input(
            "最大使用変数数（計算抑制用）",
            min_value=1, max_value=min(p, 15), value=min(10, p),
            help="""
            ▼最大使用変数数
            モデルが採用する説明変数の上限。
            過学習を防ぎ、計算負荷を抑えるための設定です。
            """
        )

    method = st.radio(
        "探索法",
        ["前進選択（高速）", "ベストサブセット（上限kまで）"],
        index=0,
        horizontal=True,
        help="""
        ▼探索法
        ● 前進選択：一つずつ変数を追加して最適モデルを探索（高速）
        ● ベストサブセット：全ての変数組み合わせから最適モデルを探索（正確だが計算重い）
        """
    )

    std_on = st.checkbox(
        "説明変数を標準化して学習（推奨）",
        value=True,
        help="""
        ▼標準化
        説明変数のスケールを揃えることで、重回帰の係数比較や
        変数選択の安定性が向上します。（推奨設定）
        """
    )

    # --- 補助関数 ---
    def kfold_indices(n, k, seed=42):
        rng = np.random.default_rng(seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        return np.array_split(idx, k)

    def fit_ols(Xm, yv):
        Xm_const = sm.add_constant(Xm, has_constant='add')
        model = sm.OLS(yv, Xm_const)
        res = model.fit()
        return res

    def cv_scores(cols):
        idx_folds = kfold_indices(len(y), int(kfold))
        r2s, rmses = [], []
        for val_idx in idx_folds:
            tr_idx = np.setdiff1d(np.arange(len(y)), val_idx)
            Xtr, ytr = X.iloc[tr_idx][cols], y[tr_idx]
            Xva, yva = X.iloc[val_idx][cols], y[val_idx]

            if std_on:
                mu = Xtr.mean(axis=0)
                sd = Xtr.std(axis=0).replace(0, 1e-9)
                Xtrz = (Xtr - mu) / sd
                Xvaz = (Xva - mu) / sd
                res = fit_ols(Xtrz, ytr)
                yhat = res.predict(sm.add_constant(Xvaz, has_constant='add'))
            else:
                res = fit_ols(Xtr, ytr)
                yhat = res.predict(sm.add_constant(Xva, has_constant='add'))

            yva = np.asarray(yva, dtype=float)
            yhat = np.asarray(yhat, dtype=float)
            ss_res = np.sum((yva - yhat)**2)
            ss_tot = np.sum((yva - np.mean(yva))**2) + 1e-12
            r2s.append(1 - ss_res/ss_tot)
            rmses.append(np.sqrt(np.mean((yva - yhat)**2)))
        return float(np.mean(r2s)), float(np.mean(rmses))

    def info_scores(res, nobs, kparams):
        aic = res.aic
        bic = res.bic
        r2 = res.rsquared
        adjr2 = 1 - (1-r2)*(nobs-1)/(nobs-kparams-1)
        return aic, bic, adjr2

    # --- ベスト保持（※mu/sdも保存） ---
    best = {"cols": [], "score": None, "res": None, "cv_r2": None, "cv_rmse": None,
            "aic": None, "bic": None, "adjr2": None, "mu": None, "sd": None}

    def evaluate(cols):
        Xsub = X[cols]
        if std_on:
            mu = Xsub.mean(axis=0)
            sd = Xsub.std(axis=0).replace(0, 1e-9)
            Xz = (Xsub - mu) / sd
            res = fit_ols(Xz, y)
        else:
            mu, sd = None, None
            res = fit_ols(Xsub, y)
        aic, bic, adjr2 = info_scores(res, res.nobs, len(cols)+1)
        cv_r2, cv_rmse = cv_scores(cols)

        if criterion == "CV-RMSE(最小)":
            score = -cv_rmse
        elif criterion == "CV-R2(最大)":
            score = cv_r2
        elif criterion == "AIC(最小)":
            score = -aic
        elif criterion == "BIC(最小)":
            score = -bic
        else:
            score = adjr2
        return score, res, cv_r2, cv_rmse, aic, bic, adjr2, mu, sd

    import itertools

    if method == "前進選択（高速）":
        remaining = feature_names.copy()
        selected = []
        last_score = -1e18
        while remaining and len(selected) < max_vars:
            cand_best = None
            for c in remaining:
                cols = selected + [c]
                score, *rest = evaluate(cols)
                if (cand_best is None) or (score > cand_best[0]):
                    cand_best = (score, cols, rest)
            if cand_best and cand_best[0] > last_score + 1e-9:
                last_score = cand_best[0]
                selected = cand_best[1]
                r = cand_best[2]
                best.update({
                    "cols": selected.copy(),
                    "score": cand_best[0],
                    "res": r[0],
                    "cv_r2": r[1],
                    "cv_rmse": r[2],
                    "aic": r[3],
                    "bic": r[4],
                    "adjr2": r[5],
                    "mu": r[6],
                    "sd": r[7],
                })
                remaining = [c for c in remaining if c not in selected]
            else:
                break
    else:  # ベストサブセット
        for k in range(1, int(max_vars)+1):
            for cols in itertools.combinations(feature_names, k):
                cols = list(cols)
                score, *rest = evaluate(cols)
                if (best["res"] is None) or (score > best["score"]):
                    best.update({
                        "cols": cols.copy(),
                        "score": score,
                        "res": rest[0],
                        "cv_r2": rest[1],
                        "cv_rmse": rest[2],
                        "aic": rest[3],
                        "bic": rest[4],
                        "adjr2": rest[5],
                        "mu": rest[6],
                        "sd": rest[7],
                    })

    # --- 最終モデルの係数（元スケールへ戻す） ---
    cols = best["cols"]
    if best["res"] is None or len(cols) == 0:
        st.error("適切なモデルを見つけられませんでした。")
        return

    if std_on:
        beta_std = best["res"].params.copy()      # index: const + cols
        intercept_std = beta_std.loc["const"]
        coef_std = beta_std.drop(index="const")

        mu = best["mu"].reindex(cols)
        sd = best["sd"].reindex(cols).replace(0, 1e-9)

        coef_orig = (coef_std / sd).rename(index=dict(zip(coef_std.index, cols)))
        intercept_orig = float(intercept_std - np.sum((coef_std * mu / sd).values))
    else:
        params = best["res"].params.copy()
        intercept_orig = float(params.loc["const"])
        coef_orig = params.drop(index="const")
        coef_orig.index = cols

    coef_tbl = pd.DataFrame({
        "variable": ["(Intercept)"] + cols,
        "coef": [intercept_orig] + [coef_orig[c] for c in cols]
    })
    st.subheader("選択された変数と係数（元スケール）")
    st.dataframe(coef_tbl.style.format({"coef": "{:.6g}"}))

    st.caption(f"CV-R2={best['cv_r2']:.3f} / CV-RMSE={best['cv_rmse']:.3g} / AIC={best['aic']:.1f} / BIC={best['bic']:.1f} / 調整R2={best['adjr2']:.3f}")

    # --- 寄与（貢献度） ---
    Xm = X[cols].copy()
    contrib = pd.DataFrame({c: coef_orig[c] * Xm[c].values for c in cols}, index=Xm.index)
    contrib["intercept"] = intercept_orig
    contrib["y_hat"] = contrib.sum(axis=1)

    st.subheader("寄与分解（上位表示）")
    st.dataframe(contrib.head().style.format("{:.3g}"))

    # 平均寄与とシェア
    avg_contrib = contrib[cols].mean().rename("avg_contrib")
    total = np.sum(np.abs(avg_contrib.values)) + 1e-12
    share = (np.abs(avg_contrib) / total).rename("share_abs")
    contrib_summary = pd.concat([avg_contrib, share], axis=1).sort_values("share_abs", ascending=False)
    st.subheader("平均寄与とシェア（|平均寄与|ベース）")
    st.dataframe(contrib_summary.style.format({"avg_contrib": "{:.3g}", "share_abs": "{:.1%}"}))

    # ダウンロード
    st.download_button("係数テーブルをCSVでダウンロード",
        data=coef_tbl.to_csv(index=False).encode("utf-8-sig"),
        file_name="reg_selected_coefs.csv", mime="text/csv")

    out_df = pd.concat([pd.Series(y, name="y_true"), contrib], axis=1)
    st.download_button("寄与分解データをCSVでダウンロード",
        data=out_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="reg_contributions.csv", mime="text/csv")

def tab_SEM():
    import streamlit as st
    import pandas as pd
    import numpy as np
    from io import BytesIO

    show_card(
    """
    <h2>共分散構造分析（SEM）</h2>

    <h3>目的</h3>
    <ul>
        <li><b>仮説モデル</b>（因果関係を含む構造）と <b>測定モデル</b>（潜在因子と観測変数の関係）を同時に推定し、データがモデルにどれだけ適合しているかを評価する。</li>
        <li>回帰では表現できない <b>複雑な因果ネットワーク</b> や <b>潜在変数（心理指標・ブランド因子）</b> を扱える点が特徴。</li>
    </ul>

    <h3>使用ケース</h3>
    <ul>
        <li>「認知 → 好意 → 行動意図」のような <b>段階モデル（AISAS 等）</b> を検証したい</li>
        <li>ブランドイメージの複数項目から潜在因子（例：安心性・革新性）を定義し、それが <b>KPI にどう効くか</b> を分析したい</li>
        <li>実験・施策における <b>メディエーション（媒介分析）</b> を行いたい</li>
        <li>回帰分析よりも <b>理論ベースのモデルを明確に示したい場合</b>（レポート・プレゼンに強い）</li>
    </ul>

    <h3>inputデータ</h3>
    <ul>
        <li><b>1列目：目的変数（y）</b></li>
        <li><b>2列目以降：説明変数（x1, x2, ...）</b></li>
        <li>1行目はヘッダー（列名）</li>
        <li>数値列のみを対象（潜在変数を測る質問項目など）</li>
        <li>Excel の場合は <b>A_入力</b> シートがあると優先して読み込む</li>
    </ul>

    <h3>アウトプット説明</h3>
    <ul>
        <li><b>パス係数（regression paths）</b>：変数間の因果的影響の強さ</li>
        <li><b>因子負荷量（loadings）</b>：観測変数が潜在因子をどれだけ反映しているか</li>
        <li><b>標準化係数（std_est）</b>：単位の異なる指標を比較しやすい</li>
        <li><b>適合度指標（Fit indices）</b>
            <ul>
                <li><b>CFI / TLI</b>（0.90以上が目安）</li>
                <li><b>RMSEA</b>（0.08以下が良い）</li>
                <li><b>SRMR</b>（0.08以下が良い）</li>
                <li><b>AIC / BIC</b>（モデル比較に使用）</li>
            </ul>
        </li>
        <li><b>CSVダウンロード可能</b>（係数表・適合度・標準化解）</li>
    </ul>
    """)

    # ここで Python 側でダウンロードボタンを表示
    with open("app/SEM.xlsx", "rb") as f:
        logistic_file = f.read()

    st.download_button(
        label="📥 入力シートをダウンロード",
        data=logistic_file,
        file_name="SEM.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    if not _SEM_OK:
        st.error("semopy を読み込めませんでした。")
        if _SEM_ERR:
            name, msg, tb = _SEM_ERR
            st.write(f"Import error: {name}: {msg}")
            st.code(tb)
        st.stop()

    up = st.file_uploader("CSV / XLSX をアップロード", type=["csv","xlsx"], key="sem_file")
    if up is None: 
        return

    # --- 読み込み ---
    try:
        if up.name.lower().endswith(".xlsx"):
            bytes_data = up.read()
            xls = pd.ExcelFile(BytesIO(bytes_data))
            sheet = "A_入力" if "A_入力" in xls.sheet_names else xls.sheet_names[0]
            df = pd.read_excel(BytesIO(bytes_data), sheet_name=sheet)
        else:
            try:
                df = pd.read_csv(up)
            except UnicodeDecodeError:
                up.seek(0)
                df = pd.read_csv(up, encoding="shift-jis")
    except Exception as e:
        st.error(f"読み込みエラー: {e}")
        return

    if df.shape[1] < 2:
        st.error("少なくとも2列（1列目=目的、2列目以降=説明変数）が必要です。")
        return

    st.write("プレビュー：")
    st.dataframe(df.head())

    # y / X
    y_name = df.columns[0]
    X_names = list(df.columns[1:])
    # 数値化・欠損処理
    y = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    X = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    na_opt = st.radio("欠損値の扱い", ["行ごとに削除（推奨）", "列平均で補完"], index=0, horizontal=True)
    if na_opt == "行ごとに削除（推奨）":
        data = pd.concat([y, X], axis=1).dropna()
    else:
        X = X.fillna(X.mean())
        data = pd.concat([y, X], axis=1).dropna(subset=[y_name])

    data.columns = [y_name] + X_names  # 安全に列名揃え

    st.markdown("### モデル指定（lavaan 風）")
    st.caption("例）潜在因子F1をx1+x2で測定し、yはF1とx3から説明：  `F1 =~ x1 + x2`  /  `y ~ F1 + x3`")
    default_syntax = f"""# 測定モデル（必要なら）
# F1 =~ {(' + '.join(X_names[:2])) if len(X_names)>=2 else ''}

# 構造モデル（自動モデルは下のチェックで作れます）
{y_name} ~ {' + '.join(X_names)}
"""
    use_auto = st.checkbox("自動モデル（観測変数→目的変数のパスのみ）を使う", value=True)
    syntax = st.text_area("モデル式（semopy 形式）", value=default_syntax, height=160)

    if use_auto:
        syntax = f"{y_name} ~ " + " + ".join(X_names)

    st.code(syntax, language="markdown")

    if st.button("推定を実行"):
        try:
            # 平均構造あり（切片推定）
            model = ModelMeans(syntax)
            model.fit(data)
        except Exception as e:
            st.error(f"推定エラー: {e}")
            return

        # 係数（推定値・SE・z・p）
        try:
            # もっとも互換性が高い
            est = inspect(model)  
        except Exception:
            # 古い semopy 用
            try:
                est = model.parameters_dataframe()
            except Exception:
                est = model.inspect()

        st.subheader("推定結果（係数）")
        st.dataframe(est)

        # 適合度指標
        try:
            stats = get_sem_stats(model, data)
            fit_df = pd.DataFrame({
                "metric": ["CFI","TLI","RMSEA","SRMR","AIC","BIC","DOF","n_params"],
                "value": [stats.get("CFI"), stats.get("TLI"), stats.get("RMSEA"),
                          stats.get("SRMR"), stats.get("AIC"), stats.get("BIC"),
                          stats.get("DoF"), stats.get("n_params")]
            })
            st.subheader("適合度指標")
            st.dataframe(fit_df)
            st.caption("目安：CFI/TLI≥0.90、RMSEA≤0.08、SRMR≤0.08（文脈依存）")
        except Exception:
            st.info("適合度統計の算出に失敗しました。")

        # 標準化解（推奨：解釈しやすい）
        try:
            std_est = inspect(model, std_est=True)
            st.subheader("標準化解（標準化係数）")
            st.dataframe(std_est)
        except Exception:
            pass

        # 予測・残差（yのみ表示）
        try:
            y_pred = model.predict_factors(data)  # 潜在因子推定
        except Exception:
            y_pred = pd.DataFrame()

        try:
            implied = model.implied_covariance  # 暗黙の共分散
        except Exception:
            implied = None

        # 出力ダウンロード
        @st.cache_data
        def _csv_bytes(df_):
            return df_.to_csv(index=False).encode("utf-8-sig")

        st.download_button("係数テーブルをCSVでダウンロード",
                           data=_csv_bytes(est), file_name="sem_params.csv", mime="text/csv")

        if 'std_est' in locals():
            st.download_button("標準化解をCSVでダウンロード",
                               data=_csv_bytes(std_est), file_name="sem_std_params.csv", mime="text/csv")


def tab_MMM():
    show_card("""
    <h2>MMM（軽量版）</h2>

    <h3>目的</h3>
    <ul>
        <li>広告投資額（TV・Web・OOH など）が <b>KPI（売上・CV・指標）にどれだけ寄与しているか</b> を定量化する。</li>
        <li><b>アドストック（広告の遅効性）</b> と <b>飽和（逓減効果）</b> を考慮し、  
        より現実的な反応曲線を推定し、媒体別の <b>真の効果量（貢献度 / ROI）</b> を明らかにする。</li>
        <li>過去の投資実績から、<b>最適な投下配分</b> や <b>追加投資の限界効果（限界効用）</b> を可視化する。</li>
    </ul>
    <h3>使用ケース</h3>
    <ul>
        <li>複数媒体の投資額とKPIを使い、<b>媒体別ROI</b> を求めたい</li>
        <li>投資を増減した際の <b>予測インパクト</b> を見たい（例：10%増ならどれだけ伸びる？）</li>
        <li>広告主レポートで一般的な <b>寄与分解（contribution analysis）</b> を行いたい</li>
        <li>広告効果の <b>遅効性（翌週・翌月に効く）</b> をモデルに入れたい</li>
        <li><b>予算シミュレーション</b>（今後の投資配分の参考）にも使いたい</li>
    </ul>
    <h3>inputデータ</h3>
    <ul>
        <li><b>1列目：date（日付 / 週次 / 月次）</b></li>
        <li><b>2列目：y（KPI：売上・CV・検索数など）</b></li>
        <li><b>3列目以降：媒体費用（tv_spend / web_spend / sns_spend …）</b></li>
        <li>例：</li>
    </ul>
    <table>
    <tr><th>date</th><th>y</th><th>tv_spend</th><th>web_spend</th><th>sns_spend</th></tr>
    <tr><td>2024-01-01</td><td>1200</td><td>300</td><td>200</td><td>150</td></tr>
    </table>
    <ul>
        <li>CSV / Excel に対応</li>
        <li>欠損値は自動除外 or 平均補完</li>
        <li>数値列以外は自動除外</li>
    </ul>
    <h3>アウトプット説明</h3>
    <ul>
        <li><b>推定モデル（反応曲線）</b></li>
        <ul>
            <li>アドストック処理：広告の蓄積効果を再現</li>
            <li>Hill式：費用増加に伴う飽和（伸びにくさ）を再現</li>
        </ul>
        <li><b>媒体別の係数（影響度）</b>：変換後特徴量の係数</li>
        <li><b>寄与分解（Contribution）</b></li>
        <ul>
            <li>各媒体が y に与えた寄与額</li>
            <li>平均寄与シェア（最も貢献した媒体は？）</li>
        </ul>
        <li><b>反応曲線（Response Curve）</b></li>
        <ul>
            <li>投入額に応じて KPI がどう変化するか</li>
        </ul>
        <li><b>限界効用（dROI）</b></li>
        <ul>
            <li>追加投資1単位あたりの増加効果</li>
            <li>最適投資の検討に必須</li>
        </ul>
        <li><b>予算シミュレーション</b></li>
        <ul>
            <li>総予算を ±○% 変えた場合の KPI 変化を自動計算</li>
        </ul>
        <li><b>CSV ダウンロード</b></li>
        <ul>
            <li>寄与分解表</li>
            <li>係数表</li>
            <li>予測データ</li>
        </ul>
    </ul>
    """
    )
            # ここで Python 側でダウンロードボタンを表示
    with open("app/MMM.xlsx", "rb") as f:
        logistic_file = f.read()

    st.download_button(
        label="📥 入力シートをダウンロード",
        data=logistic_file,
        file_name="MMM.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


    up = st.file_uploader("CSV / XLSX をアップロード", type=["csv", "xlsx"], key="mmm_lite_file")
    if up is None:
        return

    # --- 読み込み ---
    try:
        if up.name.lower().endswith(".xlsx"):
            bytes_data = up.read()
            xls = pd.ExcelFile(BytesIO(bytes_data))
            sheet = "A_入力" if "A_入力" in xls.sheet_names else xls.sheet_names[0]
            df = pd.read_excel(BytesIO(bytes_data), sheet_name=sheet)
        else:
            try:
                df = pd.read_csv(up)
            except UnicodeDecodeError:
                up.seek(0)
                df = pd.read_csv(up, encoding="shift-jis")
    except Exception as e:
        st.error(f"読み込みエラー: {e}")
        return

    # 列名整形
    df.columns = pd.Index(df.columns).map(str)
    if df.shape[1] < 3:
        st.error("列は最低3列（date, y, spend...）が必要です。")
        return

    # 基本整形
    date_col = df.columns[0]
    y_col = df.columns[1]
    spend_cols = list(df.columns[2:])

    # 型変換
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception:
        st.warning("date 列を日付に変換できませんでした。文字列のままで処理します。")

    y = pd.to_numeric(df[y_col], errors="coerce")
    X_spend = df[spend_cols].apply(pd.to_numeric, errors="coerce")
    data = pd.concat([y, X_spend], axis=1).dropna()
    y = data.iloc[:, 0].values.astype(float)
    X_spend = data.iloc[:, 1:].copy()
    spend_cols = list(X_spend.columns)

    st.write("プレビュー：")
    st.dataframe(pd.concat([pd.Series(y, name=y_col), X_spend], axis=1).head())

        # --- ハイパラ設定（UIはそのまま使える） ---
    with st.expander("ハイパラ設定（必要なら変更）", expanded=False):
        alphas = st.multiselect("アドストック減衰 α 候補（0～0.99。高いほど長い遅効）",
                                [0.3, 0.5, 0.7, 0.85, 0.9], default=[0.5, 0.7, 0.85])
        betas = st.multiselect("飽和（Hill） β 候補（>0。小さいほど早く飽和）",
                               [0.5, 1.0, 2.0, 3.0], default=[1.0, 2.0])
        lam_grid = st.multiselect("Ridge α（正則化強さ）", [0.1, 1.0, 3.0, 10.0, 30.0], default=[1.0, 3.0, 10.0])
        kfold = st.number_input("CV分割数", min_value=3, max_value=10, value=5)

    if not alphas or not betas:
        st.error("α, β の候補は1つ以上選んでください。")
        return

    # --- 変換関数 ---
    def adstock_geometric(x, alpha):
        out = np.zeros_like(x, dtype=float)
        carry = 0.0
        for t, val in enumerate(np.asarray(x, dtype=float)):
            out[t] = val + alpha * carry
            carry = out[t]
        return out

    def hill_saturation(x, beta):
        x = np.asarray(x, dtype=float)
        if np.nanmax(x) == np.nanmin(x):
            return np.zeros_like(x)
        x_norm = (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-9)
        return x_norm ** (1.0 / beta)

    # --- NumPy版 RidgeCV（切片は自前で扱う） ---
    def r2_score(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12
        return 1.0 - ss_res / ss_tot

    def ridge_fit_predict(X_tr, y_tr, X_te, lam):
        # 標準化（訓練統計量で）
        mu = X_tr.mean(axis=0, keepdims=True)
        sd = X_tr.std(axis=0, keepdims=True) + 1e-9
        Xz_tr = (X_tr - mu) / sd
        Xz_te = (X_te - mu) / sd

        # 中心化して切片分離
        y_mu = y_tr.mean()
        y_center = y_tr - y_mu

        # (X^T X + lam I)β = X^T y
        XtX = Xz_tr.T @ Xz_tr
        p = XtX.shape[0]
        beta = np.linalg.solve(XtX + lam * np.eye(p), Xz_tr.T @ y_center)
        intercept = y_mu  # 標準化後の特徴量は平均0

        y_pred_tr = Xz_tr @ beta + intercept
        y_pred_te = Xz_te @ beta + intercept
        return beta, intercept, mu, sd, y_pred_tr, y_pred_te

    def kfold_indices(n, k, seed=42):
        rng = np.random.default_rng(seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        folds = np.array_split(idx, k)
        return folds

    # --- ハイパラ探索（各チャネルで同一 α/β を採用する簡易版） ---
    best_score = -np.inf
    best_cfg = None
    best_X = None

    for a in alphas:
        # アドストック
        X_ads = np.column_stack([adstock_geometric(X_spend[c].values, a) for c in spend_cols])

        for b in betas:
            # 飽和
            X_sat = np.column_stack([hill_saturation(X_ads[:, i], b) for i in range(X_ads.shape[1])])

            # ここでは CV 内で標準化するので、今はそのまま
            n = len(y)
            folds = kfold_indices(n, int(kfold), seed=42)

            best_lam = None
            best_cv = -np.inf
            best_fit = None

            for lam in lam_grid:
                scores = []
                for vi in range(len(folds)):
                    val_idx = folds[vi]
                    tr_idx = np.setdiff1d(np.arange(n), val_idx, assume_unique=False)

                    X_tr, y_tr = X_sat[tr_idx], y[tr_idx]
                    X_va, y_va = X_sat[val_idx], y[val_idx]

                    beta, intercept, mu, sd, y_pred_tr, y_pred_va = ridge_fit_predict(X_tr, y_tr, X_va, lam)
                    scores.append(r2_score(y_va, y_pred_va))

                cv_mean = float(np.mean(scores))
                if cv_mean > best_cv:
                    best_cv = cv_mean
                    best_lam = lam

            # ベスト lam で全データにフィット（最終モデル）
            beta, intercept, mu, sd, y_pred_tr, _ = ridge_fit_predict(X_sat, y, X_sat, best_lam)

            if best_cv > best_score:
                best_score = best_cv
                best_cfg = (a, b, best_lam, mu, sd, beta, intercept)
                # 最終の標準化特徴量
                X_trans = (X_sat - mu) / sd
                best_X = X_trans

    a_star, b_star, lam_star, mu_star, sd_star, coef_star, intercept_star = best_cfg
    st.success(f"Best CV R² = {best_score:.3f} | alpha={a_star} / beta={b_star} / ridge={lam_star}")

    # --- 学習済みで寄与分解 ---
    y_hat = best_X @ coef_star + intercept_star
    resid = y - y_hat

    # チャネル寄与（分解は線形のため、各列×係数）
    contrib = best_X * coef_star  # shape [T, K]
    contrib_df = pd.DataFrame(contrib, columns=spend_cols)
    contrib_df["intercept"] = intercept_star
    contrib_df["residual"] = resid
    st.subheader("寄与分解（head）")
    st.dataframe(contrib_df.head().style.format("{:.3f}"))

    # --- 反応曲線 & 限界効率（dROI） ---
    st.subheader("反応曲線（逓減）と限界効率")

    # 曲線は「単一チャネルだけを動かす」前提で作図（他は平均）
    ngrid = 50
    fig, axes = plt.subplots(len(spend_cols), 1, figsize=(7, 3*len(spend_cols)))
    if len(spend_cols) == 1:
        axes = [axes]

    for idx, ch in enumerate(spend_cols):
        base = X_spend.copy()
        x_raw = base[ch].values
        lo, hi = np.percentile(x_raw, [1, 99])
        grid = np.linspace(max(0, lo), hi, ngrid)

        # 他チャネルは平均固定、対象だけを grid に置換 → 変換 → 標準化 → 予測
        base_vals = base.mean().to_dict()
        curves = []
        drois = []

        for g in grid:
            tmp = base.copy()
            for c in spend_cols:
                tmp[c] = base_vals[c]
            tmp[ch] = g

            # adstock -> saturation -> standardize
            Xg_ads = np.column_stack([adstock_geometric(tmp[c].values, a_star) for c in spend_cols])
            Xg_sat = np.column_stack([hill_saturation(Xg_ads[:, i], b_star) for i in range(Xg_ads.shape[1])])
            Xg = (Xg_sat - mu_star) / sd_star

            y_pred = Xg @ coef_star + intercept_star
            curves.append(np.mean(y_pred))

        curves = np.array(curves)

        # 数値微分で限界効率（dROI相当）を算出（Δy / Δspend）
        droi = np.gradient(curves, grid)

        ax = axes[idx]
        ax.plot(grid, curves, label=f"Response: {ch}")
        ax2 = ax.twinx()
        ax2.plot(grid, droi, linestyle="--", label="Marginal effect (dROI)")

        ax.set_xlabel(f"{ch}（投入額）")
        ax.set_ylabel("予測KPI")
        ax2.set_ylabel("限界効率")
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")

    st.pyplot(fig)

    # --- 係数テーブル（解釈用） ---
    st.dataframe(contrib_df.head().round(3))

    coef_tbl = pd.DataFrame({"channel": spend_cols, "coef_on_transformed": coef_star})
    st.subheader("係数（変換後特徴量上）")
    st.dataframe(coef_tbl.assign(
        coef_on_transformed=lambda d: d["coef_on_transformed"].round(4)
    ))


    # --- 予算シミュ（全体×±x%） ---
    st.subheader("簡易予算シミュレーション")
    pct = st.slider("総予算を何%増減するか", min_value=-50, max_value=100, value=10, step=5)
    scale = 1.0 + pct/100.0
    spend_new = X_spend.mean() * scale

    tmp = X_spend.copy()
    for c in spend_cols:
        tmp[c] = spend_new[c]

    Xn_ads = np.column_stack([adstock_geometric(tmp[c].values, a_star) for c in spend_cols])
    Xn_sat = np.column_stack([hill_saturation(Xn_ads[:, i], b_star) for i in range(Xn_ads.shape[1])])
    Xn = (Xn_sat - mu_star) / sd_star
    y_pred_new = Xn @ coef_star + intercept_star

    st.write(f"平均KPI（現状）: {np.mean(y_hat):.3f} → 変更後: {np.mean(y_pred_new):.3f}（{pct:+d}%予算）")



def tab_STL():
    show_card(
    """
    <h2>STL分解</h2>

    <h3>目的</h3>
    <ul>
        <li>時系列データを <b>トレンド・季節性・残差</b> に分解し、  
            データの構造（周期性・長期傾向・異常値など）を把握する。</li>
    </ul>

    <h3>使用ケース</h3>
    <ul>
        <li>Googleトレンド・DS.INSIGHT などから KW ボリュームの  
            <b>季節性や長期トレンド</b> を確認したいとき</li>
    </ul>

    <h3>inputデータ</h3>
    <ul>
        <li><b>時系列データ</b>（期間 × KW ボリューム）を入力</li>
        <li>週次・月次どちらでも自動判別して処理します</li>
    </ul>

    <h3>アウトプット説明</h3>
    <ul>
        <li><b>_raw</b>：元の時系列データ</li>
        <li><b>_trend</b>：トレンド成分（長期的な増減）</li>
        <li><b>_seasonal</b>：季節成分（周期的変動）</li>
        <li><b>_resid</b>：残差成分（トレンド＋季節性を除去した後のノイズ）</li>
    </ul>

    <p>
    KW ボリュームの過去傾向を把握し、<b>季節性 or 長期トレンド</b> が  
    どれほど影響しているかを可視化できます。  
    <strong>週次・月次データどちらにも対応</strong>し、STLが自動的に処理します。
    </p>
    """
    )

        # ここで Python 側でダウンロードボタンを表示
    with open("app/STL分解.xlsx", "rb") as f:
        logistic_file = f.read()

    st.download_button(
        label="📥 入力シートをダウンロード",
        data=logistic_file,
        file_name="STL分解.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    if 'uploaded_file_tab2' not in st.session_state:
        st.session_state.uploaded_file_tab2 = None

    uploaded_file = st.file_uploader("STL分解用inputファイルをアップロードしてください", type=["csv", "xlsx"], key='tab2_uploader')

    if uploaded_file is not None:
        st.session_state.uploaded_file_tab2 = uploaded_file
        try:
            if uploaded_file.name.endswith("csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith("xlsx"):
                df = pd.read_excel(uploaded_file)
            
            st.write("データプレビュー:")
            st.write(df.head())

            period_num = (df.iat[1, 0] - df.iat[0, 0]).days
            data_num = df.shape[1] - 1
            df_date = df.iloc[:, 0]
            df = df.set_index("date")
            df.head()

            ##■周期の設定##
            if period_num > 7:
                period = 12
            elif period_num == 7:
                period = 52
            elif period_num == 1:
                period = 365
            else:
                period = 0
                print("任意の期間を設定してください。")

            print(period)

            ##■分解##
            result = pd.DataFrame()

            # DataFrame内の各列に対してループ処理
            for i in range(data_num):
                stl = sm.tsa.seasonal_decompose(df.iloc[:, i], period=period)
                name = df.columns.values[i]

                tmp = pd.DataFrame()
                tmp[str(name) + "_raw"] = df.iloc[:, i]
                tmp[str(name) + "_trend"] = stl.trend
                tmp[str(name) + "_seasonal"] = stl.seasonal
                tmp[str(name) + "_resid"] = stl.resid

                result = pd.concat([result, tmp], axis=1)

                # それぞれの系列ごとに独立したグラフを生成する
                fig, ax = plt.subplots()
                for column in tmp.columns:
                    if "_raw" in column or "_trend" in column or "_seasonal" in column:
                        ax.plot(df.index, tmp[column], label=column)  # DataFrame の index を X軸に使用してプロット

                ax.set_xlabel('Date')
                ax.set_ylabel('Value')
                ax.set_title('Decomposition of ' + str(name))  # グラフタイトル
                ax.legend()

                st.pyplot(fig)  # グラフを表示

            st.write(result)
            download(result)

        except Exception as e:
            st.error(f"ファイルを読み込む際にエラーが発生しました: {e}")



def tab_TIME():
    show_card("""
    <h2>TIME最適化</h2>

    <h3>目的</h3>
    <ul>
        <li>TIMEの複数素材割り付けを最適化する。</li>
    </ul>

    <h3>使用ケース</h3>
    <ul>
        <li>複数ブランドを TIME で放映する場合</li>
        <li>レギュラータイム / FTB / 単発タイムなど固定枠がある場合</li>
    </ul>

    <h3>inputデータ</h3>
    <ul>
        <li>A〜Dシートをそれぞれ入力</li>
    </ul>

    <p>
    <a href="https://hakuhodody.sharepoint.com/:f:/s/msteams_d8fd35/Eu6cDQ4W-t5KlsMGSjLhfQQBaYubS13B_Ge2FzODeaZO-A?e=lvq7tE" target="_blank">
    🔗 入力フォルダを開く
    </a>
    </p>

    <h3>アウトプット説明</h3>
    <ul>
        <li>ブランドごとの最適番組フォーメーション</li>
        <li>番組追加による累積リーチ</li>
        <li>最適化後のブランド別予算</li>
        <li>AシートとCシートの番組IDは「漏れなく・ダブりなく」処理</li>
    </ul>
    """
    )

    # ---- ここから下は従来の処理（そのままでOK） ----

    st.title("モード選択")

    # プルダウン選択肢
    options = ["reach cost", "reach", "target_cost"]
    mode = st.selectbox("モードを選択してください", options, index=2)

    # アップロードされたファイルがあるか確認
    if "uploaded_file" not in st.session_state:
        st.session_state["uploaded_file"] = None

    # ファイルアップロード
    uploaded_file = st.file_uploader("Excelファイルをアップロードしてください", type=["xlsx"])

    if uploaded_file is not None:
        try:
            st.write("アップロードされたファイルの中身を読み込み中...")
            # Excelファイルの全シートを取得
            if uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                bytes_data = uploaded_file.read()
                sheets = pd.read_excel(BytesIO(bytes_data), sheet_name=None)

                # 各シートを取得
                limit_data = sheets['A_Limit'].set_index(['Program_code', 'date'])
                brand_data = sheets['B_Brand'].set_index('Brand')
                view_data = sheets['C_View'].set_index('Sample')
                target_data = sheets['D_Target'].set_index('Brand')

                # 確認のためデータを表示
                st.write("### A_Limit シートのデータ")
                st.dataframe(limit_data.head())

                st.write("### B_Brand シートのデータ")
                st.dataframe(brand_data.head())

                st.write("### C_View シートのデータ")
                st.dataframe(view_data.head())

                st.write("### D_Target シートのデータ")
                st.dataframe(target_data.head())

            else:
                st.error("アップロードされたファイルはExcel形式ではありません。")

            st.write("読込終了")



            # 「無し」という値を空白に置き換え、必須番組データと除外データを作成
            exc_data = limit_data.copy()
            must_data = limit_data.copy()

            values_to_replace_exc = [15, 30, 60, 120, 240]
            values_to_replace_must = ["無し"]
            exc_data.replace(values_to_replace_exc, '', inplace=True)  # 除外の0-1データ
            must_data.replace(values_to_replace_must, '', inplace=True)  # 必須番組の割り振り秒数データ

            # ブランド名のリストを取得
            brand_names = brand_data.index.tolist()
            #ブランドの割り付け情報が入ってる
            temp_brand_data = limit_data.copy()
            temp_brand_data = temp_brand_data.drop(columns=[col for col in limit_data.columns if 'Cost/30' in col])
            temp_brand_data = temp_brand_data.drop(columns=[col for col in limit_data.columns if 'P_seconds' in col])
            temp_brand_data = temp_brand_data.drop(columns=[col for col in limit_data.columns if 'Program' in col])

            #番組のコストと秒数
            temp_program_data = limit_data[['Cost/30', 'P_seconds']]

            # 各ブランドの当初の予算を保存
            allocated_brand_data = brand_data.copy()  # 割り付けに使うブランドごとの予算
            initial_brand_budget = allocated_brand_data.copy()  # 割り付け前の初期予算
            used_brand_budget = pd.DataFrame(0, index=brand_names, columns=[120, 60, 30, 15])  # 割り当てられた予算のデータフレーム

            # 視聴データを保持する辞書（ターゲット層に基づく長さを設定）
            brand_view_data = {}
            # target_dataがDataFrameであることを仮定
            brand_target = target_data

            for brand_column in brand_names:
                # ブランドのターゲット年齢範囲と性別を取得
                target_age_range = brand_target.loc[brand_column, ['Low', 'High']]  # 年齢範囲
                target_gender = brand_target.loc[brand_column, 'Gender']  # 性別

                # ターゲット層に一致する視聴データを絞り込み
                if target_gender == 'MF':
                    # 「MF」ターゲットの場合、性別に関係なくすべての視聴者を選択
                    filtered_view_data = view_data[
                        (view_data['Age'] >= target_age_range[0]) & 
                        (view_data['Age'] <= target_age_range[1])
                    ]
                else:
                    # 指定された性別と年齢範囲に基づいて絞り込み
                    filtered_view_data = view_data[
                        (view_data['Age'] >= target_age_range[0]) & 
                        (view_data['Age'] <= target_age_range[1]) & 
                        (view_data['Gender'] == target_gender)
                    ]
                
                # ターゲット層に一致する視聴データのインデックス長さを取得
                filtered_index = filtered_view_data.index
                print(len(filtered_index))
                # ターゲット層に基づいて視聴データを初期化
                brand_view_data[brand_column] = pd.Series([False] * len(filtered_index), index=filtered_index)


            # 割り当て結果を記録するデータフレーム
            allocated_program_data = pd.DataFrame(columns=['Program_code', 'Brand', 'Allocated_seconds', 'Allocated_cost', 'New_Viewers'])

            #アロケのした後のフレーム
            fin_data = limit_data.copy()
            #====================================================

            st.write("設定終了")

            #セル3================================================
            # brand_targetがDataFrameで、'Brand'がインデックスとして設定されている場合
            for brand_column in temp_brand_data.columns:
                print(f"\n--- {brand_column} の処理 ---")

                for index, value in temp_brand_data[brand_column].items():
                    if value == "無し" or pd.isna(value):
                        continue  # "無し"や NaN の場合はスキップ

                    if value in [15, 30, 60, 120, 240]:  # valueが秒数として有効か確認
                        program_code, date = index  # 複合キーから program_code と date を取り出す
                        
                        print(program_code)

                        # 番組のコストと秒数を取得
                        program_cost = temp_program_data.loc[(program_code, date), 'Cost/30']
                        program_seconds = temp_program_data.loc[(program_code, date), 'P_seconds']

                        # ブランドの秒数を減らす
                        brand_seconds = value  # temp_brand_dataの値がそのまま秒数と仮定
                        program_seconds_remaining = program_seconds - brand_seconds  # 残り秒数を計算

                        # 番組の秒数を更新する（必要ならtemp_program_dataに反映）
                        temp_program_data.loc[(program_code, date), 'P_seconds'] = program_seconds_remaining

                        # ブランド名と今回の秒数に基づいてコストを取得
                        brand_cost = allocated_brand_data.loc[brand_column, value]  # ブランド名と秒数が一致するコストを取得
                        
                        # ブランドの秒数とコストを取得
                        brand_seconds = value  # temp_brand_dataの値がそのまま秒数と仮定
                        allocated_cost = program_cost * (brand_seconds / 30)  # コストを計算

                        allocated_brand_data.at[brand_column, value] -= allocated_cost
                        new_cost = allocated_brand_data.loc[brand_column, value]

                        # 試聴データをターゲット層（年齢・性別）に基づいて絞り込み
                        target_age_range = brand_target.loc[brand_column, ['Low', 'High']]  # 年齢範囲を取得
                        target_gender = brand_target.loc[brand_column,'Gender']  # 例: 'Female'

                        if target_gender == 'MF':
                            # 「MF」ターゲットの場合、性別に関係なくすべての視聴者を選択
                            filtered_view_data = view_data[
                                (view_data['Age'] >= target_age_range[0]) & 
                                (view_data['Age'] <= target_age_range[1])
                            ]
                        else:
                            # 指定された性別と年齢範囲に基づいて絞り込み
                            filtered_view_data = view_data[
                                (view_data['Age'] >= target_age_range[0]) & 
                                (view_data['Age'] <= target_age_range[1]) & 
                                (view_data['Gender'] == target_gender)
                            ]

                        # 視聴データを取得（重複を除いた新しい視聴者のみ）
                        past_viewer = brand_view_data[brand_column].copy()
                        brand_view_data[brand_column] |= filtered_view_data[program_code]
                        viewer_add = sum(brand_view_data[brand_column]) - sum(past_viewer)

                        # 情報を表示
                        """
                        print(f"Brand: {brand_column}, 秒数: {value}")
                        print(f"対応するコスト: {brand_cost}")
                        print(f"Program: {program_code}, Date: {date}")
                        print(f"Program Cost/30: {program_cost}, Program Seconds: {program_seconds}")
                        print(f"Brand Allocated Seconds: {brand_seconds}, Brand Allocated Cost: {allocated_cost}")
                        print(f"新しいブランド予算: {new_cost}")
                        print(f"残り番組秒数: {program_seconds_remaining}")
                        print("-" * 50)
                        print(f"元の視聴データ: {sum(past_viewer)}")
                        print(f"新規視聴データ: {sum(brand_view_data[brand_column])}")
                        print(f"新規獲得視聴者: {viewer_add}")
                        print(f"サンプル数: {len(brand_view_data[brand_column])}")
                        """

                        # 新しい行のデータを作成
                        new_row = pd.DataFrame({
                            'Program_code': [program_code],
                            'Brand': [brand_column],
                            'Allocated_seconds': [brand_seconds],
                            'Allocated_cost': [allocated_cost],
                            'New_Viewers': [viewer_add]
                        })
                        
                        # 既存のデータフレームに新しい行を追加する
                        allocated_program_data = pd.concat([allocated_program_data, new_row], ignore_index=True)
            #====================================================
           
            st.write("必須終了")

            #セル4================================================
            pd.set_option('mode.chained_assignment', None)  # チェーンされた代入の警告を無視
            import warnings
            warnings.simplefilter(action='ignore', category=FutureWarning)


            # view_track DataFrameの初期化
            view_track = pd.DataFrame(columns=['Brand', 'Round', 'New_Viewers', 'Total_Viewers', 'Reach_Rate'])

            # 初期化
            seconds_priorities = sorted(brand_data.columns, reverse=True)
            round_number = 0  # ラウンドカウンタ
            all_brands_done = False  # 全てのブランドの割り付けが終わったかを確認するフラグ
            allocated_program_data = pd.DataFrame(columns=['Program_code', 'Brand', 'date', 'Allocated_seconds', 'Allocated_cost', 'New_Viewers'])

            # 割り当て済みの番組コードと日付の組み合わせを保存するためのセット
            assigned_programs = set()

            # 割り付け可能なブランドがある限り繰り返すループ
            while not all_brands_done:
                print(f"\n--- ラウンド {round_number} ---")
                
                all_brands_done = True  # すべてのブランドが完了したか確認するために一旦Trueにする

                # 各ブランドごとに割り当てを行う
                for brand in brand_names:
                    program_assigned = False  # フラグを初期化
                    brand_new_viewers = 0  # このラウンドでの新規視聴者数を初期化

                    # ターゲット層（年齢・性別）に基づいて視聴データを絞り込み
                    target_age_range = brand_target.loc[brand, ['Low', 'High']]  # 年齢範囲
                    target_gender = brand_target.loc[brand, 'Gender']  # 性別

                    # ターゲット層に一致する視聴データを絞り込む
                    if target_gender == 'MF':
                        # 「MF」ターゲットの場合、性別に関係なくすべての視聴者を選択
                        filtered_view_data = view_data[
                            (view_data['Age'] >= target_age_range[0]) & 
                            (view_data['Age'] <= target_age_range[1])
                        ]
                    else:
                        # 指定された性別と年齢範囲に基づいて絞り込み
                        filtered_view_data = view_data[
                            (view_data['Age'] >= target_age_range[0]) & 
                            (view_data['Age'] <= target_age_range[1]) & 
                            (view_data['Gender'] == target_gender)
                        ]

                    # 優先する秒数の順にチェック
                    for seconds in seconds_priorities:
                        if program_assigned:  # 番組が割り当てられた場合は次のブランドに移行
                            break

                        brand_rest_cost = allocated_brand_data.at[brand, seconds]
                        program_cost_arr = temp_program_data['Cost/30'] * (seconds / 30)
                        program_seconds_arr = temp_program_data['P_seconds']

                        if (program_cost_arr > brand_rest_cost).all():
                            print(f"{brand}の{seconds}は予算上限に達しています。")
                            continue

                        if (program_seconds_arr < seconds).all():
                            print(f"{brand}の{seconds}に割り当てられる番組秒数がありません。")
                            continue

                        # もし予算が残っていれば番組を割り当てる
                        if allocated_brand_data.at[brand, seconds] > 0:
                            best_program = None
                            best_new_viewers = 0
                            best_allocated_seconds = 0
                            best_date = None

                            temp_df = pd.DataFrame()
                            past_viewer = brand_view_data[brand].copy()  # ここでコピーを取る

                            # 最適な番組を選ぶための処理
                            for index, value in temp_brand_data[brand].items():
                                program_code, date = index

                                # 既に割り当てられた番組・日付の組み合わせをチェック
                                if (program_code, date, brand) in assigned_programs:
                                    print(f"{brand} に対して、プログラム {program_code}, 日付 {date} は既に割り当て済みです。")
                                    continue

                                # "無し" または視聴データがNaNでない場合はスキップ
                                if value == "無し" or not pd.isna(value):
                                    continue

                                # 番組のコストと秒数を取得
                                program_cost = temp_program_data.at[(program_code, date), 'Cost/30'] * (seconds / 30)
                                program_seconds = temp_program_data.at[(program_code, date), 'P_seconds']

                                # 割り当て可能な秒数を確認
                                if program_seconds < seconds:
                                    continue

                                # コスト確認
                                if allocated_brand_data.at[brand, seconds] < program_cost:
                                    continue

                                # 過去の視聴者数を保持し、新たな視聴者数を計算
                                if program_code in filtered_view_data.columns:
                                    new_viewers = filtered_view_data[program_code]
                                    target_cost = new_viewers.sum() / program_cost

                                    # 既存の視聴者データと結合（視聴した人を1とする場合）
                                    temp_brand_view_data = past_viewer | new_viewers
                                    viewer_add = temp_brand_view_data.sum() - past_viewer.sum()
                                    viewer_add_per_cost = viewer_add / program_cost
                                else:
                                    viewer_add = 0

                                if viewer_add <= 0:
                                    continue

                                # 番組を追加
                                temp_data = pd.DataFrame({
                                    'program_code': [program_code],
                                    'date': [date],
                                    'viewer_add': [viewer_add],
                                    'viewer_add_per_cost': [viewer_add_per_cost],
                                    'target_cost': [target_cost]
                                })

                                temp_df = pd.concat([temp_df, temp_data], ignore_index=True)

                            # temp_dfから最適な番組を選ぶ
                            if not temp_df.empty:
                                if mode == "reach":
                                    # リーチが最大のものを選ぶ
                                    best_row = temp_df.loc[temp_df["viewer_add"].idxmax()]
                                    if best_row["viewer_add"] > 0:  # 新規視聴者数が正の場合のみ割り付け
                                        best_program = best_row["program_code"]
                                        best_date = best_row["date"]
                                        best_new_viewers = best_row["viewer_add"]

                                elif mode == "reach_cost":
                                    # リーチ増分に対するコスト効率が最も高いものを選ぶ
                                    best_row = temp_df.loc[temp_df["viewer_add_per_cost"].idxmin()]
                                    if best_row["viewer_add"] > 0:  # 新規視聴者数が正の場合のみ割り付け
                                        best_program = best_row["program_code"]
                                        best_date = best_row["date"]
                                        best_new_viewers = best_row["viewer_add"]

                                elif mode == "target_cost":
                                    # target_costが最も小さいものを選ぶ（必ず割り付け）
                                    best_row = temp_df.loc[temp_df["target_cost"].idxmin()]
                                    best_program = best_row["program_code"]
                                    best_date = best_row["date"]
                                    best_new_viewers = best_row["viewer_add"]
                                    print("tgコストで選んでる")

                            # 最適な番組が見つかった場合の処理
                            if best_program and best_date is not None:
                                # 割り当てた番組の処理（コストの減算や視聴者データの更新など）
                                best_program_cost = temp_program_data.at[(best_program, best_date), 'Cost/30'] * (seconds / 30)
                                allocated_brand_data.at[brand, seconds] -= best_program_cost
                                temp_program_data.at[(best_program, best_date), 'P_seconds'] -= seconds
                                new_viewers = filtered_view_data[best_program]  # 視聴データの更新
                                brand_view_data[brand] = past_viewer | new_viewers  # 既存の視聴者データと結合（視聴した人を1とする場合）
                                total_viewers = brand_view_data[brand].sum()
                                sample_num = len(brand_view_data[brand_column])
                                view_rate = total_viewers / sample_num
                                
                                # 割り当て結果を表示
                                print(f"最適な番組: {best_program} を {brand} に割り当てます。")
                                print(f"累計到達数:{total_viewers}, 新規到達数: {best_new_viewers}, 到達率: {view_rate}")
                                print(f"残り予算: {allocated_brand_data.at[brand, seconds]}, 残り秒数: {temp_program_data.at[(best_program, best_date), 'P_seconds']}")
                                print(f"更新前サンプル数: {len(past_viewer)}")
                                print(f"追加サンプル数: {len(past_viewer)}")
                                print(f"更新後サンプル数: {len(brand_view_data[brand_column])}")
                                
                                # 新しい行のデータを作成
                                new_row = pd.DataFrame({
                                    'Program_code': [best_program],
                                    'Brand': [brand],
                                    'date': [best_date],
                                    'Allocated_seconds': [seconds],
                                    'Allocated_cost': [best_program_cost],
                                    'New_Viewers': [best_new_viewers]
                                })

                                # 既存のデータフレームに新しい行を追加する
                                allocated_program_data = pd.concat([allocated_program_data, new_row], ignore_index=True)

                                # 同じ番組、日付、ブランドの組み合わせを追跡するためにセットに追加
                                assigned_programs.add((best_program, best_date, brand))

                                # ブランドごとの新規視聴者数を累積
                                brand_new_viewers += best_new_viewers

                                # 割り当てが完了したのでフラグをTrueにし、次のブランドに移る
                                program_assigned = True
                                all_brands_done = False  # 割り当てが行われたら次のラウンドも行う

                                fin_data.at[(best_program, best_date), brand] = seconds
                                print("割り付け成功！")
                                break  # 1ラウンドで1番組のみ割り当てるので、次のブランドに移る
                            else:
                                print(f"{brand} の {seconds}秒枠で適切な番組が見つかりませんでした。次の秒数枠に移行します。")

                    # このブランドのラウンド終了時にリーチ率を計算
                    if program_assigned:
                        # view_trackにデータを追加
                        view_track = pd.concat([view_track, pd.DataFrame({
                            'Brand': [brand],
                            'Round': [round_number],
                            'New_Viewers': [brand_new_viewers],
                            'Total_Viewers': [total_viewers],
                            'Reach_Rate': [view_rate]
                        })], ignore_index=True)

                # 全ブランドで番組が割り当てられない場合はループを終了
                if all_brands_done:
                    print("すべてのブランドの割り当てが完了しました。")
                    break

                # ラウンドをカウントアップ
                round_number += 1

            # 最終割り当て結果を表示
            print("最終割り当て結果:")
            print(allocated_program_data)

            # リーチ率の追跡結果を表示
            print("リーチ率の追跡結果:")
            print(view_track)

            #====================================================
           
            st.write("割り付け終了")

            #セル5================================================
            # 最終的な視聴率データフレームを初期化
            fin_view_rate_list = pd.DataFrame(columns=['Brand', 'Total_Viewers', 'Reach_Rate'])

            # 各ブランドの視聴者数とリーチ率を計算
            for brand in brand_names:
                total_viewers = brand_view_data[brand].sum()  # ブランドの総視聴者数
                sample_num = len(brand_view_data[brand])
                view_rate = (total_viewers / sample_num) if sample_num > 0 else 0  # リーチ率の計算
                print(f"{brand} サンプル：{sample_num}リーチ{total_viewers}")

                # データを追加
                fin_view_rate_list = pd.concat([fin_view_rate_list, pd.DataFrame({
                    'Brand': [brand],
                    'Total_Viewers': [total_viewers],
                    'Reach_Rate': [view_rate]
                })], ignore_index=True)

            # 最終結果を表示
            print(fin_view_rate_list)
            #====================================================
           
            st.title("データ成形終了")

            #セル6================================================
            # Excel出力関数
            def create_excel_file():
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    fin_data.to_excel(writer, sheet_name='program×brand', index=True)
                    allocated_program_data.to_excel(writer, sheet_name='allocated_program_data', index=True)
                    view_track.to_excel(writer, sheet_name='view_track', index=True)
                    fin_view_rate_list.to_excel(writer, sheet_name='fin_view_rate_list', index=True)
                    allocated_brand_data.to_excel(writer, sheet_name='allocated_brand_cost', index=True)
                output.seek(0)
                return output
            
            excel_file = create_excel_file()
            
            # Streamlitアプリ本体
            st.title("Excelファイル出力")
            # ボタンでExcelファイルを生成・ダウンロード
            st.download_button(
                label="Excelファイルをダウンロード",
                data=excel_file,
                file_name="output.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        except Exception as e:
            st.error(f"ファイルを読み込む際にエラーが発生しました: {e}")

def tab_CausalImpact():
    show_card(
    """
    <h2>Causal Impact</h2>

    <h3>目的</h3>
    <ul>
        <li>広告出稿がKPIに与えた <b>因果的影響</b> を定量化する。</li>
        <li>出稿が無かった場合（カウンターファクト）のKPI推移を推定し、実績との差分＝<b>リフト効果</b> を把握する。</li>
    </ul>

    <h3>使用ケース</h3>
    <ul>
        <li><b>TVCM / キャンペーン効果検証</b>（出稿エリア vs 非出稿エリア）</li>
        <li><b>介入日以降が1になるフラグ</b> を用いた因果推定</li>
    </ul>

    <h3>inputデータ</h3>
    <ul>
        <li>必須列（ヘッダー名は任意）
            <ul>
                <li><b>日付列</b>（例: date / Date / 日付）</li>
                <li><b>出稿フラグ</b>（0=未出稿 / 1=出稿開始以降）</li>
                <li><b>出稿エリアKPI（treated）</b></li>
                <li><b>非出稿エリアKPI（control）</b></li>
            </ul>
        </li>
        <li>例：<code>date, flag, kpi_treated, kpi_control</code></li>
    </ul>

    <h3>アウトプット説明</h3>

    <h4>■ 1. Actual（実績値：treated）</h4>
    <ul>
        <li>出稿エリアの実測 KPI</li>
    </ul>

    <h4>■ 2. Counterfactual（反実仮想の予測値）</h4>
    <ul>
        <li>「もし出稿していなかったら」の推定値</li>
        <li>介入後は実績と乖離 → この差が効果</li>
    </ul>

    <h4>■ 3. Point Effect（瞬間効果）</h4>
    <ul>
        <li><b>実績 − カウンターファクト</b> の日次差分</li>
    </ul>

    <h4>■ 4. Cumulative Effect（累積効果）</h4>
    <ul>
        <li>介入開始以降のリフト累積</li>
        <li>「広告によって合計どれだけ押し上げられたか」</li>
    </ul>

    <h4>■ 5. Summary（サマリー）</h4>
    <ul>
        <li>平均効果（AV effect）</li>
        <li>合計効果（cumulative effect）</li>
        <li>相対効果 (%)</li>
        <li>統計的有意性（p-value）</li>
        <li>95% 予測区間（ベイズCI）</li>
    </ul>

    <h4>■ 6. Report（自然言語レポート）</h4>
    <ul>
        <li>そのままレポートに貼れる解釈文を自動生成</li>
    </ul>

    <h4>■ 7. Actual vs Counterfactual グラフ</h4>
    <ul>
        <li>青：実績</li>
        <li>オレンジ：反実仮想の推定曲線</li>
        <li>点線：介入日</li>
        <li>差分 = 因果効果（リフト）を可視化</li>
    </ul>

    <h4>■ 8. ダウンロード用 CSV</h4>
    <ul>
        <li>actual_treated（実績）</li>
        <li>counterfactual_pred（反実仮想）</li>
        <li>point_effect（瞬間効果）</li>
        <li>cumulative_effect（累積効果）</li>
    </ul>
    """
    )

                # ここで Python 側でダウンロードボタンを表示
    with open("app/CausalImpact.xlsx", "rb") as f:
        logistic_file = f.read()

    st.download_button(
        label="📥 入力シートをダウンロード",
        data=logistic_file,
        file_name="CausalImpact.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


    if not _CAUSALIMPACT_OK:
        st.error("causalimpact が未インストールです。先に環境へインストールしてください。")
        return

    up = st.file_uploader("CausalImpact用ファイル（CSV / XLSX）", type=["csv", "xlsx"], key="ci_file")
    if up is None:
        return

    # ------- 読み込み -------
    try:
        if up.name.lower().endswith(".xlsx"):
            df_raw = pd.read_excel(up)
        else:
            try:
                df_raw = pd.read_csv(up)
            except UnicodeDecodeError:
                up.seek(0); df_raw = pd.read_csv(up, encoding="shift-jis")
    except Exception as e:
        st.error(f"読み込みエラー: {e}")
        return

    if df_raw.shape[1] < 4:
        st.error("少なくとも 4 列（date, flag, treated, control）が必要です。")
        return

    st.write("アップロードプレビュー：")
    st.dataframe(df_raw.head())

    # ------- 日付列の自動検出 -------
    date_col = None
    for c in df_raw.columns:
        lc = str(c).lower()
        if "date" in lc or "日付" in lc:
            date_col = c; break
    if date_col is None:
        # 先頭列が日付っぽければ採用
        c0 = df_raw.columns[0]
        if pd.to_datetime(df_raw[c0], errors="coerce").notna().mean() > 0.8:
            date_col = c0

    if date_col is None:
        st.error("日付列を検出できませんでした。`date`/`Date`/`日付` 等の列を含めてください。")
        return

    # 列の並びを [date, flag, treated, control] に揃える（残りは無視）
    other_cols = [c for c in df_raw.columns if c != date_col]
    if len(other_cols) < 3:
        st.error("flag / treated / control の3列が不足しています。")
        return
    flag_col, treated_col, control_col = other_cols[:3]

    df = df_raw[[date_col, flag_col, treated_col, control_col]].copy()
    df.columns = ["date", "flag", "treated", "control"]

    # 型整形
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    try:
        df["flag"] = df["flag"].astype(int)
    except Exception:
        st.error("flag 列は 0/1 の数値にしてください。")
        return

    # ソート＆欠損処理
    df = df.sort_values("date").dropna(subset=["treated", "control", "flag"]).reset_index(drop=True)

    # ------- pre/post の自動決定（最初の 1 以降を post） -------
    ones = df.index[df["flag"] == 1].to_list()
    if not ones:
        st.error("flag=1 がありません。介入日以降を 1 にしてください。")
        return
    first_one_idx = ones[0]

    # 連続性チェック（推奨）
    if (df.loc[:first_one_idx-1, "flag"] != 0).any() or (df.loc[first_one_idx:, "flag"] != 1).any():
        st.warning("flag が『前半0→後半1の連続』になっていません。結果解釈に注意してください。")

       # ------- pre/post の自動決定 -------
    ones = df.index[df["flag"] == 1].to_list()
    zeros = df.index[df["flag"] == 0].to_list()
    if not ones:
        st.error("flag=1（介入以降）がありません。"); return
    if not zeros:
        st.error("flag=0（介入前）がありません。"); return
    first_one_idx = ones[0]
    if first_one_idx == 0:
        st.error("先頭行が flag=1 です。介入前（flag=0）を含めてください。"); return

    if (df.loc[:first_one_idx-1, "flag"] != 0).any() or (df.loc[first_one_idx:, "flag"] != 1).any():
        st.warning("flag が『前半0→後半1』の連続になっていません。結果の解釈に注意。")

    # ------- データ整形 -------
    ts = pd.DataFrame({
        "y":  df["treated"].astype(float).values,
        "x1": df["control"].astype(float).values
    }, index=df["date"])

    # コントロールの分散チェック（今回=0）
    if ts["x1"].std() == 0:
        add_noise = st.checkbox("コントロールが一定なので微小ノイズを加える（推奨）", value=True)
        if add_noise:
            import numpy as np
            ts["x1"] = ts["x1"] + 1e-6 * np.random.randn(len(ts))

    pre_period  = [ts.index[0], ts.index[first_one_idx-1]]
    post_period = [ts.index[first_one_idx], ts.index[-1]]

    # ------- 実行 -------
    try:
        ci = CausalImpact(ts, pre_period, post_period)
        if getattr(ci, "inferences", None) is None:
            ci.run()  # 明示実行
    except Exception as e:
        st.error(f"CausalImpact 実行エラー: {e}")
        st.stop()

    # 推定結果の存在チェック
    if getattr(ci, "inferences", None) is None or ci.inferences is None or ci.inferences.empty:
        st.error("推定結果が得られませんでした。pre/post 行数やデータ分散を見直してください。")
        st.write(f"pre 行数: {(ts.index <= pre_period[1]).sum()} / post 行数: {(ts.index >= post_period[0]).sum()}")
        st.stop()

    st.subheader("結果サマリー")
    st.text(ci.summary())
    st.subheader("レポート")
    st.text(ci.summary(output="report"))

    # inferences の中身を確認
    inf = ci.inferences.copy()
    st.write("inferences preview:", inf.head())

    # 予測値の列を探す
    pred_col = None
    for c in ["predicted", "mean", "preds"]:
        if c in inf.columns:
            pred_col = c
            break

    if pred_col is None:
        st.error(f"予測値の列が見つかりませんでした。利用可能な列: {inf.columns.tolist()}")
        st.stop()

    # 予測系列を全期間へ拡張（pre=実績, post=予測）
    pred_full = ts["y"].copy()
    pred_full.loc[post_period[0]:] = inf[pred_col]

    out = pd.DataFrame({
        "actual_treated": ts["y"],
        "counterfactual_pred": pred_full,
    })
    if "point_effect" in inf.columns:
        out["point_effect"] = inf["point_effect"].reindex(ts.index)
    if "cum_effect" in inf.columns:
        out["cum_effect"] = inf["cum_effect"].reindex(ts.index)

    st.subheader("推定テーブル（実績・予測・効果）")
    st.dataframe(out)

    st.download_button(
        "結果CSVをダウンロード",
        data=out.to_csv(index=True).encode("utf-8"),
        file_name="causal_impact_result.csv",
        mime="text/csv"
    )

    # グラフ（日付を横軸）
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(out.index, out["actual_treated"], label="Actual (treated)")
    ax.plot(out.index, out["counterfactual_pred"], label="Counterfactual (no-CM prediction)")
    ax.axvline(post_period[0], linestyle="--")
    ax.set_title("Actual vs Counterfactual (CausalImpact)")
    ax.set_xlabel("Date"); ax.set_ylabel("KPI"); ax.legend()
    st.pyplot(fig)

def tab_factor():
    show_card(
    """
    <h2>因子分析（Factor Analysis）</h2>

    <h3>目的</h3>
    <ul>
        <li>多数の質問項目やイメージ項目から <b>潜在因子（価値観・心理構造）</b> を抽出し、データの背後にある構造を理解する。</li>
    </ul>

    <h3>使用ケース</h3>
    <ul>
        <li>ブランドイメージ調査や NPS 調査の <b>心理構造</b> を把握したい。</li>
        <li>多数の項目を少数の因子へまとめ、<b>解釈しやすくしたい</b>。</li>
        <li>セグメンテーション前に、価値観・態度項目を <b>因子スコアに圧縮</b> したい。</li>
    </ul>

    <h3>inputデータ</h3>
    <ul>
        <li><b>数値列のみが対象</b></li>
        <li>1列目にID、2列目以降に「評価項目・イメージ項目」などを並べた形式</li>
        <li>CSV / Excel（A_入力シートがあれば優先）</li>
    </ul>

    <h3>アウトプット説明</h3>
    <ul>
        <li><b>因子負荷量</b>：どの項目がどの因子に強く関わるか（解釈の中心）</li>
        <li><b>因子スコア</b>：各サンプルの因子空間での位置</li>
        <li><b>固有値・寄与率</b>（必要に応じて追加可能）</li>
        <li><b>因子数は任意選択（1〜10）</b></li>
    </ul>
    """
    )
    # ここで Python 側でダウンロードボタンを表示
    with open("app/主成分OR因子分析.xlsx", "rb") as f:
        logistic_file = f.read()

    st.download_button(
        label="📥 入力シートをダウンロード",
        data=logistic_file,
        file_name="主成分OR因子分析.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # === ファイル読み込み ===
    up = st.file_uploader("CSV / XLSX をアップロード", type=["csv","xlsx"])
    if up is None:
        return

    try:
        if up.name.lower().endswith(".xlsx"):
            df = pd.read_excel(up)
        else:
            df = pd.read_csv(up)
    except:
        st.error("ファイル読み込みエラー")
        return

    st.write("データプレビュー：")
    st.dataframe(df.head())

    # === 1列目をID、2列目以降を説明変数として使用 ===
    ID = df.iloc[:, 0]              # 使わないが保持しておく
    X_raw = df.iloc[:, 1:].copy()

    # 数値列のみ使用
    X = X_raw.select_dtypes(include=[np.number])
    drop_cols = [c for c in X_raw.columns if c not in X.columns]
    if drop_cols:
        st.warning(f"非数値列を除外しました: {', '.join(drop_cols)}")

    if X.shape[1] == 0:
        st.error("因子分析には数値列が必要です。")
        return

    # 欠損値の処理
    na_opt = st.radio("欠損値の扱い", ["行ごとに削除（推奨）", "列平均で補完"], index=0, horizontal=True)
    if na_opt == "行ごとに削除（推奨）":
        X = X.dropna()
    else:
        X = X.fillna(X.mean())

    # === 因子数 ===
    n_factor = st.slider("抽出する因子数", 1, min(10, X.shape[1]), 2)

    # === 因子分析実行 ===
    from sklearn.decomposition import FactorAnalysis

    fa = FactorAnalysis(n_components=n_factor)
    F = fa.fit_transform(X)   # 因子スコア
    loadings = pd.DataFrame(
        fa.components_.T,
        index=X.columns,
        columns=[f"Factor{i+1}" for i in range(n_factor)]
    )

    # === 結果の表示 ===
    st.subheader("因子負荷量（Factor Loadings）")
    st.dataframe(loadings.style.format("{:.3f}"))

    score_df = pd.DataFrame(F, columns=[f"Factor{i+1}" for i in range(n_factor)])
    st.subheader("因子スコア（Factor Scores）")
    st.dataframe(score_df.head())

    # ダウンロード
    st.download_button("因子負荷量をCSVでダウンロード",
                    data=loadings.to_csv().encode("utf-8-sig"),
                    file_name="factor_loadings.csv",
                    mime="text/csv")

    st.download_button("因子スコアをCSVでダウンロード",
                    data=score_df.to_csv().encode("utf-8-sig"),
                    file_name="factor_scores.csv",
                    mime="text/csv")

def tab_ca():
    show_card(
    """
    <h2>コレスポンデンス分析（Correspondence Analysis）</h2>

    <h3>目的</h3>
    <ul>
        <li><b>カテゴリ × カテゴリの対応関係</b> を2次元マップとして可視化し、  
            どの属性がどのカテゴリに近いかを把握する。</li>
    </ul>

    <h3>使用ケース</h3>
    <ul>
        <li>ブランド × イメージワード の <b>ポジショニングマップ</b></li>
        <li>属性 × 購入理由、店舗 × 利用理由 などの関係整理</li>
        <li>クロス集計表を <b>視覚的に理解</b> したい場合</li>
    </ul>

    <h3>inputデータ</h3>
    <ul>
        <li><b>行：</b>ブランド / 属性</li>
        <li><b>列：</b>イメージワード / 購買理由</li>
        <li><b>クロス集計形式</b>（CSV / Excel）</li>
        <li>1列目は index（ブランド名など）</li>
    </ul>

    <h3>アウトプット説明</h3>
    <ul>
        <li><b>行プロット座標</b>：ブランド・属性の布置</li>
        <li><b>列プロット座標</b>：イメージワード・理由の布置</li>
        <li><b>CAマップ（対応分析プロット）</b>：行列間の距離を視覚化</li>
    </ul>
    """
    )
        # ここで Python 側でダウンロードボタンを表示
    with open("app/コレスポンデンス分析.xlsx", "rb") as f:
        logistic_file = f.read()

    st.download_button(
        label="📥 入力シートをダウンロード",
        data=logistic_file,
        file_name="コレスポンデンス分析.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # === ファイル読み込み ===
    up = st.file_uploader("クロス集計表（CSV / XLSX）", type=["csv","xlsx"])
    if up is None:
        return

    try:
        if up.name.lower().endswith(".xlsx"):
            df = pd.read_excel(up, index_col=0)
        else:
            df = pd.read_csv(up, index_col=0)
    except:
        st.error("ファイル読み込みエラー")
        return

    st.write("入力表：")
    st.dataframe(df)

    # === CA 実行 ===
    try:
        import prince
    except:
        st.error("ライブラリ 'prince' がありません。`pip install prince` を実行してください")
        return

    ca = prince.CA(n_components=2)
    ca = ca.fit(df)

    row_coords = ca.row_coordinates(df)
    col_coords = ca.column_coordinates(df)

    st.subheader("行（Row）座標")
    st.dataframe(row_coords)

    st.subheader("列（Column）座標")
    st.dataframe(col_coords)

    # === プロット ===
    fig, ax = plt.subplots(figsize=(7,7))

    # 行
    ax.scatter(row_coords[0], row_coords[1], label="Rows")
    for i, txt in enumerate(row_coords.index):
        ax.text(row_coords.iloc[i, 0], row_coords.iloc[i, 1], txt)

    # 列
    ax.scatter(col_coords[0], col_coords[1], marker="x", label="Columns")
    for i, txt in enumerate(col_coords.index):
        ax.text(col_coords.iloc[i, 0], col_coords.iloc[i, 1], txt)

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_title("Correspondence Analysis Map")
    ax.legend()

    st.pyplot(fig)


def tab_curve():
    latex_png = latex_to_png_base64(
        r"y = \frac{K}{1 + a\left(\frac{x}{10^{d_x}}\right)^b}\,10^{d_y}"
    )

    show_card(f"""
    <h2>Curve数式予測</h2>

    <h3>モデルの数式</h3>
    <div style="text-align:center;">
        <img src="data:image/png;base64,{latex_png}" style="width:80%; max-width:600px;">
    </div>

    <ul>
        <li>「d_x」「d_y」は桁調整用のパラメータ</li>
        <li>a, b, K, d_x, d_y を上記式に代入してモデル完成</li>
        <li><b>R²（決定係数）</b>：1に近いほどモデル精度が高い</li>
    </ul>
    """)

            # ここで Python 側でダウンロードボタンを表示
    with open("app/Curve数式予測.xlsx", "rb") as f:
        logistic_file = f.read()

    st.download_button(
        label="📥 入力シートをダウンロード",
        data=logistic_file,
        file_name="Curve数式予測.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


    uploaded_file = st.file_uploader("Curve数式予測用inputファイルをアップロードしてください", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            st.write("アップロードされたファイルの中身:")
            if uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                bytes_data = uploaded_file.read()
                xl = pd.ExcelFile(BytesIO(bytes_data))
                # シート名が "A_入力" の場合のみ読み込む
                if "A_入力" in xl.sheet_names:
                    df = pd.read_excel(xl, sheet_name="A_入力")
                    st.write(df)
                else:
                    st.warning("指定されたシートが見つかりませんでした。")
            else:
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                df = pd.read_csv(stringio, encoding="shift-jis")
                st.write(df)

            num = int(df.shape[1] / 2)
            for i in range(num):
                df_temp = df.iloc[:, [i * 2, i * 2 + 1]]
                df_temp.dropna()

            st.write(df)  # 一旦読み込んだデータのNaNを削除して表示

            name_list = []
            a_list = []
            b_list = []
            K_list = []
            R_list = []
            d_x_list = []
            d_y_list = []

            max_fev = 100000000
            df2 = pd.DataFrame()

            for i in range(num):
                df_temp = df.iloc[:, [i * 2, i * 2 + 1]]
                df_temp = df_temp.dropna()

                x_observed = df_temp.iloc[:, 0]
                y = df_temp.iloc[:, 1]

                # 説明変数と目的変数の桁数を計算する
                max_num = max(x_observed)
                s_x = str(max_num)
                if '.' in s_x:
                    s_x_i, s_x_d = s_x.split('.')
                else:
                    s_x_i = s_x
                    s_x_d = '0'
                d_x = float(len(s_x_i))

                max_num = max(y)
                s_y = str(max_num)
                s_y_i, s_y_d = s_y.split('.')
                d_y = float(len(s_y_i))

                x_observed = x_observed / 10 ** d_x
                y = y / 10 ** d_y
                max_num = max(y) * 10

                bounds = ((0, -5, 0), (100, 0, max_num))
                # bounds = ((0,-3,0),(10000000,0,50000))

                name = df.columns.values[i * 2]
                param, pcov = curve_fit(func_fit, x_observed, y, bounds=bounds, maxfev=max_fev)
                fit_y = func_fit(x_observed, param[0], param[1], param[2])
                df2[name + "_x"] = x_observed * 10 ** d_x
                df2[name + "_y"] = y * 10 ** d_y
                df2[name + "_fit"] = fit_y * 10 ** d_y
                R2 = r2_score(fit_y, y)

                name_list.append(name)
                a_list.append(param[0])
                b_list.append(param[1])
                K_list.append(param[2])
                d_x_list.append(d_x)
                d_y_list.append(d_y)
                R_list.append(R2)

            df_param = pd.DataFrame({"name": name_list, "a": a_list, "b": b_list, "max_value": K_list,
                                     "d_x": d_x_list, "d_y": d_y_list, "R2": R_list})
            st.write(df_param)  # 一旦読み込んだデータのNaNを削除したよ
            download(df_param)

            # プルダウンによるグラフ表示
            selected_name = st.selectbox("グラフ化するデータを選択してください", df_param['name'].unique())
            if selected_name:
                plt.figure(figsize=(10, 6))
                plt.scatter(df2[selected_name + "_x"], df2[selected_name + "_y"], label="Data")
                plt.plot(df2[selected_name + "_x"], df2[selected_name + "_fit"], 'r-', label="Fit")
                plt.xlabel("X")
                plt.ylabel("Y")
                plt.title(f"Fit for {selected_name}")
                plt.legend()
                st.pyplot(plt)

        except Exception as e:
            st.error(f"ファイルを読み込む際にエラーが発生しました: {e}")


#tab_TIME用の初期化、実行に関わる関数==========================
def initialize_session_state():
    """セッションステートの初期化"""
    defaults = {
        "current_step": "モード選択",  # 初期ステップ
        "uploaded_config_file": None,  # アップロードされた条件ファイル
        "uploaded_view_file": None, #アップロードされた視聴データファイル
        "processed_data": None,  # 処理されたデータ
        "allocated_cost_data": None,  # 残コストデータ
        "allocated_program_data": None,  # 割り付けログ
        "mode": "",  # モード選択
        "step_status": {
            "モード選択": True,  # 最初のステップをTrue
            "条件ファイルアップロード": False,
            "Viewファイルアップロード": False,
            "実行": False,
        },
        # ログイン情報（例: ユーザー情報）はここで保持
        "user_info": st.session_state.get("user_info", None),  # ログイン情報を保持
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def reset_app():
    """特定のセッションステート項目のみをリセット"""
    keys_to_reset = [
        "current_step", 
        "uploaded_config_file", 
        "uploaded_view_file",
        "processed_data", 
        "allocated_cost_data", 
        "allocated_program_data", 
        "mode", 
        "step_status",
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    initialize_session_state()  # 再初期化

def display_mode_selection():
    """モード選択画面"""
    if st.session_state["step_status"]["モード選択"]:
        st.header("モード選択")
        options = ["", "reach_cost", "reach", "target_cost"]  # 空欄を追加
        st.session_state["mode"] = st.selectbox("モードを選択してください", options)
        
        if st.session_state["mode"] == "":
            st.warning("モードを選択してください")
        else:
            st.write(f"選択されたモード: {st.session_state['mode']}")
            if st.button("条件ファイルアップロードへ", key="to_upload"):
                st.session_state["current_step"] = "条件ファイルアップロード"
                st.session_state["step_status"]["条件ファイルアップロード"] = True

def display_config_file_upload():
    """ファイルアップロード画面"""
    if st.session_state["step_status"]["条件ファイルアップロード"]:
        st.header("条件ファイルアップロード")
        if st.session_state["uploaded_config_file"] is None:
            uploaded_config_file = st.file_uploader("条件Excelファイルをアップロードしてください", type=["xlsx"])
            if uploaded_config_file is not None:
                st.session_state["uploaded_config_file"] = uploaded_config_file
        else:
            st.write("既にアップロードされた条件ファイルがあります。")
            st.write(f"アップロード済み条件ファイル: {st.session_state['uploaded_config_file'].name}")

        if st.session_state["uploaded_config_file"] is not None:
            if st.button("Viewファイルアップロードへ", key="to_execute_config"):
                st.session_state["current_step"] = "Viewファイルアップロード"
                st.session_state["step_status"]["Viewファイルアップロード"] = True

def display_view_file_upload():
    """ファイルアップロード画面"""
    if st.session_state["step_status"]["Viewファイルアップロード"]:
        st.header("Viewファイルアップロード")
        if st.session_state["uploaded_view_file"] is None:
            uploaded_view_file = st.file_uploader("CSV Viewファイルをアップロードしてください", type=["csv"])
            if uploaded_view_file is not None:
                st.session_state["uploaded_view_file"] = uploaded_view_file
        else:
            st.write("既にアップロードされたViewファイルがあります。")
            st.write(f"アップロード済みViewファイル: {st.session_state['uploaded_view_file'].name}")

        if st.session_state["uploaded_view_file"] is not None:
            if st.button("次へ", key="to_execute_view"):
                st.session_state["current_step"] = "実行"
                st.session_state["step_status"]["実行"] = True


def display_execution():
    """実行画面"""
    if st.session_state["step_status"]["実行"]:
        st.header("最適化の実行")
        st.write(f"選択されたモード: {st.session_state['mode']}")

        # config_fileとview_fileがアップロードされている場合のみ処理を実行
        if st.session_state["processed_data"] is None and st.session_state["uploaded_config_file"] is not None and st.session_state["uploaded_view_file"] is not None:
            st.write("処理を実行しています...")

            # configファイル（Excel）を読み込む
            bytes_data_config = st.session_state["uploaded_config_file"].read()  # 正しく読み込む
            config_data = pd.read_excel(BytesIO(bytes_data_config), sheet_name=None)

            # 各シートを取得
            limit_data = config_data['A_Limit'].set_index(['Program_code', 'date'])
            brand_data = config_data['B_Brand'].set_index('Brand')
            target_data = config_data['D_Target'].set_index('Brand')

            # viewファイル（CSV）からデータを読み込む
            bytes_data_view = st.session_state["uploaded_view_file"].read()  # 正しく読み込む
            view_data = pd.read_csv(BytesIO(bytes_data_view))

            # 必要な処理を行う（例: インデックスを設定）
            view_data = view_data.set_index('Sample')

            # データを表示
            st.write("A_Limit シートのデータ")
            st.dataframe(limit_data.head())
            st.write("B_Brand シートのデータ")
            st.dataframe(brand_data.head())
            st.write("D_Target シートのデータ")
            st.dataframe(target_data.head())
            st.write("C_View シートのデータ")
            st.dataframe(view_data.head())

            # 「無し」という値を空白に置き換え、必須番組データと除外データを作成
            exc_data = limit_data.copy()
            must_data = limit_data.copy()

            values_to_replace_exc = [15, 30, 60, 120, 240]
            values_to_replace_must = ["無し"]
            exc_data.replace(values_to_replace_exc, '', inplace=True)  # 除外の0-1データ
            must_data.replace(values_to_replace_must, '', inplace=True)  # 必須番組の割り振り秒数データ

            # ブランド名のリストを取得
            brand_names = brand_data.index.tolist()
            #ブランドの割り付け情報が入ってる
            temp_brand_data = limit_data.copy()
            temp_brand_data = temp_brand_data.drop(columns=[col for col in limit_data.columns if 'Cost/30' in col])
            temp_brand_data = temp_brand_data.drop(columns=[col for col in limit_data.columns if 'P_seconds' in col])
            temp_brand_data = temp_brand_data.drop(columns=[col for col in limit_data.columns if 'Program' in col])

            #番組のコストと秒数
            temp_program_data = limit_data[['Cost/30', 'P_seconds']]

            # 各ブランドの当初の予算を保存
            allocated_brand_data = brand_data.copy()  # 割り付けに使うブランドごとの予算
            initial_brand_budget = allocated_brand_data.copy()  # 割り付け前の初期予算
            used_brand_budget = pd.DataFrame(0, index=brand_names, columns=[120, 60, 30, 15])  # 割り当てられた予算のデータフレーム

            # 視聴データを保持する辞書（ターゲット層に基づく長さを設定）
            brand_view_data = {}
            # target_dataがDataFrameであることを仮定
            brand_target = target_data

            for brand_column in brand_names:
                # ブランドのターゲット年齢範囲と性別を取得
                target_age_range = brand_target.loc[brand_column, ['Low', 'High']]  # 年齢範囲
                target_gender = brand_target.loc[brand_column, 'Gender']  # 性別

                # ターゲット層に一致する視聴データを絞り込み
                if target_gender == 'MF':
                    # 「MF」ターゲットの場合、性別に関係なくすべての視聴者を選択
                    filtered_view_data = view_data[
                        (view_data['Age'] >= target_age_range[0]) & 
                        (view_data['Age'] <= target_age_range[1])
                    ]
                else:
                    # 指定された性別と年齢範囲に基づいて絞り込み
                    filtered_view_data = view_data[
                        (view_data['Age'] >= target_age_range[0]) & 
                        (view_data['Age'] <= target_age_range[1]) & 
                        (view_data['Gender'] == target_gender)
                    ]
                
                # ターゲット層に一致する視聴データのインデックス長さを取得
                filtered_index = filtered_view_data.index
                print(len(filtered_index))
                # ターゲット層に基づいて視聴データを初期化
                brand_view_data[brand_column] = pd.Series([False] * len(filtered_index), index=filtered_index)


            # 割り当て結果を記録するデータフレーム
            allocated_program_data = pd.DataFrame(columns=['Program_code', 'Brand', 'Allocated_seconds', 'Allocated_cost', 'New_Viewers','Total_Viewers','Potential','Reach_Rate','Round'])

            #アロケのした後のフレーム
            fin_data = limit_data.copy()
            #====================================================

            st.write("設定終了")

            #セル3================================================
            # brand_targetがDataFrameで、'Brand'がインデックスとして設定されている場合
            # 空のプレースホルダを作成（このエリアがログ表示エリアになります）
            log_config_placeholder = st.empty()
            # 初期のログ内容
            log_config = ""
            for brand_column in temp_brand_data.columns:
                print(f"\n--- {brand_column} の処理 ---")

                for index, value in temp_brand_data[brand_column].items():
                    if value == "無し" or pd.isna(value):
                        continue  # "無し"や NaN の場合はスキップ

                    if value in [15, 30, 60, 120, 240]:  # valueが秒数として有効か確認
                        program_code, date = index  # 複合キーから program_code と date を取り出す
                        
                        print(program_code)

                        # 番組のコストと秒数を取得
                        program_cost = temp_program_data.loc[(program_code, date), 'Cost/30']
                        program_seconds = temp_program_data.loc[(program_code, date), 'P_seconds']

                        # ブランドの秒数を減らす
                        brand_seconds = value  # temp_brand_dataの値がそのまま秒数と仮定
                        program_seconds_remaining = program_seconds - brand_seconds  # 残り秒数を計算

                        # 番組の秒数を更新する（必要ならtemp_program_dataに反映）
                        temp_program_data.loc[(program_code, date), 'P_seconds'] = program_seconds_remaining

                        # ブランド名と今回の秒数に基づいてコストを取得
                        brand_cost = allocated_brand_data.loc[brand_column, value]  # ブランド名と秒数が一致するコストを取得
                        
                        # ブランドの秒数とコストを取得
                        brand_seconds = value  # temp_brand_dataの値がそのまま秒数と仮定
                        allocated_cost = program_cost * (brand_seconds / 30)  # コストを計算

                        # 1. インデックスが一致しているか確認
                        print(allocated_brand_data.index)  # インデックスを確認
                        print(brand_column, value)  # 使用しているインデックスも確認

                        # 2. データ型を確認し、必要なら変換
                        if not isinstance(allocated_cost, (int, float)):
                            allocated_cost = float(allocated_cost)
                        
                        # ブランドの予算を減らす
                        allocated_brand_data.at[brand_column, value] -= allocated_cost
                        new_cost = allocated_brand_data.loc[brand_column, value]

                        # 試聴データをターゲット層（年齢・性別）に基づいて絞り込み
                        target_age_range = brand_target.loc[brand_column, ['Low', 'High']]  # 年齢範囲を取得
                        target_gender = brand_target.loc[brand_column,'Gender']  # 例: 'Female'

                        if target_gender == 'MF':
                            # 「MF」ターゲットの場合、性別に関係なくすべての視聴者を選択
                            filtered_view_data = view_data[
                                (view_data['Age'] >= target_age_range[0]) & 
                                (view_data['Age'] <= target_age_range[1])
                            ]
                        else:
                            # 指定された性別と年齢範囲に基づいて絞り込み
                            filtered_view_data = view_data[
                                (view_data['Age'] >= target_age_range[0]) & 
                                (view_data['Age'] <= target_age_range[1]) & 
                                (view_data['Gender'] == target_gender)
                            ]

                        # 視聴データを取得（重複を除いた新しい視聴者のみ）
                        past_viewer = brand_view_data[brand_column].copy()
                        brand_view_data[brand_column] |= filtered_view_data[program_code]
                        viewer_add = sum(brand_view_data[brand_column]) - sum(past_viewer)
                        Reach_rate = brand_view_data[brand_column] / len(brand_view_data[brand_column])

                        log_config += f"====================================================================================="
                        log_config += f"{brand_column}の{value}秒を{program_code}:{date}に{program_cost}円で割り付け\n"
                        log_config += f"{brand_column}の{value}秒の元予算{brand_cost}から残り予算{new_cost}へ\n"
                        log_config += f"{brand_column}のリーチ数は{sum(past_viewer)}から{sum(brand_view_data[brand_column])}へ\n"

                        # ログ表示を更新
                        log_config_placeholder.text_area("必須番組処理ログ", log_config, height=300)

                        print(f"Brand: {brand_column}, 秒数: {value}")
                        print(f"対応するコスト: {brand_cost}")
                        print(f"Program: {program_code}, Date: {date}")
                        print(f"Program Cost/30: {program_cost}, Program Seconds: {program_seconds}")
                        print(f"Brand Allocated Seconds: {brand_seconds}, Brand Allocated Cost: {allocated_cost}")
                        print(f"新しいブランド予算: {new_cost}")
                        print(f"残り番組秒数: {program_seconds_remaining}")
                        print("-" * 50)
                        print(f"元の視聴データ: {sum(past_viewer)}")
                        print(f"新規視聴データ: {sum(brand_view_data[brand_column])}")
                        print(f"新規獲得視聴者: {viewer_add}")
                        print(f"サンプル数: {len(brand_view_data[brand_column])}")


                        # 新しい行のデータを作成
                        new_row = pd.DataFrame({
                            'Program_code': [program_code],
                            'Brand': [brand_column],
                            'Allocated_seconds': [brand_seconds],
                            'Allocated_cost': [allocated_cost],
                            'New_Viewers': [viewer_add],
                            'Total_Viewers': [brand_view_data[brand_column]],
                            'Potential': [len(brand_view_data[brand_column])],
                            'Reach_Rate': [Reach_rate],
                            'Round':[None]
                        })

                        #'Program_code', 'Brand', 'Allocated_seconds', 'Allocated_cost', 'New_Viewers','Total_Viewers','Potential','Reach_Rate','Round'])
                        
                        # 既存のデータフレームに新しい行を追加する
                        allocated_program_data = pd.concat([allocated_program_data, new_row], ignore_index=True)
            #====================================================
        
            st.write("必須終了")

            #セル4================================================
            pd.set_option('mode.chained_assignment', None)  # チェーンされた代入の警告を無視
            import warnings
            warnings.simplefilter(action='ignore', category=FutureWarning)


            # view_track DataFrameの初期化
            view_track = pd.DataFrame(columns=['Brand', 'Round', 'New_Viewers', 'Total_Viewers', 'Reach_Rate'])

            # 初期化
            seconds_priorities = sorted(brand_data.columns, reverse=True)
            round_number = 0  # ラウンドカウンタ
            all_brands_done = False  # 全てのブランドの割り付けが終わったかを確認するフラグ
            allocated_program_data = pd.DataFrame(columns=['Program_code', 'Brand', 'date', 'Allocated_seconds', 'Allocated_cost', 'New_Viewers'])

            # 割り当て済みの番組コードと日付の組み合わせを保存するためのセット
            assigned_programs = set()

            log_opt_placeholder = st.empty()
            # 初期のログ内容
            log_opt = ""
            # 割り付け可能なブランドがある限り繰り返すループ
            while not all_brands_done:
                print(f"\n--- ラウンド {round_number} ---")
                
                all_brands_done = True  # すべてのブランドが完了したか確認するために一旦Trueにする

                # 各ブランドごとに割り当てを行う
                for brand in brand_names:
                    program_assigned = False  # フラグを初期化
                    brand_new_viewers = 0  # このラウンドでの新規視聴者数を初期化

                    # ターゲット層（年齢・性別）に基づいて視聴データを絞り込み
                    target_age_range = brand_target.loc[brand, ['Low', 'High']]  # 年齢範囲
                    target_gender = brand_target.loc[brand, 'Gender']  # 性別

                    # ターゲット層に一致する視聴データを絞り込む
                    if target_gender == 'MF':
                        # 「MF」ターゲットの場合、性別に関係なくすべての視聴者を選択
                        filtered_view_data = view_data[
                            (view_data['Age'] >= target_age_range[0]) & 
                            (view_data['Age'] <= target_age_range[1])
                        ]
                    else:
                        # 指定された性別と年齢範囲に基づいて絞り込み
                        filtered_view_data = view_data[
                            (view_data['Age'] >= target_age_range[0]) & 
                            (view_data['Age'] <= target_age_range[1]) & 
                            (view_data['Gender'] == target_gender)
                        ]

                    # 優先する秒数の順にチェック
                    for seconds in seconds_priorities:
                        if program_assigned:  # 番組が割り当てられた場合は次のブランドに移行
                            break

                        brand_rest_cost = allocated_brand_data.at[brand, seconds]
                        program_cost_arr = temp_program_data['Cost/30'] * (seconds / 30)
                        program_seconds_arr = temp_program_data['P_seconds']

                        if (program_cost_arr > brand_rest_cost).all():
                            print(f"{brand}の{seconds}は予算上限に達しています。")
                            continue

                        if (program_seconds_arr < seconds).all():
                            print(f"{brand}の{seconds}に割り当てられる番組秒数がありません。")
                            continue

                        # もし予算が残っていれば番組を割り当てる
                        if allocated_brand_data.at[brand, seconds] > 0:
                            best_program = None
                            best_new_viewers = 0
                            best_allocated_seconds = 0
                            best_date = None

                            temp_df = pd.DataFrame()
                            past_viewer = brand_view_data[brand].copy()  # ここでコピーを取る

                            # 最適な番組を選ぶための処理
                            for index, value in temp_brand_data[brand].items():
                                program_code, date = index

                                # 既に割り当てられた番組・日付の組み合わせをチェック
                                if (program_code, date, brand) in assigned_programs:
                                    print(f"{brand} に対して、プログラム {program_code}, 日付 {date} は既に割り当て済みです。")
                                    continue

                                # "無し" または視聴データがNaNでない場合はスキップ
                                if value == "無し" or not pd.isna(value):
                                    continue

                                # 番組のコストと秒数を取得
                                program_cost = temp_program_data.at[(program_code, date), 'Cost/30'] * (seconds / 30)
                                program_seconds = temp_program_data.at[(program_code, date), 'P_seconds']

                                # 割り当て可能な秒数を確認
                                if program_seconds < seconds:
                                    continue

                                # コスト確認
                                if allocated_brand_data.at[brand, seconds] < program_cost:
                                    continue

                                # 過去の視聴者数を保持し、新たな視聴者数を計算
                                if program_code in filtered_view_data.columns:
                                    new_viewers = filtered_view_data[program_code]
                                    target_cost = new_viewers.sum() / program_cost

                                    # 既存の視聴者データと結合（視聴した人を1とする場合）
                                    temp_brand_view_data = past_viewer | new_viewers
                                    viewer_add = temp_brand_view_data.sum() - past_viewer.sum()
                                    viewer_add_per_cost = viewer_add / program_cost
                                else:
                                    viewer_add = 0

                                #if viewer_add <= 0:
                                    #continue

                                #新しいviewrが増えないとtempdfに追加されてないから増えないんだ

                                # 番組を追加
                                temp_data = pd.DataFrame({
                                    'program_code': [program_code],
                                    'date': [date],
                                    'viewer_add': [viewer_add],
                                    'viewer_add_per_cost': [viewer_add_per_cost],
                                    'target_cost': [target_cost]
                                })

                                temp_df = pd.concat([temp_df, temp_data], ignore_index=True)

                            mode = str(st.session_state["mode"])
                            print(mode)

                            # temp_dfから最適な番組を選ぶ
                            if not temp_df.empty:
                                print("えへ")
                                if mode == "reach":
                                    print("リーチになってる")
                                    # リーチが最大のものを選ぶ
                                    best_row = temp_df.loc[temp_df["viewer_add"].idxmax()]
                                    if best_row["viewer_add"] > 0:  # 新規視聴者数が正の場合のみ割り付け
                                        best_program = best_row["program_code"]
                                        best_date = best_row["date"]
                                        best_new_viewers = best_row["viewer_add"]

                                elif mode == "reach_cost":
                                    print("best")
                                    # リーチ増分に対するコスト効率が最も高いものを選ぶ
                                    best_row = temp_df.loc[temp_df["viewer_add_per_cost"].idxmin()]
                                    if best_row["viewer_add"] > 0:  # 新規視聴者数が正の場合のみ割り付け
                                        best_program = best_row["program_code"]
                                        best_date = best_row["date"]
                                        best_new_viewers = best_row["viewer_add"]

                                elif mode == "target_cost":
                                    print("ターゲットコストを選択できてる")
                                    # target_costが最も小さいものを選ぶ（必ず割り付け）
                                    best_row = temp_df.loc[temp_df["target_cost"].idxmin()]
                                    best_program = best_row["program_code"]
                                    best_date = best_row["date"]
                                    best_new_viewers = best_row["viewer_add"]
                                    print(best_program)

                            print("ここじゃない")

                            # 最適な番組が見つかった場合の処理
                            if best_program and best_date is not None:
                                # 割り当てた番組の処理（コストの減算や視聴者データの更新など）
                                best_program_cost = temp_program_data.at[(best_program, best_date), 'Cost/30'] * (seconds / 30)
                                old_cost = allocated_brand_data.at[brand, seconds]
                                allocated_brand_data.at[brand, seconds] -= best_program_cost
                                temp_program_data.at[(best_program, best_date), 'P_seconds'] -= seconds
                                new_viewers = filtered_view_data[best_program]  # 視聴データの更新
                                brand_view_data[brand] = past_viewer | new_viewers  # 既存の視聴者データと結合（視聴した人を1とする場合）
                                total_viewers = brand_view_data[brand].sum()
                                sample_num = len(brand_view_data[brand_column])
                                view_rate = total_viewers / sample_num
                                
                                # 割り当て結果を表示
                                print(f"最適な番組: {best_program} を {brand} に割り当てます。")
                                print(f"累計到達数:{total_viewers}, 新規到達数: {best_new_viewers}, 到達率: {view_rate}")
                                print(f"残り予算: {allocated_brand_data.at[brand, seconds]}, 残り秒数: {temp_program_data.at[(best_program, best_date), 'P_seconds']}")
                                print(f"更新前サンプル数: {len(past_viewer)}")
                                print(f"追加サンプル数: {len(past_viewer)}")
                                print(f"更新後サンプル数: {len(brand_view_data[brand_column])}")

                                log_opt += f"================================================================================"
                                log_opt += f"{brand}の{seconds}秒を{best_program}:{best_date}に{best_program_cost}円で割り付け\n"
                                log_opt += f"{brand}の{seconds}秒の元予算{old_cost}から残り予算{allocated_brand_data.at[brand, seconds]}へ\n"
                                log_opt += f"{brand}のリーチ数は{sum(past_viewer)}から{total_viewers}へ\n"
                                # ログ表示を更新
                                log_opt_placeholder.text_area("最適番組処理ログ", log_opt, height=300)
                                
                                # 新しい行のデータを作成
                                new_row = pd.DataFrame({
                                    'Program_code': [best_program],
                                    'Brand': [brand],
                                    'date': [best_date],
                                    'Allocated_seconds': [seconds],
                                    'Allocated_cost': [best_program_cost],
                                    'New_Viewers': [best_new_viewers],
                                    'Total_Viewers': [total_viewers],
                                    'Potential': [sample_num],
                                    'Reach_Rate': [view_rate],
                                    'Round':[round_number]
                                })

                                # 既存のデータフレームに新しい行を追加する
                                allocated_program_data = pd.concat([allocated_program_data, new_row], ignore_index=True)

                                # 同じ番組、日付、ブランドの組み合わせを追跡するためにセットに追加
                                assigned_programs.add((best_program, best_date, brand))

                                # ブランドごとの新規視聴者数を累積
                                brand_new_viewers += best_new_viewers

                                # 割り当てが完了したのでフラグをTrueにし、次のブランドに移る
                                program_assigned = True
                                all_brands_done = False  # 割り当てが行われたら次のラウンドも行う

                                fin_data.at[(best_program, best_date), brand] = seconds
                                print("割り付け成功！")
                                break  # 1ラウンドで1番組のみ割り当てるので、次のブランドに移る
                            else:
                                print(f"{brand} の {seconds}秒枠で適切な番組が見つかりませんでした。次の秒数枠に移行します。")

                    # このブランドのラウンド終了時にリーチ率を計算
                    if program_assigned:
                        # view_trackにデータを追加
                        view_track = pd.concat([view_track, pd.DataFrame({
                            'Brand': [brand],
                            'Round': [round_number],
                            'New_Viewers': [brand_new_viewers],
                            'Total_Viewers': [total_viewers],
                            'Reach_Rate': [view_rate]
                        })], ignore_index=True)

                # 全ブランドで番組が割り当てられない場合はループを終了
                if all_brands_done:
                    print("すべてのブランドの割り当てが完了しました。")
                    break

                # ラウンドをカウントアップ
                round_number += 1

            # 最終割り当て結果を表示
            print("最終割り当て結果:")
            print(allocated_program_data)

            # リーチ率の追跡結果を表示
            print("リーチ率の追跡結果:")
            print(view_track)

            #====================================================
        
            st.write("割り付け終了")

            #セル5================================================
            # 最終的な視聴率データフレームを初期化
            fin_view_rate_list = pd.DataFrame(columns=['Brand', 'Total_Viewers', 'Reach_Rate'])

            # 各ブランドの視聴者数とリーチ率を計算
            for brand in brand_names:
                total_viewers = brand_view_data[brand].sum()  # ブランドの総視聴者数
                sample_num = len(brand_view_data[brand])
                view_rate = (total_viewers / sample_num) if sample_num > 0 else 0  # リーチ率の計算
                print(f"{brand} サンプル：{sample_num}リーチ{total_viewers}")

                # データを追加
                fin_view_rate_list = pd.concat([fin_view_rate_list, pd.DataFrame({
                    'Brand': [brand],
                    'Total_Viewers': [total_viewers],
                    'Reach_Rate': [view_rate]
                })], ignore_index=True)

            # 最終結果を表示
            st.write(fin_view_rate_list)
            #====================================================

            st.session_state["processed_data"] = fin_data #素材を割り付けた状態のデータ
            st.session_state["allocated_cost_data"] = allocated_brand_data #ブランドの残コストデータ
            st.session_state["allocated_program_data"] = allocated_program_data #割り付けのログ

        # 結果を表示
        if st.session_state["processed_data"] is not None:
            st.write("割り付け結果:")
            st.write(st.session_state["processed_data"])
            st.write("ブランド残予算:")
            st.write(st.session_state["allocated_cost_data"])
            st.write("割り付けトラッキングデータ:")
            st.write(st.session_state["allocated_program_data"])

def tab_time():
    """アプリケーションのメイン関数"""
    initialize_session_state()

    # リセットボタン
    if st.button("リセット", key="reset"):
        reset_app()

    # 各ステップの画面を表示（過去のステップも残す）
    display_mode_selection()
    display_config_file_upload()
    display_view_file_upload()
    display_execution()



#Streamlitを実行する関数
def main():
    if login():
        tabs = st.sidebar.radio(
            "メニュー",
            [
                "主成分分析",
                "因子分析",
                "コレスポンデンス分析",
                "共分散構造分析（SEM）",
                "Logistic回帰",
                "順序Logistic回帰",
                "重回帰（自動選択）",
                "MMM（軽量版）",
                "STL分解",  
                "TIME最適化",
                "Causal Impact",
                "Curve数式予測",
            ]
        )

        # ログアウトボタン
        if st.button("ログアウト"):
            st.session_state["logged_in"] = False
            st.session_state["user"] = None

            # rerun
            st.rerun()

        if tabs == "主成分分析":
            tab_PCA()
        elif tabs == "因子分析":
            tab_factor()
        elif tabs == "コレスポンデンス分析":
            tab_ca()
        elif tabs == "共分散構造分析（SEM）":
            tab_SEM()
        elif tabs == "Logistic回帰":
            tab_Logistic()
        elif tabs == "順序Logistic回帰":
            tab_LogisticNum()
        elif tabs == "重回帰（自動選択）":
            tab_MultipleRegression()   
        elif tabs == "MMM（軽量版）":
            tab_MMM()
        elif tabs == "STL分解":
            tab_STL()
        elif tabs == "TIME最適化":
            tab_time()
        elif tabs == "Causal Impact":
            tab_CausalImpact()
        elif tabs == "Curve数式予測":
            tab_curve()


#実行コード
if __name__ == "__main__":
    main()