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

def tab1():
    st.write("主成分分析（PCA）")

    # === セクション：説明 ===
    st.subheader("目的")
    st.markdown("""
    - 多数の説明変数に潜む共通因子を抽出し、次元圧縮して全体構造を把握する。
    """)

    st.subheader("使用ケース")
    st.markdown("""
    - **多変量の要約**：媒体接触や属性が多いときに、少数の指標（主成分）へ要約。  
    - **可視化**：2次元に圧縮してクラスタ傾向・外れ値を把握。  
    - **前処理**：回帰やクラスタリング前に多重共線性を緩和。
    """)

    st.subheader("inputデータ")
    st.markdown("""
    - 1列目：**目的変数（y）**  
    - 2列目以降：**説明変数（X）**（数値列）  
    ※Excel/CSV対応。Excelは **A_入力** シートがあれば優先、無ければ先頭シートを読み込みます。
    """)

    st.subheader("アウトプット説明")
    st.markdown("""
    - **固有値・寄与率・累積寄与率**：各主成分がどれだけ分散を説明するか。  
    - **成分負荷量（loadings）**：各変数が主成分にどれだけ寄与するか。  
    - **スコア（scores）**：各サンプルの主成分上の座標。  
    - **スクリープロット** と **バイプロット（PC1×PC2）** を表示。  
    - **CSVダウンロード**：成分負荷量・スコアを保存可能。
    """)

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
        st.error("少なくとも2列（1列目=目的変数、2列目以降=説明変数）が必要です。")
        return

    st.write("データプレビュー：")
    st.dataframe(df.head())

    # === y / X 分割（1列目=目的変数, 2列目以降=説明変数） ===
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



def tab2():
    st.write("Logistic回帰")

    st.subheader("目的")
    text_31="""
    - ある特定の事象が起きる確率を分析し、結果を予測する。"""
    st.markdown(text_31)
    st.subheader("使用ケース")
    text_32="""
    - 調査結果の個票データ解析: 説明変数として各メディアの接触有無（0,1データ）、目的変数として認知などのKPI有無（0,1データ）を使用して、各メディアの接触がKPIに与える影響を定量化する。GoogleトレンドやDS.INSIGHTなどからKWボリュームの過去傾向を分析し、季節性や長期トレンドを確認。
    - CV起点でのCP評価: IDベースに、CPごとにFQしたかどうかを説明変数として（0,1データ）、ある指定期間内にCVしたかどうかを目的変数としたときに（0,1データ）、過去蓄積効果があったのか確認する。"""
    st.markdown(text_32)
    st.subheader("inputデータ")
    text_33="""
    - 目的変数となる値とそれに伴う説明変数を入力。"""
    if st.button("Click me to go to folder"):
        st.write('[Go to folder](https://hakuhodody-my.sharepoint.com/:f:/r/personal/sd000905_hakuhodody-holdings_co_jp/Documents/%E7%B5%B1%E5%90%88AP%E5%B1%80_AaaS1-4%E9%83%A8_%E5%85%B1%E6%9C%89OneDrive/04.%20%E3%83%84%E3%83%BC%E3%83%AB%EF%BC%8F%E3%82%BD%E3%83%AA%E3%83%A5%E3%83%BC%E3%82%B7%E3%83%A7%E3%83%B3/megupy/01.input?csf=1&web=1&e=waFpBB)')
    st.markdown(text_33)
    st.subheader("アウトプット説明")
    text_34="""
    - **★importance**: 説明変数（各メディア接触有無）が目的変数（KPI）に与える貢献度をはかるための指標。
    - **odds**: オッズ比。importanceと大小関係は基本同じ。1より大きいならKPIに対して＋に働く1よりい低いなら－に働く。
    - P>|z|：P値。有意水準0.05を下回ればその説明変数は有意な偏回帰係数であることが言える。"""
    st.markdown(text_34)
    text_35="""
    - inputデータの目的変数と説明変数の入力位置に注意。"""
    st.markdown(text_35)

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

def tab3():
    st.write("順序Logistic回帰")

    st.subheader("目的")
    st.markdown("""
    - 段階的（順序あり）な目的変数を、説明変数で説明・予測する。
    """)

    st.subheader("使用ケース")
    st.markdown("""
    - 満足度1〜5、評価A/B/C 等の**順序ありカテゴリ**を扱いたいとき。
    """)

    st.subheader("inputデータ")
    st.markdown("""
    - 1列目：目的変数（順序カテゴリ or 数値/ラベル）
    - 2列目以降：説明変数（数値列）
    """)

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

    if df.shape[1] < 2:
        st.error("少なくとも2列（1列目=目的、2列目以降=説明変数）が必要です。")
        return

    st.write("データプレビュー：")
    st.dataframe(df.head())

    # --- y / X ---
    y_raw = df.iloc[:, 0]
    X = df.iloc[:, 1:].copy()
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

def tab4():
    st.write("MMM（軽量版）")

    st.subheader("目的")
    st.markdown("""
    - メディア投資に **アドストック（遅効）** と **飽和（逓減）** を入れた反応曲線を推定し、
      チャネル別の寄与・ROI・最適配分のヒントを得る。
    """)

    st.subheader("入力フォーマット")
    st.markdown("""
    - 1列目: `date`（日付/週の初日など）
    - 2列目: `y`（売上/コンバージョン等のKPI）
    - 3列目以降: 各チャネルの費用（例: `tv_spend`, `webcm_spend`, `search`, ...）
    """)

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
    contrib_df = pd.DataFrame(contrib, columns=spend_cols := spend_cols)
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
    coef_tbl = pd.DataFrame({"channel": spend_cols, "coef_on_transformed": coef_star})
    st.subheader("係数（変換後特徴量上）")
    st.dataframe(coef_tbl.style.format("{:.4f}"))

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



def tab5():
    st.write("STL分解")
    st.subheader("目的")
    text_21="""
    - 時系列データをトレンド、季節成分、残差に分解することにより、データの特性を把握する。"""
    st.markdown(text_21)
    st.subheader("使用ケース")
    text_22="""
    - GoogleトレンドやDS.INSIGHTなどからKWボリュームの過去傾向を分析し、季節性や長期トレンドを確認。"""
    st.markdown(text_22)
    st.subheader("inputデータ")
    text_23="""
    - 時系列での、期間とKWボリュームを入力。"""
    if st.button("Click me to go to folder"):
        st.write('[Go to folder](https://hakuhodody-my.sharepoint.com/:f:/r/personal/sd000905_hakuhodody-holdings_co_jp/Documents/%E7%B5%B1%E5%90%88AP%E5%B1%80_AaaS1-4%E9%83%A8_%E5%85%B1%E6%9C%89OneDrive/04.%20%E3%83%84%E3%83%BC%E3%83%AB%EF%BC%8F%E3%82%BD%E3%83%AA%E3%83%A5%E3%83%BC%E3%82%B7%E3%83%A7%E3%83%B3/megupy/01.input?csf=1&web=1&e=waFpBB)')
    st.markdown(text_23)
    st.subheader("アウトプット説明")
    text_24="""
    - **_raw**: 元の時系列データ
    - **_trend**: トレンド成分
    - **_seasonal**: 季節成分
    - **_resid**: 残差成分（トレンド&季節成分を除去した後のデータ）"""
    st.markdown(text_24)
    text_25="""
    - GoogleトレンドやDS.INSIGHTなどからKWボリュームの過去傾向を分析し、季節性や長期トレンドを確認。週別のデータも、月別のデータでもコードが判断してくれるので、どちらのケースでも使用可能。"""
    st.markdown(text_25)


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



def tab6():
    st.write("TIME最適化")

    st.subheader("目的")
    text_41="""
    - TIMEの複数素材割り付けを最適化"""
    st.markdown(text_41)
    st.subheader("使用ケース")
    text_42="""
    - 複数ブランドをTIMEで放映する場合
    - レギュラータイム/FTB/単発タイムなど固定の枠がある場合"""
    st.markdown(text_42)
    st.subheader("inputデータ")
    text_43="""
    - A-Dシートをそれぞれ入力"""
    if st.button("Click me to go to folder"):
        st.write('[Go to folder](https://hakuhodody.sharepoint.com/:f:/s/msteams_d8fd35/Eu6cDQ4W-t5KlsMGSjLhfQQBaYubS13B_Ge2FzODeaZO-A?e=lvq7tE)')
    st.markdown(text_43)
    st.subheader("アウトプット説明")
    text_44="""
    - ブランドごとの最適な番組フォーメーション
    - 番組追加による累積リーチ
    - 最適化後のブランドごとの予算"""
    st.markdown(text_44)
    text_45="""
    - AシートとCシートの番組IDは漏れなくダブりなく"""
    st.markdown(text_45)

    # タイトル
    st.title("モード選択")

    # プルダウン選択肢
    options = ["reach cost", "reach", "target_cost"]
    mode = st.selectbox("モードを選択してください", options, index=2)  # indexでデフォルト選択

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
def tab7():
    st.write("Causal Impact")

    st.subheader("目的")
    st.markdown("""
    - 広告出稿がKPIに与えた因果的影響を定量化する。
    - 出稿が無かった場合（カウンターファクト）のKPI推移を推定して、実績との差分＝リフトを把握する。
    """)

    st.subheader("使用ケース")
    st.markdown("""
    - **TVCM/キャンペーン効果検証**（出稿エリア vs 非出稿エリア）。
    - **介入日**を境に **前半が0、以降はずっと1** のフラグで評価。
    """)

    st.subheader("inputデータ")
    st.markdown("""
    - 必須列（ヘッダー名は任意）  
      1) **日付列**（例: `date` / `Date` / `日付`）  
      2) **出稿フラグ**（0=未出稿, 1=出稿。※ある時点から全て1）  
      3) **出稿エリアKPI（treated）**  
      4) **非出稿エリアKPI（control）**
    - 例：`date, flag, kpi_treated, kpi_control`
    """)

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
 
def tab8():
    st.write("Cuerve数式予測")
    st.subheader("目的")
    text_11="""
    - 目的変数（出稿量や予算）に対する説明変数（リーチや認知）の曲線を作成する。"""
    st.markdown(text_11)
    st.subheader("使用ケース")
    text_12="""
    - **出稿量（予算）とリーチの関係分析**: 広告出稿量の増加に対して、どの程度リーチが増加するかを予測。
    - **出稿量（予算）と認知度の関係分析**: 広告出稿量が増加に対して、どの程度認知度が上昇するかを予測。"""
    st.markdown(text_12)
    st.subheader("inputデータ")
    text_13="""
    - 目的変数となる値とそれに伴う説明変数を入力。"""
    if st.button("Click me to go to folder"):
        st.write('[Go to folder](https://hakuhodody-my.sharepoint.com/:f:/r/personal/sd000905_hakuhodody-holdings_co_jp/Documents/%E7%B5%B1%E5%90%88AP%E5%B1%80_AaaS1-4%E9%83%A8_%E5%85%B1%E6%9C%89OneDrive/04.%20%E3%83%84%E3%83%BC%E3%83%AB%EF%BC%8F%E3%82%BD%E3%83%AA%E3%83%A5%E3%83%BC%E3%82%B7%E3%83%A7%E3%83%B3/megupy/01.input?csf=1&web=1&e=waFpBB)')
    st.markdown(text_13)
    st.subheader("アウトプット説明")
    st.markdown("- モデルの数式")
    st.latex(r"""
    y = \frac{K}{1 + \left(a \left(\frac{x}{10^{dx}}\right)^b\right)} \cdot 10^{dy}
    """)
    text_14="""
    - 「dx」「dy」は桁数を揃えるための数値。
    - 出力された、「a」「b」「K」「dx」「dy」を上記式に代入。
    - Number of decisions（決定係数, R²）: モデルの適合度を示す。1に近い程モデルがデータにフィットしていることを意味する。"""
    st.markdown(text_14)




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


#tab5用の初期化、実行に関わる関数==========================
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

def tab6():
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
        tabs = st.sidebar.radio("メニュー", ["主成分分析","Logistic回帰", "順序Logistic回帰","MMM（軽量版）","STL分解", "TIME最適化", "Causal Impact", "Curve数式予測"])

        # ログアウトボタン
        if st.button("ログアウト"):
            st.session_state.logged_in = False
            st.experimental_rerun()  # ログアウト後にページを再実行してログイン画面に戻る

        if tabs == "主成分分析":
            tab1()
        elif tabs == "Logistic回帰":
            tab2()
        elif tabs == "順序Logistic回帰":
            tab3()
        elif tabs == "MMM（軽量版）":
            tab4()
        elif tabs == "STL分解":
            tab5()
        elif tabs == "TIME最適化":
            tab6()
        elif tabs == "Causal Impact":
            tab7()
        elif tabs == "Curve数式予測":
            tab8()


#実行コード
if __name__ == "__main__":
    main()