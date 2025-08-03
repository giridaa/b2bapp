#--ライブラリインポート--
import streamlit as st
import pandas as pd
import numpy as np
import io # バイトデータを読み込むioモジュール
import joblib
import lightgbm as lgb
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import google.generativeai as genai
import json # Geminiからの出力をJSON形式で扱うため
import time # APIレート制限対策のため追加
from pathlib import Path
#--------------------------------------------------

#--セッションステートの初期化--
if "df_master" not in st.session_state:
    st.session_state.df_master = None
if "df_encode_train" not in st.session_state:
    st.session_state.df_encode_train = None
if "df_encoded" not in st.session_state:
    st.session_state.df_encoded = None
if "train_encoded_df" not in st.session_state:
    st.session_state.train_encoded_df = None
if "df_hot_lead" not in st.session_state:
    st.session_state.df_hot_lead = None
if "model" not in st.session_state:
    st.session_state.model = None

#--トークスクリプトの提案商材設定URL初期化--
if "product_url" not in st.session_state:
    st.session_state.product_url = ""
#--------------------------------------------------

#--Gemini APIキー設定--
try:
    #--環境変数からAPIキー読み込み--
    gemini_api_key = os.environ["GEMINI_API_B2BAPP"]

    if gemini_api_key:
        genai.configure(api_key = gemini_api_key)
        # st.success("OK:API kye")
    else:
        st.error("APIキーが正しく設定されていません。")
        st.stop("APIキーエラーのため処理を停止します。")
except Exception as e:
    st.error(f"APIキー設定に予期せぬエラーが発生しました:{e}")
    exit()
#--------------------------------------------------

#--Geminiモデル読み込み--
try:
    # --Geminiモデル読み込み--
    model_gemini = genai.GenerativeModel(
        "gemini-2.5-flash-preview-05-20",
        generation_config={
            "response_mime_type": "application/json",  # JSON形式での応答を強制
            "response_schema": {  # 応答スキーマを定義
                "type": "object",
                "properties": {
                    "persona_name": {"type": "string"},
                    "persona_description": {"type": "string"},
                    "proposal_background": {"type": "string"},
                    "service_features": {"type": "string"},
                    "customer_benefits": {"type": "string"},
                },
                "required": ["persona_name", "persona_description"],
            },
        },
    )
    # st.success("Geminiモデルの読み込みが完了しました。")
except Exception as e:
    st.error(f"Geminiモデルの読み込み中にエラーが発生しました: {e}")
    st.stop() # モデル読み込みエラーの場合は処理を停止
#--------------------------------------------------

#--学習済モデルの読み込み--
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "b2b_lgbm.pkl")

#--モデルの存在を確認・読み込み--
if os.path.exists(MODEL_PATH):
    st.session_state.model_path = MODEL_PATH # モデルパスをセッションステートに保存
    try:
        st.session_state.model = joblib.load(MODEL_PATH)
        # st.success(f"OK: LGBM")
    except Exception as e:
        st.error(f"モデルの読み込み中にエラーが発生しました: {e}")
        model = None
else:
    st.error("モデルファイルが見つかりません。ファイルが同じディレクトリにあるか確認してください。")
    model = None
#--------------------------------------------------
st.set_page_config(layout="centered", page_title="インサイドセールスアプリ")

#--見出し：アプリのタイトル--
st.title("インサイドセールスアシスト")

#--------------------------------------------------

#--ファイルのアップロード--
#--見出し：リストのアップロード--
st.markdown('''### 1.推定用リストのアップロード :page_facing_up:''')

df_hot_lead = None
model = None
#--------------------------------------------------

#--ファイルアップローダー--
#--ファイルアップロードUIの表示--
uploaded_file = st.file_uploader("見込み客リストをドラッグ＆ドロップでアップロードしてください", type=["csv"])
#--ファイルがアップロードされたあとの処理--
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue() # ファイルの内容をバイトデータで取得し、bytes_dataに格納
    
    #--ファイル読み込みとdf_master作成、後続処理--
    df_master = None # 初期化
    try:
        # まずutf-8でデコードをTry
        df_master = pd.read_csv(io.StringIO(bytes_data.decode("utf-8")))
    except UnicodeDecodeError:
        # utf-8で失敗したらShift-JisでデコードをTry
        try:
            df_master = pd.read_csv(io.StringIO(bytes_data.decode("shift_jis")))
            st.session_state.df_master = df_master
        except UnicodeDecodeError:
            st.error("ファイル読み込みエラー: エンコーディングがUTF-8、またはSHIFT-JISではありません")
            df_master = None
        except Exception as e:
            st.error(f"SHIFT-JISでの読込中にエラーが発生しまた: {e}")
            df_master = None
    except Exception as e:
        st.error(f"ファイル読み込み中に予期せぬエラーが発生しました: {e}")
        df_master = None
    if st.session_state.df_master is not None: # データが正常に読み込まれた場合の処理
        st.success("リストがアップロードされました。") # アップロード成功メッセージを表示
        st.write("リストのビュー") # アップロードされたリストのビューを表示
        disprly_columns = [col for col in st.session_state.df_master.columns if col not in ["緯度", "経度"]]
        st.dataframe(st.session_state.df_master[disprly_columns].head()) # アップロードされたDataFrameを表示

        # original_display_cols をここで定義することで、後続の処理で常に利用可能にする
        original_display_cols = [
            col
            for col in st.session_state.df_master.columns
            if col not in ["メールアドレス", "姓", "名", "企業名", "緯度", "経度"]
        ]

        #--データ事前処理--
        #--One-hot学習用データ読み込み--
        FILE_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
        FILE_PATH = os.path.join(FILE_DIR, "20250609_testdata.csv")

        try: # 学習用データの読み込み
            #df_encode_train = pd.read_csv(FILE_PATH, io.StringIO(bytes_data.decode("utf-8")))
            df_encode_train = pd.read_csv(FILE_PATH, encoding="shift_jis")
        except FileNotFoundError:
            st.error(f"学習用データが見つかりません: {FILE_PATH}")
            st.stop() # 学習用データがない場合は処理を停止
        except Exception as e:
            st.error(f"学習用データの読み込み中にエラーが発生しました: {e}")

        #--One-hotエンコーディングのインスタンス作成
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sparse_outputに修正

        # One-hotエンコーディングの対象となるカテゴリカル列を定義
        categorical_cols = [
            "業種", "セクター", "所属部署", "職種", "役職",
            "メールマガジン登録状況", "最新のWeb", "最新の資料DL", "最新の参加イベント"
        ]        

        # --学習用データでエンコーダーを学習(fit)--
        # ##ここから修正
        encoder.fit(df_encode_train[categorical_cols])
        # ##ここまで修正

        # --推定用データに学習済のエンコーダーを適用(transform)--

        # 元データを変更しないよう、df_encodedにコピーし、変換を実行
        df_encoded_transform_array = encoder.transform(st.session_state.df_master[categorical_cols])

        # 変換後の特徴量名を取得
        feature_names = encoder.get_feature_names_out(categorical_cols)

        # Numpy配列をDataFrameに変換
        encoded_df_ohe_part = pd.DataFrame(
            df_encoded_transform_array,
            columns=feature_names,
            index=st.session_state.df_master.index # df_masterのインデックスを使用
        )

        # 元のdf_masterからOne-Hotエンコードした列を削除したDataFrameを作成
        df_master_numeric_etc = st.session_state.df_master.drop(columns=categorical_cols)

        # 削除した列とOne-Hotエンコードされた列を結合して、最終的なst.session_state.df_encodedを作成
        st.session_state.df_encoded = pd.concat([df_master_numeric_etc, encoded_df_ohe_part], axis=1)

    else: # 読み込み・エンコードエラー発生した場合のアラート表示
        st.warning("ファイル読み込み、またはエンコードエラーが発生しました、ファイル形式・内容を確認してください")   
else:
    st.write("---")
#--------------------------------------------------

#--HOT見込み客を推定--
# One-hotエンコーディングが完了した場合のみ表示
if st.session_state.df_encoded is not None:
    #--推定--
    #--見出し：HOT見込み客の推定--
    st.markdown('''### 2.HOT見込み客を推定 :dart:''')

    #--推定処理--
    if st.button("HOT見込み客の推定を実行"):
        if st.session_state.model is not None and st.session_state.df_encoded is not None:
            #--除外列の定義--
            predict_exclude_cols = ["id", "メールアドレス", "姓", "名", "企業名", "緯度", "経度", "都道府県", "市区町村", "住所"]
            #--推定用にdf_encodedのコピーを作成し、そのコピーから列を削除する--
            x_predict = st.session_state.df_encoded.copy()
            cols_to_drop_from_predict = [col for col in predict_exclude_cols if col in x_predict.columns]
            x_predict = x_predict.drop(columns=cols_to_drop_from_predict, errors="ignore")

            ##--予測結果:0/1を取得--
            #predictions = st.session_state.model.predict(x_predict)
            ##--予測確率を取得--
            #probabilities = st.session_state.model.predict_proba(x_predict)[:, 1] # クラス1(HOT)の確率

            # 予測確率を取得
            # lgb.Boosterの場合、predict()が予測確率を返す
            probabilities = st.session_state.model.predict(x_predict)
            # 0か1のラベルを取得
            # 予測確率が0.5より大きければ1、そうでなければ0
            predictions = np.where(probabilities > 0.5, 1, 0)

            #--推定結果をdfに反映--
            #--元データを変更しないよう、df_hot_leadにコピー--
            st.session_state.df_hot_lead = st.session_state.df_master.copy() # df_masterからコピー
            #--df_hot_leadにHOT判定とHOT予測確率を追加する前に、結合対象のdf_masterのインデックスと一致しているか確認--
            if len(st.session_state.df_hot_lead) == len(predictions):
                st.session_state.df_hot_lead["HOT判定"] = predictions
                st.session_state.df_hot_lead["HOT予測確率"] = probabilities
            else:
                st.error("HOT判定結果の行数と元のデータ行数が一致しません。")

            #--推定結果の表示--
            st.subheader("HOT見込み客の推定結果")
            st.success("HOT見込み客の推定が完了しました")
            selected_columns_hot_lead = ["id","HOT判定", "HOT予測確率", "姓", "名", "企業名", "所属部署", "役職"] # 表示する列を指定
            st.dataframe(st.session_state.df_hot_lead[selected_columns_hot_lead].head())
        else:
            st.warning("モデル読み込みエラー、またはファイルがアップロードされていません。")
#--------------------------------------------------

#--クラスタリング--
#--df_encodedとdf_hot_leadが存在する場合処理を実行--
if st.session_state.df_encoded is not None and st.session_state.df_hot_lead is not None:
    #--見出し：HOT見込み客のペルソナ生成--
    st.markdown('''### 3.HOT見込み客のペルソナを生成 :face_in_clouds:''')

    #--クラスター数の選択--
    persona_num = st.slider("生成したいペルソナ数を設定", 2, 5, 3)
    st.write(persona_num, "種類のペルソナを生成します。")

    #--クラスタリング実行--
    if st.button("ペルソナを生成する"):
        #--除外列を指定--
        kmeans_exclude_cols = ["id", "メールアドレス", "姓", "名", "企業名", "緯度", "経度", "都道府県", "市区町村", "住所"]

        #--クラスタリング用にdf_encodedのコピーを作成し、そのコピーから列を削除する--
        x_kmeans = st.session_state.df_encoded.copy()
        x_kmeans = x_kmeans.drop(
            columns=[col for col in kmeans_exclude_cols if col in x_kmeans.columns],
            errors="ignore"
        )
        #--クラスタリング前に欠損値がある場合、0で埋める--
        if x_kmeans.isnull().sum().sum() > 0:
            st.warning("クラスタリングデータに欠損値が存在します。欠損値を0で埋めます。")
            x_kmeans = x_kmeans.fillna(0)

        #--K-Means用DataFrameが空であるか、または全ての値が同じであるかチェック--
        if x_kmeans.empty or x_kmeans.nunique().sum() == 0:
            st.error("クラスタリング用のデータが空であるか、全ての値が同じであるため、クラスタリングを実行できません。")
            st.info("入力データを確認してください。")
        else:
            try:
                #--K-meansの実行--
                kmeans = KMeans(n_clusters=persona_num, random_state=0, n_init="auto")
                personas = kmeans.fit_predict(x_kmeans)

                #--クラスタリング結果をdf_hot_leadに追加--
                if len(st.session_state.df_hot_lead) == len(personas):
                    st.session_state.df_hot_lead["persona"] = personas
                else:
                    st.error("HOT見込み客データとクラスタリング結果の行数が一致しません。")
                    st.info("データ処理フローを確認してください。")

                ##ペルソナ生成
                #--ペルソナ説明テキストの生成と追加--
                st.markdown('''#### 生成したペルソナの説明''')

                # ペルソナを説明するために使用する列
                persona_feature_cols = ["業種", "セクター", "所属部署", "職種", "役職"]
                
                # ペルソナごとの説明テキストを格納する辞書
                persona_descriptions = {}

                with st.spinner("Geminiがペルソナ像を生成中です...少々お待ちください。"):
                    for i in range(persona_num):
                        # 各ペルソナグループのデータを抽出
                        persona_df = st.session_state.df_hot_lead[st.session_state.df_hot_lead["persona"] == i]

                        if not persona_df.empty:
                            # 各特徴量の最頻値（最も出現回数が多い値）を取得
                            persona_features = {}
                            for col in persona_feature_cols:
                                # 最頻値が複数ある場合は最初のものを採用
                                if not persona_df[col].mode().empty:
                                    persona_features[col] = persona_df[col].mode()[0]
                                else:
                                    persona_features[col] = "不明" # データがない場合

                            # Geminiに送るプロンプトの作成
                            prompt = f"""
                            以下の特徴を持つペルソナの「ペルソナ名」を10文字程度、「ペルソナの特徴」を100文字程度で生成してください。
                            ペルソナ名はペルソナの特徴を要約して作成してください。
                            ペルソナ名には、職種、役職、姓、名の値は含めないでください。
                            ペルソナの特徴には、業種、セクター、所属部署、職種、役職は表示しないでください。
                            ペルソナの特徴には、こちらを参照して、業種、セクターの最新情報を含めてください: https://www.tdb.co.jp/report/industry/
                            ペルソナ名とペルソナの特徴は、JSON形式で返してください。

                            特徴:
                            - 業種: {persona_features.get("業種", "不明")}
                            - セクター: {persona_features.get("セクター", "不明")}
                            - 所属部署: {persona_features.get("所属部署", "不明")}
                            - 職種: {persona_features.get("職種", "不明")}
                            - 役職: {persona_features.get("役職", "不明")}

                            応答は以下のJSONスキーマに従ってください:
                            {{
                                "persona_name": "string",
                                "persona_description": "string"
                            }}
                            """

                            try:
                                response = model_gemini.generate_content(prompt)
                                # JSON形式で返されるため、json.loadsでパース
                                persona_data = json.loads(response.text)
                                persona_name = persona_data.get("persona_name", f"ペルソナ {i}")
                                persona_description = persona_data.get("persona_description", "特徴を説明できませんでした。")
                                
                                persona_descriptions[i] = f"【{persona_name}】{persona_description}"
                                st.write(f"■ペルソナ {i} の説明: {persona_descriptions[i]}")
                                time.sleep(1) # APIレート制限対策
                            except Exception as e:
                                st.error(f"ペルソナ {i} の生成中にエラーが発生しました: {e}")
                                st.info("Gemini APIからの応答形式が正しくないか、APIの呼び出しに問題がある可能性があります。")
                                persona_descriptions[i] = "ペルソナ説明生成エラー"
                        else:
                            persona_descriptions[i] = "このペルソナにはデータがありません。"

                    # df_hot_leadに新しい列「ペルソナ像」を追加し、ペルソナ説明を格納
                    # persona_descriptions辞書をpd.Seriesに変換し、df_hot_leadの'persona'列をキーとして結合
                    if "persona" in st.session_state.df_hot_lead.columns:
                        st.session_state.df_hot_lead["ペルソナ像"] = st.session_state.df_hot_lead["persona"].map(persona_descriptions)
                        st.success("ペルソナ像の生成とDataFrameへの追加が完了しました。")
                        
                        # ペルソナ像を含む最終的なDataFrameの表示
                        selected_columns_persona = ["id", "HOT判定", "HOT予測確率", "persona", "ペルソナ像", "姓", "名", "企業名", "所属部署", "役職"] # 表示する列を指定
                        st.dataframe(st.session_state.df_hot_lead[selected_columns_persona].head())
                    else:
                        st.error("ペルソナ列が見つかりません。ペルソナ像の追加をスキップします。")

                ##ペルソナ生成終わり

            except Exception as e:
                st.error(f"K-Meansクラスタリング中にエラーが発生しました: {e}")
                st.info("入力データに数値以外のデータが含まれていないか、またはデータが全て同じ値になっていないか確認してください。")

#else:
    #st.info("ファイルがアップロードされていないか、前処理が完了していません。")

#--------------------------------------------------

#--トークスクリプト生成--
if (
    st.session_state.df_hot_lead is not None
    and "ペルソナ像" in st.session_state.df_hot_lead.columns
):
    #--見出し：トークスクリプト生成--
    st.markdown('''### 4.各ペルソナへのトークスクリプトを生成 :speech_balloon:''')

    # 「提案する商材」のURL設定フォーム
    product_url = st.text_input(
        "提案する商材のURLを入力してください（例: https://www.example.com/product）",
        value="提案する商材のWebページのURLを入力",  # 初期値を設定
        key="main_product_url"  # キーを追加して、一意性を確保
    )

    if not product_url:
        st.warning("提案する商材のURLを入力してください。")
    
    if st.button("設定した提案する商材でペルソナ向けトークスクリプトを生成する"):
        if not product_url:
            st.warning("提案する商材のURLが設定されていません。")
        else:
            st.subheader("トークスクリプト生成結果")
            all_personas = st.session_state.df_hot_lead["persona"].unique()
            talk_scripts = {}

            with st.spinner("Geminiがトークスクリプトを生成中です...少々お待ちください。"):
                for persona_id in sorted(all_personas):
                    # 該当ペルソナの代表的な説明を取得
                    # ここで、persona_dfが空でないことを確認してからiloc[0]にアクセスします
                    persona_df = st.session_state.df_hot_lead[
                        st.session_state.df_hot_lead["persona"] == persona_id
                    ]
                    if not persona_df.empty:
                        persona_description_text = persona_df["ペルソナ像"].iloc[0]
                    else:
                        persona_description_text = "データなし" # データがない場合のデフォルト値

                    st.write(f"--- ペルソナ {persona_id} のトークスクリプト ---")
                    st.write(f"**ペルソナ像:** {persona_description_text}")

                    # トークスクリプト生成用のプロンプト
                    talk_script_prompt = f"""
                    以下の情報と提案する商材のURLを元に、ペルソナ向けの営業トークスクリプトを生成してください。
                    「提案の背景」を20文字程度、「提案するサービスの特徴」を100文字程度、「お客様へのメリット」を100文字程度で生成してください。
                    生成するトークスクリプトはJSON形式で返してください。

                    ペルソナ像: {persona_description_text}
                    提案する商材URL: {product_url}

                    応答は以下のJSONスキーマに従ってください:
                    {{
                        "proposal_background": "string",
                        "service_features": "string",
                        "customer_benefits": "string"
                    }}
                    """

                    try:
                        response_talk = model_gemini.generate_content(talk_script_prompt)
                        talk_script_data = json.loads(response_talk.text)

                        proposal_background = talk_script_data.get(
                            "proposal_background", "背景を生成できませんでした。"
                        )
                        service_features = talk_script_data.get(
                            "service_features", "サービスの特徴を生成できませんでした。"
                        )
                        customer_benefits = talk_script_data.get(
                            "customer_benefits", "お客様へのメリットを生成できませんでした。"
                        )

                        full_talk_script = (
                            f"【提案の背景】: {proposal_background}\n"
                            f"【提案するサービスの特徴】: {service_features}\n"
                            f"【お客様へのメリット】: {customer_benefits}"
                        )
                        talk_scripts[persona_id] = full_talk_script

                        st.markdown(f"**提案の背景:** {proposal_background}")
                        st.markdown(f"**提案するサービスの特徴:** {service_features}")
                        st.markdown(f"**お客様へのメリット:** {customer_benefits}")
                        time.sleep(1)  # APIレート制限対策

                    except Exception as e:
                        st.error(
                            f"ペルソナ {persona_id} のトークスクリプト生成中にエラーが発生しました: {e}"
                        )
                        st.info(
                            "Gemini APIからの応答形式が正しくないか、APIの呼び出しに問題がある可能性があります。"
                        )
                        talk_scripts[persona_id] = "トークスクリプト生成エラー"

            # df_hot_leadに新しい列「トークスクリプト」を追加し、生成したトークスクリプトを格納
            if "persona" in st.session_state.df_hot_lead.columns:
                st.session_state.df_hot_lead["トークスクリプト"] = st.session_state.df_hot_lead[
                    "persona"
                ].map(talk_scripts)
                st.success("トークスクリプトの生成とDataFrameへの追加が完了しました。")

                # トークスクリプトを含む最終的なDataFrameの表示
                final_display_cols_with_talk_script = [
                    "id",
                    "HOT判定",
                    "HOT予測確率",
                    "persona",
                    "ペルソナ像",
                    "トークスクリプト",
                ] + [
                    col
                    for col in original_display_cols
                    if col
                    in st.session_state.df_hot_lead.columns
                    and col not in ["id", "HOT判定", "HOT予測確率", "persona", "ペルソナ像"]
                ]
                st.dataframe(
                    st.session_state.df_hot_lead[
                        final_display_cols_with_talk_script
                    ].head()
                )   

                #--ダッシュボード向けにHOT見込み客リストをCSVとして保存
                def ganerate_and_save_hot_lea_csv():
                    st.session_state.df_hot_lead.to_csv("hot_lead.csv", index=False)
                if __name__ == "__main__":
                    ganerate_and_save_hot_lea_csv()

                #--HOTリストをデスクトップに保存--
                st.download_button(
                    label = "HOTリストをデスクトップに保存", 
                    data = st.session_state.df_hot_lead.to_csv(
                        index=False,
                        encoding="utf-8-sig", # Excelで開くときに文字化けしないようエンコーディングも変更
                        quoting=1, # csv.QUOTE_ALLと同じ意味
                        quotechar='"'
                    ).encode('utf-8-sig'), # バイナリデータも同様に変更
                    file_name = "hot_lead.csv",
                    mime = "text/csv",
                    help = "クリックするとHOTリストをCSVファイルでDLします"        
                )
                st.success("CSVの利用が可能です")

                st.markdown("---")
                st.page_link("pages/dashboard.py", label="生成したリストをダッシュボードで分析する", icon="📊")

            else:
                st.error("ペルソナ列が見つかりません。トークスクリプトの追加をスキップします。")
#else:
    #st.info("ペルソナ生成が完了していないか、データがアップロードされていません。")