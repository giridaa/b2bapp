#--------------------------------------------------
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
#--------------------------------------------------

# --- セッションステートの初期化 ---
if "df_master" not in st.session_state:
    st.session_state.df_master = None
if "df_encoded" not in st.session_state:
    st.session_state.df_encoded = None
if "df_hot_lead" not in st.session_state:
    st.session_state.df_hot_lead = None
if "model" not in st.session_state:
    st.session_state.model = None
#--------------------------------------------------

#--見出し：アプリのタイトル
st.title("HOTリード予測・ペルソナ＆トークスクリプト生成")
#--------------------------------------------------

#--ファイルのアップロード
#--見出し：ファイルのアップロード
st.header("リードリストアップロード")

df_hot_lead = None
model = None
#--------------------------------------------------

#--ファイルアップローダー
#--ファイルアップロードUIの表示
uploaded_file = st.file_uploader("リードリストをドラッグ＆ドロップでアップロードしてください", type=["csv"])
#--ファイルがアップロードされたあとの処理
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue() # ファイルの内容をバイトデータで取得し、bytes_dataに格納
    
    #--ファイル読み込みとdf_master作成、後続処理
    try: #--まず、utf-8で読み込みをTry
        df_master = pd.read_csv(io.StringIO(bytes_data.decode("utf-8"))) # バイトデータをUTF-8でdf_masterに格納
        st.session_state.df_master = df_master # df_masterをセッションステートに保存
    except UnicodeDecodeError:
        st.error("ファイル読み込みエラー:エンコーディングがutf-8/shift-jisではありません") # utf-8で読み込めなかった場合エラーメッセージを表示
        df_master = None # 読み込みエラー時はdf_masterをNone
    except Exception as e:
        st.error(f"予期せぬエラーが発生しました: {e}") # その他の予期せぬエラーの場合エラーメッセージを表示
        df_master = None # 読み込みエラー時はdf_masterをNone

    if st.session_state.df_master is not None: # データが正常に読み込まれた場合の処理
        st.write("アップロードしたファイル名:", uploaded_file.name) # アップロードされたファイル名を表示
        st.success("ファイルがアップロードされました。") # アップロード成功メッセージを表示
        st.dataframe(st.session_state.df_master.head()) # アップロードされたDataFrameを表示

        #--データ事前処理
        #--元データを変更しないよう、df_encodedにコピー
        df_encoded = st.session_state.df_master.copy()
        st.session_state.df_encoded = df_encoded # df_encodedをセッションステートに保存

        #--One-hotエンコーディング
        one_hot_cols = ["業種", "セクター", "所属部署", "職種", "役職", "メールマガジン登録状況", "最新のWeb", "最新の資料DL", "最新の参加イベント"]
        one_hot_encoder = OneHotEncoder(sparse_output = False, handle_unknown = "ignore") # "ignore": 未知のカテゴリが出現した場合にエラーとせず、全て0の列として扱う
        onehot_encoded = one_hot_encoder.fit_transform(st.session_state.df_master[one_hot_cols])
        encoded_df_ohe_part = pd.DataFrame(onehot_encoded, columns=one_hot_encoder.get_feature_names_out(one_hot_cols), index=st.session_state.df_master.index) # df_masterのインデックスを使用

        #--元のdf_masterからOne-Hotエンコードした列を削除したDataFrameを作成
        df_master_numeric_etc = st.session_state.df_master.drop(columns=one_hot_cols)
        #--削除した列とOne-Hotエンコードされた列を結合して、最終的なdf_encodedを作成
        st.session_state.df_encoded = pd.concat([df_master_numeric_etc, encoded_df_ohe_part], axis=1)
        st.write("エンコード結果プレビュー")
        st.dataframe(st.session_state.df_encoded.head())

    else: # 読み込み・エンコードエラー発生した場合のアラート表示
        st.warning("ファイル読み込み、またはエンコードエラーが発生しました、ファイル形式・内容を確認してください")
    
else:
    st.info("まだファイルはアップロードされていません。") # 未アップロードの場合、アップロードを促す表示
#--------------------------------------------------
#--HOTリードを推論
#--学習済モデルの読み込み
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "b2b_lgbm.pkl")
#--モデルの存在を確認・読み込み
if os.path.exists(MODEL_PATH):
    st.session_state.model_path = MODEL_PATH # モデルパスをセッションステートに保存
    try:
        st.session_state.model = joblib.load(MODEL_PATH)
        st.success(f"学習済モデル '{MODEL_PATH}' を読み込みました。")
    except Exception as e:
        st.error(f"モデルの読み込み中にエラーが発生しました: {e}")
        model = None
else:
    st.error("モデルファイルが見つかりません。ファイルが同じディレクトリにあるか確認してください。")
    model = None

#--推論
#--見出し：HOTリードの推定
st.header("HOTリードを推定する")

#--推定処理
if st.button("HOTリードの推定を実行"):
    if st.session_state.model is not None and st.session_state.df_encoded is not None:
        #--除外列の定義
        predict_exclude_cols = ["id", "メールアドレス", "姓", "名", "企業名", "緯度", "経度"]

        # ⭐️修正点1: 推論用にdf_encodedのコピーを作成し、そのコピーから列を削除する
        x_predict = st.session_state.df_encoded.copy()
        cols_to_drop_from_predict = [col for col in predict_exclude_cols if col in x_predict.columns]
        x_predict = x_predict.drop(columns=cols_to_drop_from_predict, errors="ignore")

        #--予測結果:0/1を取得
        predictions = st.session_state.model.predict(x_predict)
        #--予測確率を取得
        probabilities = st.session_state.model.predict_proba(x_predict)[:, 1] # クラス1(HOT)の確率

        #--推定結果をdfに反映
        #--元データを変更しないよう、df_hot_leadにコピー
        st.session_state.df_hot_lead = st.session_state.df_master.copy() # df_masterからコピー
        # df_hot_leadにHOT判定とHOT率を追加する前に、結合対象のdf_masterのインデックスと一致しているか確認
        if len(st.session_state.df_hot_lead) == len(predictions):
            st.session_state.df_hot_lead["HOT判定"] = predictions
            st.session_state.df_hot_lead["HOT率"] = probabilities
        else:
            st.error("HOT判定結果の行数と元のデータ行数が一致しません。")

        #--推定結果の表示
        st.subheader("HOTリードの推定結果")
        st.dataframe(st.session_state.df_hot_lead.head(20))
        st.success("HOTリードの推定が完了しました")

    else:
        st.warning("モデル読み込みエラー、またはファイルがアップロードされていません。")
else:
    st.info("「HOTリードの推定を実行」ボタンを押してください。")
#--------------------------------------------------

#--クラスタリング
#--df_encodedとdf_hot_leadが存在する場合処理を実行
if st.session_state.df_encoded is not None and st.session_state.df_hot_lead is not None:
    #--見出し：ペルソナ・トークスクリプト生成
    st.header("ペルソナ・トークスクリプトを生成する")

    #--クラスター数の選択
    cluster_num = st.slider("生成したいペルソナ数を設定", 2, 5, 3)
    st.write(cluster_num, "種類のペルソナを生成します。")

    #--クラスタリング実行
    if st.button("ペルソナを生成する"):
        #--除外列を指定
        kmeans_exclude_cols = ["id", "メールアドレス", "姓", "名", "企業名", "緯度", "経度"]

        # ⭐️修正点2: クラスタリング用にdf_encodedのコピーを作成し、そのコピーから列を削除する
        x_kmeans = st.session_state.df_encoded.copy()
        x_kmeans = x_kmeans.drop(
            columns=[col for col in kmeans_exclude_cols if col in x_kmeans.columns],
            errors="ignore"
        )
        # クラスタリング前に欠損値がある場合、0で埋めます
        if x_kmeans.isnull().sum().sum() > 0:
            st.warning("クラスタリングデータに欠損値が存在します。欠損値を0で埋めます。")
            x_kmeans = x_kmeans.fillna(0)

        # K-Means用DataFrameが空であるか、または全ての値が同じであるかチェックします
        if x_kmeans.empty or x_kmeans.nunique().sum() == 0:
            st.error("クラスタリング用のデータが空であるか、全ての値が同じであるため、クラスタリングを実行できません。")
            st.info("入力データを確認してください。")
        else:
            try:
                # K-meansの実行
                kmeans = KMeans(n_clusters=cluster_num, random_state=0, n_init="auto")
                clusters = kmeans.fit_predict(x_kmeans)

                # クラスタリング結果をdf_hot_leadに追加します
                if len(st.session_state.df_hot_lead) == len(clusters):
                    st.session_state.df_hot_lead["cluster"] = clusters
                else:
                    st.error("HOTリードデータとクラスタリング結果の行数が一致しません。")
                    st.info("データ処理フローを確認してください。")

                st.success(f"{cluster_num}種類のペルソナを生成しました。")

                # 最終的なDataFrameの表示列を調整します
                # df_hot_lead は df_master のコピーから作成されているため、id 列も含まれています。
                original_display_cols = [
                    col for col in st.session_state.df_master.columns
                    if col not in ["メールアドレス", "姓", "名", "企業名", "緯度", "経度"]
                ]

                final_display_cols = ["id", "HOT判定", "HOT率", "cluster"] + [
                    col for col in original_display_cols if col in st.session_state.df_hot_lead.columns and col != "id"
                ]

                st.dataframe(st.session_state.df_hot_lead[final_display_cols].head(100))

            except Exception as e:
                st.error(f"K-Meansクラスタリング中にエラーが発生しました: {e}")
                st.info("入力データに数値以外のデータが含まれていないか、またはデータが全て同じ値になっていないか確認してください。")

else:
    st.info("ファイルがアップロードされていないか、前処理が完了していません。")