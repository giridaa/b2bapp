#--ライブラリインポート--
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
#import japanize_matplotlib
import seaborn as sns
import folium
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time # 処理時間計測用
st.set_page_config(layout="centered", page_title="セールスアプリ")
#--------------------------------------------------

#--HOT見込み客CSV読み込み--
csv_file_name = "pages/hot_lead.csv"

#--CSVファイルをDataFrameとして読み込み
try:
    df =pd.read_csv("pages/hot_lead.csv")
except FileNotFoundError:
    st.error(f"エラー: '{pages/hot_lead.csv}'が見つかりません")
    st.stop()
except Exception as e:
    st.error(f"ファイル読み込みエラー: {e}")
    st.stop()
#--------------------------------------------------


#--ダッシュボードタイトル--
st.title("セールスアシスト")
#--------------------------------------------------

#--サイドバー1--
st.sidebar.subheader("1.アプローチ候補HOT見込み客閾値設定")

#--HOT予測確率選択スライダ--
selected_threshold_value = st.sidebar.slider(
    "HOT閾値を選択してください",
    min_value = 0.1,
    max_value = 0.9,
    value = 0.5, # 初期値を50%
    step = 0.1, # 10%刻み
    format = "%.1f" # 表示形式を小数点第一位まで
)

# スライダーで選択された値に基づいてHOT予測確率のラベルを動的に表示
# 0.1刻みの場合、例えば0.4を選択したら「40%以下」とする
if selected_threshold_value <= 0.4:
    selected_hot_rate_label = f"{int(selected_threshold_value * 100)}%以下"
else:
    selected_hot_rate_label = f"{int(selected_threshold_value * 100)}%以上"

# HOT予測確率でのフィルタリング
# selected_threshold_value を使ってフィルタリング
if selected_threshold_value <= 0.4:
    df_filtered = df[df["HOT予測確率"] <= selected_threshold_value]
else:
    df_filtered = df[df["HOT予測確率"] >= selected_threshold_value]


#--集計軸の選択
st.sidebar.write("**集計軸選択**")
available_columns = ["企業名", "業種", "セクター", "所属部署", "職種", "役職"]
actual_aggregation_axes = [col for col in available_columns if col in df_filtered.columns]

if not actual_aggregation_axes:
    st.sidebar.warning("集計可能な列が見つかりません。CSVファイルに '業種', 'セクター' などの列があるか確認してください。")
    # st.stop() # サイドバーの警告なので、ここではアプリを停止しない
aggregation_axis = st.sidebar.selectbox("集計軸を選択してください:", actual_aggregation_axes)

#--集計形式の選択
st.sidebar.write("**集計形式選択**")
option = st.sidebar.radio("表示形式を選択してください:", ["表", "バーチャート"])
#--------------------------------------------------


# ダッシュボード表示
#--見出し：アプローチするHOT見込み客抽出--
st.markdown('''### 1.アプローチ候補のHOT見込み客抽出 :mag:''')
st.markdown(f"**選択したHOT予測確率: <span style='color:red; text-decoration: underline;'>{selected_hot_rate_label}</span>以上のリード**", unsafe_allow_html=True)

# フィルタリング後のデータフレームと該当レコード数を表示
st.write(f"**該当リード数: <span style='color:red; text-decoration: underline;'>{len(df_filtered)}</span>人**", unsafe_allow_html=True)
#--フィルタリングで表示する項目を設定--
selected_columns_df = df_filtered[["HOT予測確率", "企業名", "姓", "名", "所属部署", "職種", "役職"]]
sorted_df = selected_columns_df.sort_values(by = "HOT予測確率", ascending = False) # HOT予測確率で降順ソート
st.dataframe(sorted_df.head())

# 集計軸ごとのサマリー表示
if aggregation_axis and not df_filtered.empty:
    st.write(f"**{aggregation_axis}別サマリー**")

        # 選択された集計軸でグループ化し、レコード数のみを計算
    summary_df = df_filtered.groupby(aggregation_axis).agg(
        レコード数=('HOT予測確率', 'count') # レコード数のみ
    ).reset_index()

    # ソート（任意）
    summary_df = summary_df.sort_values(by='レコード数', ascending=False) # レコード数でソート

    if option == "表":
        st.dataframe(summary_df)
    elif option == "バーチャート":
        # バーチャート用のデータフレーム
        fig, ax = plt.subplots(figsize=(12, 7)) # 図のサイズを調整

        # バーチャート：レコード数のみ
        sns.barplot(x=aggregation_axis, y="レコード数", data=summary_df, ax=ax, palette='viridis')
        ax.set_title(f"レコード数（{aggregation_axis}別）")
        ax.set_xlabel(aggregation_axis)
        ax.set_ylabel("レコード数")

        # X軸のラベルが重ならないように回転
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout() # レイアウトの調整
        st.pyplot(fig)

else:
    st.info("選択されたHOT予測確率の範囲では、表示するデータがありません。")
#--------------------------------------------------


#--サイドバー2--
st.sidebar.subheader("2.アプローチするHOT見込み客選択")

#--HOT見込み客選択スライダ--
#--降順ソートされたデータフレームの行数を取得
sorted_attack_df = df_filtered.sort_values(by = "HOT予測確率", ascending = False) # HOT予測確率で降順ソート
max_rows = len(sorted_attack_df)

#--スライダ設定
start_rank, end_rank = st.sidebar.slider(
    "アプローチするHOT見込み客の範囲を選択してください",
    min_value = 1,          # 最小値は1
    max_value = max_rows,   # 最大はデータ件数
    value = (1, 5)   # 初期値は1位から5位
)

#--選択された亜愛が有効かチェック(開始順位が終了順位より大きくないか)
if start_rank > end_rank:
    st.sidebar.error("開始順位は終了順位以下にしてください")
#--選択されたレコードをdf_attackに格納
# ilocで指定した範囲の行を抽出。Pythonのインデックすは0から始まるので、スライダー値から-1
else:
    df_attack = sorted_attack_df.iloc[start_rank - 1: end_rank]

#--選択されたレコードを表示
#--見出し：アプローチするHOT見込み客抽出--
st.markdown('''### 2.アプローチするHOT見込み客選択 :busts_in_silhouette:''')
st.markdown(f"**選択されたアプローチするHOT見込み客: <span style='color:red; text-decoration: underline;'>HOT予測確率上位{start_rank}から{end_rank}まで</span>**", unsafe_allow_html=True)
st.markdown(f"**該当リード数: <span style='color:red; text-decoration: underline;'>{len(df_attack)}</span>人**", unsafe_allow_html=True)
selected_columns_df = df_attack[["HOT予測確率", "企業名", "姓", "名", "都道府県", "市区町村"]] # 表示項目を選択
sorted_df = selected_columns_df.sort_values(by = "HOT予測確率", ascending = False) # HOT予測確率で降順ソート
st.dataframe(sorted_df)

#--アプローチルート表示--

# --- 1. 東京駅の緯度・経度を取得し、スタート地点として追加 ---
@st.cache_data
def get_tokyo_station_coords():
    geolocator = Nominatim(user_agent="tsp_streamlit_app")
    try:
        location_tokyo_station = geolocator.geocode("東京駅")
        return location_tokyo_station.latitude, location_tokyo_station.longitude
    except Exception as e:
        st.error(f"東京駅の緯度・経度の取得に失敗しました。デフォルト値を使用します。エラー: {e}")
        # 失敗した場合のフォールバック（手動設定）
        return 35.681236, 139.767125 # 東京駅の代表的な緯度・経度

# --- 2. 距離行列の作成 ---
@st.cache_data
def create_distance_matrix(coords):
    num_locations = len(coords)
    matrix = np.zeros((num_locations, num_locations))
    for i in range(num_locations):
        for j in range(num_locations):
            if i == j:
                matrix[i, j] = 0
            else:
                # 緯度経度からメートル単位の距離を計算
                matrix[i, j] = geodesic(coords[i], coords[j]).meters
    return matrix.astype(int) # ortoolsは整数を要求するため

# --- 3. 巡回セールスマン問題の解決 (OR-Tools) ---
@st.cache_data
def solve_tsp(distance_matrix):
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, 0) # 1 vehicle, start node is 0
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # 探索パラメータの設定
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = 30 # 探索時間制限（秒）

    # 経路を探索
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        index = routing.Start(0)
        route_indices = []
        while not routing.IsEnd(index):
            route_indices.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route_indices.append(manager.IndexToNode(index)) # 最後のノードを追加
        return route_indices, solution.ObjectiveValue()
    else:
        return None, None

# --- Streamlit アプリケーションの開始 ---
def main_app_logic(): # main関数名を変更し、ボタンクリック後に呼び出すようにする
    st.markdown('''### 3.最短アプローチルートを確認 :world_map:''')

    # ここから修正
    # 「最適なアプローチルートを計算する」ボタンを配置
    if st.button("最適なアプローチルートを計算する"):
        # df_attackが空の場合のハンドリング
        if df_attack.empty or not all(col in df_attack.columns for col in ['企業名', '緯度', '経度']):
            st.warning("`df_attack` にデータがありません。または必要な列（'企業名', '緯度', '経度'）が存在しません。")
            st.info("アプローチするHOT見込み客を選択してからボタンをクリックしてください。")
            return # ここで処理を中断
        
        # 実際に処理する df_attack
        df_for_processing = df_attack

        # 処理対象の企業名と緯度経度のみを抽出
        locations = df_for_processing[['企業名', '緯度', '経度']].copy()

        tokyo_station_lat, tokyo_station_lon = get_tokyo_station_coords()

        # スタート地点をDataFrameの先頭に追加
        start_location = pd.DataFrame({
            '企業名': ['東京駅 (スタート)'],
            '緯度': [tokyo_station_lat],
            '経度': [tokyo_station_lon]
        })
        all_locations = pd.concat([start_location, locations], ignore_index=True)

        st.write("---")
        st.write("#### アプローチ対象地点一覧 (東京駅含む)")
        display_columns = [col for col in all_locations.columns if col not in ['緯度', '経度']]
        st.dataframe(all_locations[display_columns])
        st.write("---")

        coords = all_locations[['緯度', '経度']].values

        st.write("#### 最短アプローチルートを計算中...")
        with st.spinner('計算に時間がかかる場合があります...'):
            distance_matrix = create_distance_matrix(coords)
            route_indices, total_distance = solve_tsp(distance_matrix)

        if route_indices is None:
            st.error("最短ルートが見つかりませんでした。地点数やタイムアウト設定を確認してください。")
            return

        st.success("最短アプローチルートの計算が完了しました！")
        st.write(f"**総移動距離**: {total_distance / 1000:.2f} km")

        # 最適ルートの企業名、緯度・経度を取得
        optimal_route_info = all_locations.iloc[route_indices].reset_index(drop=True)

        st.write("#### 最短アプローチルートの順序")
        # 巡回順序に「順位」列を追加
        optimal_route_info['順序'] = range(1, len(optimal_route_info) + 1)
        display_columns2 = [col for col in optimal_route_info.columns if col not in ['緯度', '経度']]
        st.dataframe(optimal_route_info[display_columns2])
        #st.dataframe(optimal_route_info[['順序', '企業名', '緯度', '経度']])


        st.write("#### 最短アプローチルートマップ")

        # 地図の中心を東京駅に設定
        m = folium.Map(location=[tokyo_station_lat, tokyo_station_lon], zoom_start=12)

        # 各ポイントを地図に追加し、ポップアップに情報を表示
        prev_coords = None
        for i, row in optimal_route_info.iterrows():
            current_coords = (row['緯度'], row['経度'])
            popup_text = f"**{i+1}. {row['企業名']}**<br>" # 巡回する順番
            popup_text += f"緯度: {row['緯度']:.4f}<br>経度: {row['経度']:.4f}"

            if prev_coords:
                distance_segment = geodesic(prev_coords, current_coords).meters
                popup_text += f"<br>前の地点からの距離: {distance_segment:.2f} m"

            # スタート地点は赤、その他は青のマーカー
            icon_color = 'red' if i == 0 else 'blue'
            icon_name = 'home' if i == 0 else 'info-sign' # スタート地点は家のアイコン

            folium.Marker(
                location=current_coords,
                popup=popup_text,
                icon=folium.Icon(color=icon_color, icon=icon_name)
            ).add_to(m)

            prev_coords = current_coords

        # 最適ルートのラインを地図に追加
        points = optimal_route_info[['緯度', '経度']].values.tolist()
        folium.PolyLine(points, color='red', weight=2.5, opacity=1).add_to(m)

        # StreamlitでFolium地図を表示
        st_data = folium.Figure(width=700, height=500)
        m.add_to(st_data)
        st.components.v1.html(st_data._repr_html_(), width=720, height=520)
    #修正ここまで

if __name__ == "__main__":
    # main_app_logic関数を呼び出すように変更
    main_app_logic()