import streamlit as st
import requests
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from shapely.geometry import Polygon
import json
from bs4 import BeautifulSoup
import networkx as nx

with st.echo(code_location="below"):
    st.title("Финальный проект")

    st.header("Часть 1. Аварии")
    st.markdown(
        "В первой части своего финального проекта я хочу проанализировать данные по ДТП в городе Москве. ГИБДД предоставляет все данные по ДТП, если в нем были пострадавшие. Я возьму данные с сайта проекта 'Карта ДТП' (dtp-stat.ru), так как они уже сагрегировали разрозненные данные ГИБДД."
    )

    st.markdown(
        "Этот проект предоставляет готовые файлы .geojson, но они оказались битые и нечитаемые, поэтому я возьму данные прямо с их карты, найдя методы незадокументированного api."
    )

    r = requests.get(
        "https://cms.dtp-stat.ru/api/dtp_load/?year=2021&region_slug=moskva&format=json"
    )

    df = pd.DataFrame(r.json())
    st.code(
        'r = requests.get("https://cms.dtp-stat.ru/api/dtp_load/?year=2021&region_slug=moskva&format=json")',
        language="python",
    )
    st.write(df.head())

    st.markdown(
        "Заметим, что некоторые данные (погода, нарушения и т.д.) закодированы. Я нашел еще один незадокументированный метод API, который содержит словари с расшифровкой значений."
    )

    r = requests.get("https://cms.dtp-stat.ru/api/filters")
    st.code(
        'r = requests.get("https://cms.dtp-stat.ru/api/filters")', language="python"
    )
    st.write(r.json()[1])

    st.markdown("Я перекодирую значения в таблице для более удобного чтения.")

    @st.cache
    def df_map(df):
        for item in range(1, len(r.json())):
            if "key" in r.json()[item]:
                column_name = r.json()[item]["key"]
            else:
                column_name = r.json()[item]["name"]
            values = r.json()[item]["values"]

            a = [i["value"] for i in values]
            b = [i["preview"] for i in values]
            d = dict(zip(a, b))

            df[column_name] = df[column_name].apply(lambda x: list(pd.Series(x).map(d)))
        return df

    df = df_map(df)

    st.write(df.head())

    st.markdown(
        "Удалим ненужные столбцы и добавим свои `injured_ratio` и `dead_ratio`, которые будут показывать сколько людей пострадало и умерло от общего числа участников аварии. Также пора сделать этот датафрейм геодатафремом."
    )

    df = df.set_index("id")
    df.drop(["address", "street", "tags", "category"], axis=1, inplace=True)
    df["injured_ratio"] = df["injured"] / df["participants"]
    df["dead_ratio"] = df["dead"] / df["participants"]

    df[["lat", "lon"]] = df["point"].apply(pd.Series)
    gdf_gibdd = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]))
    gdf_gibdd.drop(["point"], axis=1, inplace=True)
    df.drop(["point"], axis=1, inplace=True)

    st.write(df.drop("geometry", axis=1))

    st.markdown("Переходим теперь к анализу данных и их визуализации.")
    st.map(df, zoom=9)
    st.markdown(
        "Посчитаем характеристики вещественных признаков. Я добавил одну нестандартную характеристику (RMS - Root Mean Square), чтобы добавить альтернативный взгляд на эти выборки."
    )

    gdf_gibdd.drop(["lat", "lon", "datetime"], axis=1, inplace=True)
    df.drop(["lat", "lon", "datetime"], axis=1, inplace=True)

    def RMS(sample):
        # Numpy here makes calculations much simpler and faster than vanilla python
        sample = np.array(sample)
        sample = sample**2
        return np.sqrt(np.sum(sample) / len(sample))

    num_cols = ["participants", "injured", "dead", "injured_ratio", "dead_ratio"]

    st.write(df[num_cols].agg(("mean", "max", "min", RMS)))
    # TODO remove thrid layer
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 15))
    fig.suptitle("Displots for numeric features")
    for i, column in enumerate(num_cols):
        sns.distplot(df[column], ax=axes[i // 3, i % 3])
        sns.ecdfplot(df[column], ax=axes[i // 3, i % 3])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        ("Probability density function", "Cumulative distribution function"),
        loc="upper right",
    )

    st.pyplot(fig)

    st.markdown("Теперь я выведу тепловую карту Москвы по количеству аварий")
    r = requests.get("https://www.dropbox.com/s/zktq4vif5c6jzh1/admin_level_8.geojson?dl=1", allow_redirects=True)
    open('admin_level_8.geojson', 'wb').write(r.content)
    @st.cache
    def plot_moscow(gdf_gibdd):
        gdf_gibdd = gdf_gibdd
        ### FROM: Homework 14 (Анна Денисова)
        with open("admin_level_8.geojson", encoding="utf-8") as f:
            a = json.load(f)
        with open("moscow.geojson", encoding="utf-8") as f:
            moscow_poly = Polygon(
                json.load(f)["features"][0]["geometry"]["coordinates"][0]
            )
        data = []
        for i in a["features"]:
            row = {"id": i["id"], "name": i["name"]}
            poly = Polygon(i["geometry"]["coordinates"][0][0])
            row["lat"] = poly.centroid.y
            row["lon"] = poly.centroid.x

            data.append(row)
        df = pd.DataFrame(data)
        df1 = pd.DataFrame([{"name": "Москва", "poly": moscow_poly}])
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]))
        gdf1 = gpd.GeoDataFrame(df1, geometry="poly")

        gdf_moscow = gdf.sjoin(gdf1, op="intersects", how="inner")

        data = []
        for i in a["features"]:
            row = {
                "id": i["id"],
                "name": i["name"],
                "poly": Polygon(i["geometry"]["coordinates"][0][0]),
            }
            data.append(row)
        ### END FROM

        _gdf = pd.DataFrame(data)

        gdf_moscow = gdf_moscow.set_index("id").join(_gdf.set_index("id"), how="inner")
        gdf_moscow.drop(
            ["name_left", "lat", "lon", "geometry", "index_right", "name_right"],
            axis=1,
            inplace=True,
        )
        gdf_moscow.reset_index(inplace=True)
        gdf_moscow.set_geometry("poly", inplace=True)

        gdf_gibdd = gdf_gibdd.sjoin(gdf_moscow, how="inner")

        gdf_gibdd = (
            gdf_gibdd.groupby("name")
            .count()[["participants"]]
            .rename({"participants": "count"}, axis=1)
        )
        gdf_gibdd = gdf_moscow.set_index("name").join(gdf_gibdd).reset_index()
        return gdf_gibdd

    gdf_gibdd = plot_moscow(gdf_gibdd)
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_gibdd.plot(column="count", legend=True, ax=ax)
    ax.set_title("Moscow heatmap (number of car accidents)")
    st.pyplot(fig)

    cat_list = list(set(df.columns) - set(num_cols))
    cat_list.remove("geometry")
    option = st.selectbox(
        "Посмотрим, как распределены данные по некоторым категориальным призанкам. Для этого выберете признак для агрегации:",
        cat_list,
    )
    if option:
        df.explode(option).groupby(by=option).sum()[["participants", "injured", "dead"]]

    st.markdown(
        "Теперь построим предсказательную модель. Это будет классификатор на основе рещающего дерева. Таргет переменная - есть ли погибшие в ДТП. Также нужно закодировать категориальные признаки."
    )

    df["is_dead"] = df["dead"] > 0

    df.drop(["injured", "dead", "severity"], axis=1, inplace=True)
    st.write(df.drop("geometry", axis=1).head())

    num_cols = ["participants"]
    cat_list.remove("severity")
    df = df[~df.index.duplicated(keep="first")]
    data = df[num_cols]

    for i in cat_list:
        temp = pd.get_dummies(df.explode(i)[i]).sum(level=0)
        data = data.join(temp)

    target = df["is_dead"]

    target = target.reindex(data.index)

    code = """clf = tree.DecisionTreeClassifier()
    
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.33, random_state=42)
    
    clf.fit(X_train, y_train)
    
    accuracy = accuracy_score(clf.predict(X_test), y_test)
    
    """
    st.code(code, language="python")
    clf = tree.DecisionTreeClassifier()

    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.33, random_state=42
    )

    clf.fit(X_train, y_train)

    accuracy = accuracy_score(clf.predict(X_test), y_test)

    st.markdown(f"Получаем точность классификатора `{accuracy}`")
    st.markdown(
        "Ниже представлены 5 самых важных причин, влиящих на то, что ДТП будет с погибшими (participants - это количество участников аварии):"
    )
    feature_importance = dict(zip(data.columns, clf.feature_importances_))
    feature_importance = [
        f"{k}: {v}"
        for k, v in sorted(
            feature_importance.items(), key=lambda item: item[1], reverse=True
        )
    ]
    st.write(feature_importance[:5])

    st.header("Часть 2. Метро")
    st.markdown(
        "Если после прошлой части Вы решили поменьше пользоваьтся личным транспортом, побольше общественным, то я предлагаю посмотреть на московское метро."
    )
    st.markdown(
        "В этой части я соберу данные по метро и отвечу на вопрос: за какое минимальное количество станций можно добраться от A до B. То есть, сколько минимально станций нужно проехать от станции A до станции B. Конечно, было бы интреснее минимизировать время поездки, но это отложим на другой раз."
    )
    st.markdown("Чтобы получить данные по метро, будем обрабатывать википедию:")
    
    soup = BeautifulSoup(requests.get("https://ru.wikipedia.org/wiki/%D0%A1%D0%BF%D0%B8%D1%81%D0%BE%D0%BA_%D1%81%D1%82%D0%B0%D0%BD%D1%86%D0%B8%D0%B9_%D0%9C%D0%BE%D1%81%D0%BA%D0%BE%D0%B2%D1%81%D0%BA%D0%BE%D0%B3%D0%BE_%D0%BC%D0%B5%D1%82%D1%80%D0%BE%D0%BF%D0%BE%D0%BB%D0%B8%D1%82%D0%B5%D0%BD%D0%B0").text)
    
    G = nx.Graph()
    
    uid ={}
    nodes = []
    current_line = "1"
    table = soup.find_all("table", {"class":"standard sortable"})[0].findChildren("tbody")[0]
    for i in table.findChildren("tr")[1:]:
        if (i.has_attr("class")):
            break
        block = i.findChildren("td")
        line = block[0]["data-sort-value"]
        sid = block[1].findChild("a")["href"]
        node_id = None
        if sid in uid:
            node_id = uid[sid]
        else:
            node_id = len(uid)
            uid[sid] = node_id
        if line == current_line:
            try:
                G.add_edge(node_id, nodes[-1])
            except:
                pass
            
        for j in block[3].findChildren("span"):
            if j.has_attr("title"):
                cnx_id = None
                if j.findChild("a")["href"] in uid:
                    cnx_id = uid[j.findChild("a")["href"]]
                else:
                    cnx_id = len(uid)
                    uid[j.findChild("a")["href"]] = cnx_id
                G.add_edge(node_id, cnx_id)
                
        nodes.append(node_id)
        current_line = line
    
    dot = nx.nx_pydot.to_pydot(G)
    st.graphviz_chart(dot.to_string())
    
    st.markdown("Я не Артемий Лебедев, но и Вы не Яндекс.")
    st.markdown("Теперь найдем кратчайщий путь от какой-ниюудь станции до любой другой. Например, от Кузбминок до Чкаловской.")
    
    st.markdown("### Спасибо за внимание!")
    
    
    
    
