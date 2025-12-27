import streamlit as st
import requests
from bs4 import BeautifulSoup
import jieba
import re
from collections import Counter
import pyecharts.options as opts
from pyecharts.charts import WordCloud, Bar, Line, Pie, Radar, Scatter, HeatMap, Funnel
from streamlit_echarts import st_pyecharts
import numpy as np

# 定义停用词列表（基础版）
STOPWORDS = set([
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也',
    '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那',
    '他', '她', '它', '们', '而', '及', '与', '或', '对于', '关于', '通过', '为了', '因为',
    '所以', '但是', '如果', '就', '都', '只', '又', '还', '个', '位', '本', '该', '其',
    '将', '应', '可', '能', '所', '以', '之', '于', '也', '则', '且', '并', '即', '如'
])



def fetch_url_content(url):
    """抓取URL的文本内容"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        response.encoding = response.apparent_encoding or 'utf-8'

        soup = BeautifulSoup(response.text, 'html.parser')

        # 移除脚本和样式
        for script in soup(["script", "style"]):
            script.decompose()

        # 获取文本
        text = soup.get_text()
        # 清理文本
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        return text
    except Exception as e:
        st.error(f"抓取URL内容失败: {str(e)}")
        return None


def text_segmentation(text):
    """文本分词并过滤停用词"""
    # 清理非中文字符
    text = re.sub(r'[^\u4e00-\u9fa5\s]', '', text)

    # 分词
    words = jieba.lcut(text)

    # 过滤停用词和单字
    filtered_words = [
        word for word in words
        if len(word) > 1 and word not in STOPWORDS and word.strip()
    ]

    return filtered_words


def get_word_frequency(words, min_freq=1):
    """统计词频并过滤低频词"""
    word_counts = Counter(words)
    # 过滤低频词
    filtered_counts = {
        word: count for word, count in word_counts.items()
        if count >= min_freq
    }
    # 按词频排序
    sorted_counts = dict(sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True))
    return sorted_counts


def create_wordcloud(word_freq):
    """创建词云"""
    if not word_freq:
        return None

    data = list(word_freq.items())[:100]  # 取前100个词

    wc = (
        WordCloud()
            .add(series_name="词频", data_pair=data, word_size_range=[20, 100])
            .set_global_opts(
            title_opts=opts.TitleOpts(
                title="文本词云", title_textstyle_opts=opts.TextStyleOpts(font_size=20)
            ),
            tooltip_opts=opts.TooltipOpts(is_show=True),
        )
    )
    return wc


def create_chart(chart_type, word_freq, top_n=20):
    """创建不同类型的图表"""
    if not word_freq:
        return None

    # 取前N个词
    top_words = list(word_freq.items())[:top_n]
    words = [item[0] for item in top_words]
    counts = [item[1] for item in top_words]

    if chart_type == "柱状图":
        chart = (
            Bar()
                .add_xaxis(words)
                .add_yaxis("词频", counts)
                .reversal_axis()  # 横向显示
                .set_global_opts(
                title_opts=opts.TitleOpts(title="词频排名柱状图"),
                xaxis_opts=opts.AxisOpts(name="词频"),
                yaxis_opts=opts.AxisOpts(name="词汇"),
                datazoom_opts=[opts.DataZoomOpts(type_="slider")],
            )
        )

    elif chart_type == "折线图":
        chart = (
            Line()
                .add_xaxis(words)
                .add_yaxis("词频", counts, is_smooth=True)
                .set_global_opts(
                title_opts=opts.TitleOpts(title="词频排名折线图"),
                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-45)),
                datazoom_opts=[opts.DataZoomOpts(type_="slider")],
            )
        )

    elif chart_type == "饼图":
        chart = (
            Pie()
                .add("", list(zip(words, counts)))
                .set_global_opts(
                title_opts=opts.TitleOpts(title="词频占比饼图"),
                legend_opts=opts.LegendOpts(orient="vertical", pos_top="15%", pos_left="80%"),
            )
                .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c} ({d}%)"))
        )

    elif chart_type == "雷达图":
        # 雷达图最多显示8个维度
        radar_words = words[:8]
        radar_counts = counts[:8]
        max_count = max(radar_counts) if radar_counts else 1

        chart = (
            Radar()
                .add_schema(
                schema=[opts.RadarIndicatorOpts(name=word, max_=max_count) for word in radar_words],
                splitarea_opt=opts.SplitAreaOpts(is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)),
            )
                .add("词频", [radar_counts])
                .set_global_opts(title_opts=opts.TitleOpts(title="词频雷达图"))
        )

    elif chart_type == "散点图":
        chart = (
            Scatter()
                .add_xaxis(words)
                .add_yaxis("词频", counts)
                .set_global_opts(
                title_opts=opts.TitleOpts(title="词频散点图"),
                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-45)),
                datazoom_opts=[opts.DataZoomOpts(type_="slider")],
            )
        )

    elif chart_type == "热力图":
        # 构建热力图数据（简化版）
        heat_data = []
        for i, (word, count) in enumerate(top_words[:10]):  # 取前10个
            for j in range(count):
                if j < 10:  # 限制y轴范围
                    heat_data.append([i, j, count])

        chart = (
            HeatMap()
                .add_xaxis([str(i) for i in range(10)])
                .add_yaxis("词频", [str(i) for i in range(10)], heat_data)
                .set_global_opts(
                title_opts=opts.TitleOpts(title="词频热力图"),
                visualmap_opts=opts.VisualMapOpts(min_=1, max_=max(counts) if counts else 1),
            )
        )

    elif chart_type == "漏斗图":
        chart = (
            Funnel()
                .add("词频", list(zip(words, counts)))
                .set_global_opts(
                title_opts=opts.TitleOpts(title="词频漏斗图"),
                legend_opts=opts.LegendOpts(pos_left="left", orient="vertical"),
            )
                .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"))
        )

    else:
        chart = None

    return chart


# 主界面
def main():
    # 移除了set_page_config，避免旧版本不支持
    st.title("URL文本词频分析系统")

    # 侧边栏
    with st.sidebar:
        st.header("📊 图表筛选")
        chart_type = st.selectbox(
            "选择图表类型",
            ["词云", "柱状图", "折线图", "饼图", "雷达图", "散点图", "热力图", "漏斗图"],
            index=0
        )

        st.header("⚙️ 过滤设置")
        min_frequency = st.slider(
            "最低词频过滤",
            min_value=1,
            max_value=20,
            value=2,
            step=1,
            help="过滤掉出现次数低于此值的词汇"
        )

        top_n = st.slider(
            "显示排名前N的词汇",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            help="图表中显示的词汇数量"
        )

        st.info(
            """
            ### 使用说明
            1. 输入文章URL并点击分析
            2. 调整侧边栏参数过滤低频词
            3. 选择不同图表类型查看分析结果
            4. 查看词频排名前20的词汇列表
            """
        )

    # 主内容区
    col1, col2 = st.columns([2, 1])

    with col1:
        # URL输入
        url = st.text_input(
            "🔗 输入文章URL",
            placeholder="https://example.com/article.html",
            help="请输入可访问的文章URL地址"
        )

        # 分析按钮 - 完全移除type参数
        if st.button("🚀 开始分析"):
            if not url:
                st.warning("请输入有效的URL地址")
            else:
                with st.spinner("正在抓取URL内容..."):
                    # 抓取内容
                    text = fetch_url_content(url)
                    if text:
                        st.success("✅ URL内容抓取成功")

                        with st.spinner("正在分词和统计词频..."):
                            # 分词
                            words = text_segmentation(text)
                            if not words:
                                st.warning("未能提取到有效词汇")
                                return

                            # 统计词频
                            word_freq = get_word_frequency(words, min_frequency)

                            if not word_freq:
                                st.warning("过滤后无有效词汇，请降低最低词频阈值")
                                return

                            # 保存到session state
                            st.session_state['word_freq'] = word_freq
                            st.session_state['analysis_done'] = True

                            st.success(f"✅ 分析完成！共提取到 {len(word_freq)} 个有效词汇")

        # 显示图表
        if 'analysis_done' in st.session_state and st.session_state['analysis_done']:
            word_freq = st.session_state['word_freq']

            st.subheader(f"📈 {chart_type}展示")

            if chart_type == "词云":
                wc = create_wordcloud(word_freq)
                if wc:
                    st_pyecharts(wc, height="600px")
            else:
                chart = create_chart(chart_type, word_freq, top_n)
                if chart:
                    st_pyecharts(chart, height="600px")

    with col2:
        # 词频排名
        st.subheader("🏆 词频排名前20")

        if 'word_freq' in st.session_state and st.session_state['word_freq']:
            word_freq = st.session_state['word_freq']
            top_20 = list(word_freq.items())[:20]

            # 创建排名表格 - 简化样式，避免复杂HTML兼容性问题
            for i, (word, count) in enumerate(top_20, 1):
                st.write(f"**第{i}名:** {word} - {count}次")
        else:
            st.info("请输入URL并点击分析按钮查看词频排名")

    # 原始数据展示 - 移除use_container_width参数
    with st.expander("📋 查看完整词频数据"):
        if 'word_freq' in st.session_state and st.session_state['word_freq']:
            word_freq = st.session_state['word_freq']
            # 转换为DataFrame展示 - 移除use_container_width参数
            try:
                import pandas as pd
                df = pd.DataFrame(
                    list(word_freq.items()),
                    columns=["词汇", "出现次数"]
                )
                # 移除use_container_width参数，兼容旧版本
                st.dataframe(df)
            except ImportError:
                # 如果没有pandas，用普通方式展示
                st.write("完整词频数据（前50个）：")
                for idx, (word, count) in enumerate(list(word_freq.items())[:50]):
                    st.write(f"{idx + 1}. {word}: {count}次")
                if len(word_freq) > 50:
                    st.write(f"... 还有 {len(word_freq) - 50} 个词汇未显示")


if __name__ == "__main__":
    # 初始化session state
    if 'analysis_done' not in st.session_state:
        st.session_state['analysis_done'] = False
    if 'word_freq' not in st.session_state:
        st.session_state['word_freq'] = {}

    main()
