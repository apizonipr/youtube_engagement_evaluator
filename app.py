import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
from io import StringIO
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import wordcloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

st.set_page_config(
    page_title="Avaliador de Engajamento do YouTube",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main {
        background-color: rgb(250, 250, 250);
    }
    .stApp {
        background-color: rgb(250, 250, 250);
    }
    h1, h2, h3 {
        color: rgb(33, 33, 33);
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stProgress > div > div > div > div {
        background-color: rgb(255, 0, 0);
    }
    .stButton>button {
        background-color: rgb(255, 0, 0);
        color: white;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: rgb(200, 0, 0);
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

STOPWORDS_PT = {
    "a", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo", "as", "até",
    "com", "como", "da", "das", "de", "dela", "delas", "dele", "deles", "depois", "do",
    "dos", "e", "ela", "elas", "ele", "eles", "em", "entre", "era", "eram", "essa",
    "essas", "esse", "esses", "esta", "estas", "este", "estes", "eu", "isso", "isto",
    "já", "lhe", "lhes", "mais", "mas", "me", "mesmo", "meu", "meus", "minha", "minhas",
    "muito", "na", "não", "nas", "no", "nos", "nossa", "nossas", "nosso", "nossos", "num",
    "numa", "o", "os", "ou", "para", "pela", "pelas", "pelo", "pelos", "por", "pra", "qual",
    "quando", "que", "quem", "se", "sem", "seu", "seus", "sua", "suas", "também", "te",
    "tem", "tenho", "ter", "teu", "teus", "tu", "tua", "tuas", "um", "uma", "você", "vocês",
    "vos", "vou", "q", "vc", "tá", "to", "tbm", "pq", "mt", "tb", "ne", "ai", "sobre",
    "esse", "essa", "pro", "pros", "pras", "ta", "tipo", "igual", "mto", "cada", "fazer",
    "assim", "aqui", "agora", "depois", "antes", "desde", "dentro", "fora", "onde", "porque",
    "porquê", "quê", "quanto", "tudo", "nada", "todos", "todas", "todo", "toda", "outro",
    "outra", "outros", "outras", "mesma", "mesmas", "mesmo", "mesmos", "sendo", "sendo",
    "parte", "forma", "sendo", "apenas", "após", "até", "bem", "bom", "boa", "bons", "boas"
}

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)

URL_PATTERN = re.compile(
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)
MENTION_PATTERN = re.compile(r"@\w+")


def calcular_tempo_medio_resposta(datas):
    if len(datas) < 2:
        return None
    datas_ordenadas = sorted(datas)
    diferencas = [
        (datas_ordenadas[i + 1] - datas_ordenadas[i]).total_seconds()
        for i in range(len(datas_ordenadas) - 1)
    ]
    return np.mean(diferencas) / 3600


def detectar_coluna(df, candidatas):
    for col in df.columns:
        if col.strip().lower() in [c.lower() for c in candidatas]:
            return col
    for col in df.columns:
        if any(c.lower() in col.strip().lower() for c in candidatas):
            return col
    return None


def limpar_texto(texto):
    if not isinstance(texto, str):
        return ""
    texto = URL_PATTERN.sub("", texto)
    texto = texto.lower()
    texto = re.sub(r"[^\w\s\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def extrair_palavras(textos):
    todas = []
    for texto in textos:
        palavras = texto.split()
        for p in palavras:
            p = p.strip()
            if len(p) > 2 and p not in STOPWORDS_PT and not p.isdigit():
                todas.append(p)
    return todas


def calcular_pontuacao_engajamento(total_likes, total_comentarios, respostas, tempo_medio_resposta, palavras_positivas, total_palavras, inscritos=None):
    pontos = 0.0
    if inscritos and inscritos > 0:
        taxa_comentarios = (total_comentarios / inscritos) * 100
    else:
        taxa_comentarios = 0
    if taxa_comentarios >= 1:
        pontos += 25
    elif taxa_comentarios >= 0.5:
        pontos += 15
    elif taxa_comentarios > 0:
        pontos += 5
    if total_comentarios > 0 and total_likes > 0:
        taxa_like = total_likes / total_comentarios
    else:
        taxa_like = 0
    if taxa_like >= 3:
        pontos += 25
    elif taxa_like >= 1.5:
        pontos += 15
    elif taxa_like > 0:
        pontos += 5
    if total_comentarios > 0 and respostas > 0:
        taxa_resposta = (respostas / total_comentarios) * 100
    else:
        taxa_resposta = 0
    if taxa_resposta >= 50:
        pontos += 25
    elif taxa_resposta >= 20:
        pontos += 15
    elif taxa_resposta > 0:
        pontos += 5
    if tempo_medio_resposta is not None:
        if tempo_medio_resposta <= 1:
            pontos += 12.5
        elif tempo_medio_resposta <= 6:
            pontos += 7.5
        elif tempo_medio_resposta <= 24:
            pontos += 2.5
    else:
        pontos += 5
    if total_palavras > 0:
        proporcao_positiva = (palavras_positivas / total_palavras) * 100
    else:
        proporcao_positiva = 0
    if proporcao_positiva >= 40:
        pontos += 12.5
    elif proporcao_positiva >= 20:
        pontos += 7.5
    elif proporcao_positiva > 0:
        pontos += 2.5
    return min(pontos, 100)


def carregar_dados(arquivo):
    ext = arquivo.name.split(".")[-1].lower()
    try:
        if ext == "csv":
            df = pd.read_csv(arquivo)
        elif ext in ["xls", "xlsx"]:
            df = pd.read_excel(arquivo)
        else:
            st.error("Formato de arquivo não suportado. Use CSV ou Excel.")
            return None
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        return None
    return df


def padronizar_dataframe(df):
    col_comentario = detectar_coluna(df, ["comentario", "comentário", "comment", "texto", "text", "mensagem", "message"])
    col_autor = detectar_coluna(df, ["autor", "author", "usuário", "usuario", "user", "canal", "channel"])
    col_data = detectar_coluna(df, ["data", "date", "data_hora", "datahora", "datetime", "timestamp", "hora"])
    col_likes = detectar_coluna(df, ["curtidas", "curtida", "likes", "like", "gostei"])
    col_resposta = detectar_coluna(df, ["resposta", "reply", "respostas", "replies"])
    if not col_comentario:
        st.error("Não foi possível detectar a coluna de comentários. Colunas disponíveis: " + ", ".join(df.columns))
        return None
    df_padrao = pd.DataFrame()
    df_padrao["comentario"] = df[col_comentario].astype(str)
    if col_autor:
        df_padrao["autor"] = df[col_autor].astype(str)
    else:
        df_padrao["autor"] = "Autor desconhecido"
    if col_data:
        df_padrao["data"] = pd.to_datetime(df[col_data], errors="coerce")
    else:
        df_padrao["data"] = pd.NaT
    if col_likes:
        df_padrao["curtidas"] = pd.to_numeric(df[col_likes], errors="coerce").fillna(0).astype(int)
    else:
        df_padrao["curtidas"] = 0
    if col_resposta:
        df_padrao["resposta"] = df[col_resposta].fillna("").astype(str).str.len() > 0
    else:
        df_padrao["resposta"] = False
    return df_padrao


def contar_emojis(textos):
    emojis = []
    for t in textos:
        emojis.extend(EMOJI_PATTERN.findall(t))
    return Counter(emojis)


def contar_mentions(textos):
    contagem = 0
    for t in textos:
        contagem += len(MENTION_PATTERN.findall(t))
    return contagem


def classificar_sentimento(texto):
    palavras_positivas = {
        "bom", "boa", "ótimo", "otimo", "ótima", "otima", "excelente", "maravilhoso",
        "maravilhosa", "incrível", "incrivel", "fantástico", "fantastico", "fantástica",
        "fantastica", "legal", "gostei", "amei", "adoro", "adorei", "parabéns", "parabens",
        "show", "top", "massa", "demais", "perfeito", "perfeita", "sensacional", "bravo",
        "brava", "magnífico", "magnifico", "lindo", "linda", "bonito", "bonita", "sensacional",
        "recomendo", "recomendado", "inspirador", "inspiradora", "motivador", "motivadora",
        "útil", "util", "interessante", "educativo", "educativa", "aprendi", "entendi",
        "valeu", "obrigado", "obrigada", "grato", "grata", "excelente", "genial", "esperto",
        "esperta", "inteligente", "sábio", "sabia", "profissional", "competente", "capricho",
        "caprichado", "caprichada", "diferenciado", "diferenciada", "impressionante", "surpreendente"
    }
    palavras_negativas = {
        "ruim", "ruins", "péssimo", "pessimo", "péssima", "pessima", "horrível", "horrivel",
        "terrível", "terrivel", "odiei", "odeio", "detesto", "detestei", "chato", "chata",
        "tedioso", "tediosa", "sem graça", "sem-graça", "semgraca", "sem-graca", "fraco",
        "fraca", "pobre", "ruindade", "lixo", "merda", "bosta", "estupido", "estúpido",
        "estupida", "estúpida", "idiota", "burro", "burra", "ignorante", "preconceituoso",
        "preconceituosa", "racista", "machista", "fascista", "homofóbico", "homofobico",
        "homofóbica", "homofobica", "desrespeitoso", "desrespeitosa", "arrogante", "prepótente",
        "prepotente", "maldoso", "maldosa", "cínico", "cinico", "cínica", "cinica", "falso",
        "falsa", "mentiroso", "mentirosa", "enganador", "enganadora", "desonesto", "desonesta",
        "manipulador", "manipuladora", "propaganda", "spam", "irrelevante", "perda de tempo",
        "clicbait", "clickbait", "enganoso", "enganosa", "falácia", "falacia", "mentira",
        "errado", "errada", "triste", "decepcionante", "decepcionado", "decepcionada", "fracasso"
    }
    tokens = texto.split()
    positivos = sum(1 for p in tokens if p in palavras_positivas)
    negativos = sum(1 for p in tokens if p in palavras_negativas)
    if negativos > positivos:
        return "negativo"
    elif positivos > negativos:
        return "positivo"
    else:
        return "neutro"


def main():
    st.title("📊 Avaliador de Engajamento do YouTube")
    st.markdown("Faça upload de um arquivo CSV ou Excel com os comentários de um vídeo do YouTube e receba uma análise completa de engajamento.")

    with st.sidebar:
        st.header("Configurações")
        arquivo = st.file_uploader("Carregar arquivo de comentários", type=["csv", "xlsx", "xls"])
        inscritos = st.number_input("Número aproximado de inscritos do canal (opcional)", min_value=0, value=0, step=100)
        st.markdown("---")
        st.markdown("**Sobre**")
        st.markdown("Esta ferramenta analisa comentários, curtidas, respostas e sentimentos dos usuários para calcular uma pontuação de engajamento.")

    if not arquivo:
        st.info("Por favor, carregue um arquivo de comentários para começar a análise.")
        st.stop()

    df_raw = carregar_dados(arquivo)
    if df_raw is None:
        st.stop()

    df = padronizar_dataframe(df_raw)
    if df is None:
        st.stop()

    total_comentarios = len(df)
    total_likes = int(df["curtidas"].sum())
    respostas = int(df["resposta"].sum())
    datas_validas = df["data"].dropna()
    tempo_medio_resposta = calcular_tempo_medio_resposta(datas_validas.tolist()) if not datas_validas.empty else None

    df["texto_limpo"] = df["comentario"].apply(limpar_texto)
    df["sentimento"] = df["texto_limpo"].apply(classificar_sentimento)

    palavras = extrair_palavras(df["texto_limpo"].tolist())
    palavras_counter = Counter(palavras)
    total_palavras = len(palavras)

    emojis_counter = contar_emojis(df["comentario"].tolist())
    mentions = contar_mentions(df["comentario"].tolist())

    sentimentos = df["sentimento"].value_counts().to_dict()
    positivos = sentimentos.get("positivo", 0)
    negativos = sentimentos.get("negativo", 0)
    neutros = sentimentos.get("neutro", 0)

    pontuacao = calcular_pontuacao_engajamento(
        total_likes,
        total_comentarios,
        respostas,
        tempo_medio_resposta,
        positivos,
        total_comentarios,
        inscritos if inscritos > 0 else None,
    )

    st.markdown("---")
    st.subheader("Resumo da Análise")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Comentários", total_comentarios)
    with col2:
        st.metric("Curtidas Totais", total_likes)
    with col3:
        st.metric("Respostas", respostas)
    with col4:
        st.metric("Menções @", mentions)

    col5, col6, col7 = st.columns(3)
    with col5:
        st.metric("Pontuação de Engajamento", f"{pontuacao:.1f}/100")
    with col6:
        if tempo_medio_resposta is not None:
            st.metric("Tempo Médio de Resposta", f"{tempo_medio_resposta:.1f}h")
        else:
            st.metric("Tempo Médio de Resposta", "N/A")
    with col7:
        st.metric("Sentimento Predominante", df["sentimento"].mode()[0].capitalize() if not df["sentimento"].mode().empty else "N/A")

    st.markdown("---")
    st.subheader("Distribuição de Sentimentos")
    sentimento_data = pd.DataFrame({
        "Sentimento": ["Positivo", "Negativo", "Neutro"],
        "Quantidade": [positivos, negativos, neutros],
    })
    if PLOTLY_AVAILABLE:
        fig = px.pie(
            sentimento_data,
            names="Sentimento",
            values="Quantidade",
            color="Sentimento",
            color_discrete_map={
                "Positivo": "green",
                "Negativo": "red",
                "Neutro": "gray",
            },
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots()
        ax.pie(
            sentimento_data["Quantidade"],
            labels=sentimento_data["Sentimento"],
            colors=["green", "red", "gray"],
            autopct="%1.1f%%",
        )
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Top 15 Palavras Mais Frequentes")
    if palavras_counter:
        top_palavras = pd.DataFrame(palavras_counter.most_common(15), columns=["Palavra", "Frequência"])
        if PLOTLY_AVAILABLE:
            fig2 = px.bar(
                top_palavras,
                x="Frequência",
                y="Palavra",
                orientation="h",
                color="Frequência",
                color_continuous_scale="Reds",
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            fig2, ax2 = plt.subplots()
            ax2.barh(top_palavras["Palavra"], top_palavras["Frequência"], color="red")
            ax2.invert_yaxis()
            st.pyplot(fig2)
    else:
        st.write("Não foram encontradas palavras suficientes para análise.")

    st.markdown("---")
    st.subheader("Top 10 Emojis Mais Usados")
    if emojis_counter:
        top_emojis = pd.DataFrame(emojis_counter.most_common(10), columns=["Emoji", "Frequência"])
        st.dataframe(top_emojis, use_container_width=True)
    else:
        st.write("Nenhum emoji encontrado nos comentários.")

    if WORDCLOUD_AVAILABLE and palavras_counter:
        st.markdown("---")
        st.subheader("Nuvem de Palavras")
        wc = wordcloud.WordCloud(
            width=800,
            height=400,
            background_color="white",
            colormap="Reds",
            stopwords=STOPWORDS_PT,
            max_words=100,
        ).generate_from_frequencies(palavras_counter)
        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wc, interpolation="bilinear")
        ax_wc.axis("off")
        st.pyplot(fig_wc)

    st.markdown("---")
    st.subheader("Tabela de Comentários")
    st.dataframe(df[["autor", "comentario", "curtidas", "data", "sentimento"]].sort_values("curtidas", ascending=False), use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Baixar relatório completo em CSV",
        data=csv,
        file_name=f"relatorio_engajamento_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
