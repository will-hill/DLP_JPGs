import google.auth
import google.cloud.storage
import streamlit as st
import pandas as pd
from itertools import cycle
from downcast import reduce
from PIL import ImageDraw
from PIL import Image
import os
import io

st.set_page_config(layout="wide")
# CONFIG ------------------------------------------------------------------------------------------
CREDENTIALS, PROJECT = google.auth.default()
BUCKET = f"{PROJECT}-dlp"
FILE_CATALOG_URL = os.getenv("FILE_CATALOG_URL")
BIG_QUERY_DATASET = "dlp"
BIG_QUERY_TABLE = "dlp-results"
TBL = f"`{PROJECT}.{BIG_QUERY_DATASET}.{BIG_QUERY_TABLE}`"
PARENT = f"projects/{PROJECT}/locations/global"
google_cloud_storage = google.cloud.storage.Client()
bucket = google_cloud_storage.bucket(BUCKET)
document_query_params = {
    "top_n": 100,
    "order_col": "ttl_risk",
    "order_type": "DESC",
    "pii_filter": ['PERSON_NAME', 'AGE', 'ETHNIC_GROUP', 'EMAIL_ADDRESS', 'PASSWORD', 'PHONE_NUMBER', 'STREET_ADDRESS'],
    "likelihood_filter": ['UNLIKELY', 'POSSIBLE', 'VERY_LIKELY', 'LIKELY']
}
# GIANT DOCUMENT QUERY  ------------------------------------------------------------------------------------------
convert_filter = lambda filter_list: "('" + "', '".join(filter_list) + "')"
get_giant_document_query = lambda query_params: f"""
    WITH summary AS (
    WITH quantized AS 
    (
      SELECT 
        quote,
        LENGTH(quote) AS quote_len,
        info_type.name AS pii_type,
        CASE info_type.name
          WHEN 'PERSON_NAME' THEN 0.1 
          WHEN 'FDA_CODE' THEN 0.3
          WHEN 'AGE' THEN 0.9
          WHEN 'ETHNIC_GROUP' THEN 0.8
          WHEN 'EMAIL_ADDRESS' THEN 0.7 
          WHEN 'PASSWORD' THEN 1.0 
          WHEN 'PHONE_NUMBER' THEN 0.6 
          WHEN 'STREET_ADDRESS' THEN 0.8 
          WHEN 'GERMANY_IDENTITY_CARD_NUMBER' THEN 1.0 
          WHEN 'US_SOCIAL_SECURITY_NUMBER' THEN 1.0
          WHEN 'GENDER' THEN 0.5 
          WHEN 'NETHERLANDS_BSN_NUMBER' THEN 1.0 
          WHEN 'DATE_OF_BIRTH' THEN 1.0 
          WHEN 'CANADA_SOCIAL_INSURANCE_NUMBER' THEN 1.0 
        ELSE 1.0
        END
        AS pii_score,
        likelihood,
        CASE likelihood
          WHEN 'VERY_LIKELY' THEN 0.90
          WHEN 'LIKELY' THEN 0.85
          WHEN 'POSSIBLE' THEN 0.60
          WHEN 'UNLIKELY' THEN 0.45
          WHEN 'VERY_UNLIKELY' THEN 0.30    
          ELSE 0.1
          END
          AS likelihood_score,
        location.content_locations[OFFSET(0)].container_name AS blob_path,
        location.content_locations[OFFSET(0)].image_location.bounding_boxes
      FROM 
        {TBL}
      WHERE 
          info_type.name in {convert_filter(query_params["pii_filter"])}
      AND
         likelihood in {convert_filter(query_params["likelihood_filter"])} 
      )
    SELECT 
      quote, quote_len, pii_type, pii_score, likelihood, likelihood_score, (quote_len * pii_score * likelihood_score) AS risk, blob_path, bounding_boxes 
    FROM 
      quantized
    )
    SELECT 
      blob_path, 
      COUNT(blob_path) AS pii_hits,

      SUM(summary.quote_len) AS ttl_quote_len,
      AVG(summary.quote_len) AS avg_quote_len,
      ARRAY_AGG(summary.quote) AS quote_ary,

      SUM(summary.pii_score) AS ttl_pii_score,
      AVG(summary.pii_score) AS avg_pii_score,

      SUM(summary.likelihood_score) AS ttl_likelihood_score,
      AVG(summary.likelihood_score) AS avg_likelihood_score,

      SUM(summary.risk) AS ttl_risk,
      AVG(summary.risk) AS avg_risk,
      MAX(summary.risk) AS max_risk,

      ARRAY_CONCAT_AGG(summary.bounding_boxes) AS bounding_boxes_ary

    FROM summary
    GROUP BY blob_path
    ORDER BY {query_params["order_col"]} {query_params["order_type"]}
    LIMIT {query_params["top_n"]}
    """

# DATA ------------------------------------------------------------------------------------------
data = dict()
data["hit_counts"] = pd.read_feather("hit_counts.ftr")
data["types_probas"] = pd.read_feather("probas.ftr")
data["types_probas_ttl"] = pd.read_feather("types_probas_ttl.ftr")
data["doc_counts"] = pd.read_feather("docs_hits.ftr")[["file_name", "hits"]].sort_values("hits", ascending=False)
types = data["hit_counts"]['name']
probas = data["types_probas"]['likelihood']

# STYLE ------------------------------------------------------------------------------------------
blue = "#4285F4"
red = "#DB4437"
yellow = "#F4B400"
green = "#0F9D58"
grey = "grey"
google_palette = [blue, red, yellow, green, grey]
zip_cat_colors = lambda cat, colors: dict(zip(cat, cycle(colors)) if len(cat) > len(colors) else zip(cycle(cat), colors))
color_map = {**zip_cat_colors(types, google_palette), **zip_cat_colors(probas, google_palette)}
highlight_cells = lambda x: 'background-color: ' + x.map(color_map)

# FILTER ------------------------------------------------------------------------------------------
type_filter_map = dict()
proba_filter_map = dict()
with st.sidebar:
    st.header("Query Limit")
    top_n = st.select_slider("LIMIT n", options=[1, 10, 100, 1000, 10000], value=100, format_func=str, key="limit_n")
    document_query_params["top_n"] = top_n

    st.header("PII Filter")
    st.caption("Filter by PII Type")
    for pii in types:
        type_filter_map[pii] = st.checkbox(pii, key=pii, value=True)

    st.header("Likelihood Filter")
    st.caption("Filter by Likelihood")
    for proba in ["VERY_LIKELY", "LIKELY", "POSSIBLE", "UNLIKELY", "VERY_UNLIKELY"]:
        proba_filter_map[proba] = st.checkbox(proba, key=proba, value=True)

pii_types = [t for t in types if t in [i[0] for i in type_filter_map.items() if i[1]]]
document_query_params["pii_filter"] = pii_types
name_filter_query = f"name in {pii_types}"

likelihoods = [i[0] for i in proba_filter_map.items() if i[1]]
likelihood_filter_query = f"likelihood in {likelihoods}"
document_query_params["likelihood_filter"] = likelihoods

for tbl in data:
    if 'name' in data[tbl].columns:
        data[tbl] = data[tbl].query(name_filter_query)
    if 'likelihood' in data[tbl].columns:
        data[tbl] = data[tbl].query(likelihood_filter_query)

# HEADER ------------------------------------------------------------------------------------------
left_col, dlp_col = st.columns([2, 1])

with left_col:
    st.image("DLP_BIG.png", width=850, )
with dlp_col:
    st.title("Data Loss Prevention", anchor="top")
    st.image("https://fonts.gstatic.com/s/i/gcpiconscolors/data_loss_prevention_api/v1/24px.svg", width=200, )

# KPIs ------------------------------------------------------------------------------------------
st.markdown(f"""---""")
st.title("KPIs")

pii_col, types_probas_col, type_proba_hits_col, hits_col, metrics_col = st.columns([3, 2, 3, 3, 1])
df_height = 400
df_width = 400
df_row = 10

with pii_col:
    st.header("PII by Type")
    st.caption("All PII Types and the total number of findings")
    st.dataframe(data["hit_counts"].head(df_row).style.apply(highlight_cells).background_gradient(cmap='RdYlGn_r', vmax=200, vmin=1), height=df_height, width=df_width)
    st.caption("hit_counts")

with types_probas_col:
    st.header("Count of Likelihoods")
    st.caption("All Likelihoods and their total findings")
    st.dataframe(data["types_probas"].head(df_row).style.apply(highlight_cells).background_gradient(cmap='RdYlGn_r', vmax=388388, vmin=30456, low=0.2, high=0.99), height=df_height, width=df_width)
    st.caption("types_probas")

with type_proba_hits_col:
    st.header("Count of PII x Likelihoods")
    st.caption("Count of All PII Types and the association Likelihoods")
    st.dataframe(data["types_probas_ttl"].head(df_row).style.apply(highlight_cells).background_gradient(cmap='RdYlGn_r', vmax=1904, vmin=1), height=df_height,
                 width=df_width)  # .style.applymap(type_color, subset=["name"]))
    st.caption("types_probas_ttl")

with hits_col:
    st.header("Docs by PII Count")
    st.caption("All Documents and the total number of PII findings in each document")
    st.dataframe(data["doc_counts"].head(df_row).style.apply(highlight_cells).background_gradient(cmap='RdYlGn_r'), height=df_height)  # .style.applymap(type_color, subset=["name"]))
    st.caption("doc_counts")

# REPORT ------------------------------------------------------------------------------------------
bq_documents_query = get_giant_document_query(document_query_params)
document_report_df = pd.read_gbq(bq_documents_query, use_bqstorage_api=True)
document_report_df.to_feather("document_report_df.ftr")
cols = list(document_report_df.columns)
cols.remove("quote_ary")
cols.remove("bounding_boxes_ary")
document_report_df = document_report_df[cols]
document_report_df = reduce(document_report_df)

st.markdown(f"""---""")
st.title(f"Document Risk Report ({document_report_df.shape[0]})")

st.dataframe(document_report_df.style.background_gradient(cmap='RdYlGn_r'))

# REDACT ------------------------------------------------------------------------------------------
st.title("Images")
if st.button('Show Images', ):
    n_images = 100
    show = 109
    shown = 0
    top_rows = pd.read_feather("document_report_df.ftr").head(n_images)
    for i in range(n_images):

        if i >= top_rows.shape[0]:
            break
        row = top_rows.iloc[[i]]
        bbs = list(row["bounding_boxes_ary"].values[0])
        if len(bbs) < 1:
            continue

        st.markdown(f"""---""")
        st.header(f"{shown + 1}.")
        document_image = Image.open(io.BytesIO(bucket.blob(row["blob_path"].values[0].split(f"gs://{BUCKET}/")[1]).download_as_string()))
        raw_col, bb_col, redact_col = st.columns(3)
        with raw_col:
            st.image(document_image)
        with bb_col:
            document_image = document_image.convert("RGB")
            draw = ImageDraw.Draw(document_image, "RGBA")
            for bb in bbs:
                bb_left = 0 if bb["left"] is None else bb["left"]
                draw.rectangle([bb_left, bb["top"], bb_left + bb["width"], bb["top"] + bb["height"], ], fill=(200, 100, 0, 64))
                draw.rectangle([bb_left, bb["top"], bb_left + bb["width"], bb["top"] + bb["height"], ], outline="red", width=7)
            st.image(document_image)
        with redact_col:
            document_image = document_image.convert("RGB")
            draw = ImageDraw.Draw(document_image, "RGB")
            for bb in bbs:
                bb_left = 0 if bb["left"] is None else bb["left"]
                draw.rectangle([bb_left, bb["top"], bb_left + bb["width"], bb["top"] + bb["height"], ], fill="black")
            st.image(document_image)
            shown += 1
        if shown >= show:
            break
