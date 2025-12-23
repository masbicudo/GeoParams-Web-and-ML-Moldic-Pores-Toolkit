from importlib import reload
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import libs.data as data
import localizable_resources as lr
import json

# Set this to True to use local data
# for testing without loading from
# files that came from public repo.
ENABLE_PARTICIPANT_LEVEL_VIEWS = False

st.set_page_config(
        page_title=lr.str.app_title,
        page_icon="static/favicon.png"  # Local file, emoji, or URL
    )

import libs.plots as plots
reload(lr)
reload(plots)
reload(data)

from dotenv import load_dotenv
load_dotenv(".env", override=False)

# Load data from the directory (replace with your directory path)
df = data.load_data_from_files(debug=False, print=lambda x: st.text(x))

if not ENABLE_PARTICIPANT_LEVEL_VIEWS:
    st.info(
        "Running in anonymized public-data mode. "
        "Participant-level visualizations are disabled. "
        "Toggle this by setting ENABLE_PARTICIPANT_LEVEL_VIEWS to True in sl_app.py."
    )

st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            min-width: 400px;
            max-width: 400px;
        }
    </style>
    """, unsafe_allow_html=True)

st.sidebar.header(lr.str.select_plot_type)

plot_type_options = {
        "params_c_k_segment_porosity": lr.str.best_color_params_plot_title,
        "params_c_k_data_by_user": lr.str.best_color_params_user_data,
    }
if not ENABLE_PARTICIPANT_LEVEL_VIEWS:
    del plot_type_options["params_c_k_data_by_user"]
selected_label = st.sidebar.selectbox(
    lr.str.select_plot_type,
    options=list(plot_type_options.keys()),  # Internal values
    format_func=lambda x: plot_type_options[x]  # Displayed labels
)

def get_participant_df(df):
    return df[
        df["name"].notnull()
        & df["email"].notnull()
    ]

if selected_label == "params_c_k_segment_porosity":
    
    st.sidebar.header(lr.str.plot_options)

    # selected_experience_range = st.sidebar.slider(
    #     lr.str.select_experience_level_range,
    #     min_value=1,
    #     max_value=5,
    #     value=(3, 5),
    #     step=1
    # )

    selected_min_pore_size_range = st.sidebar.slider(
        lr.str.select_min_pore_size_range,
        min_value=100,
        max_value=10000,
        value=(480, 10000),
        step=10,
    )

    # Filter based on the selected experience range and min_pore_size range
    def filter_data(df, experience_range=None, min_pore_size_range=None):
        if experience_range is not None:
            min_exp, max_exp = experience_range
            df = df[(df['experience'] >= min_exp) & (df['experience'] <= max_exp)]
        if min_pore_size_range is not None:
            min_size, max_size = min_pore_size_range
            df = df[(df['min_pore_size'] >= min_size) & (df['min_pore_size'] <= max_size)]
        return df

    # Filter data based on user input
    filtered_df = filter_data(df, None, selected_min_pore_size_range)

    use_canceled_or_tests = st.sidebar.checkbox(
            lr.str.use_canceled_or_tests, value=False
        )

    show_means = st.sidebar.checkbox(lr.str.show_means, value=True)
    show_dispersion = st.sidebar.checkbox(lr.str.show_dispersions, value=True)
    show_aggregation = st.sidebar.checkbox(lr.str.show_aggregations, value=True)
    show_exp_levels = st.sidebar.multiselect(
            lr.str.select_experience_levels,
            options=[1, 2, 3, 4, 5],
            default=[1, 2, 3, 4, 5]
        )

    if not use_canceled_or_tests:
        if ENABLE_PARTICIPANT_LEVEL_VIEWS:
            filtered_df = filtered_df[
                ~filtered_df["canceled"]
                & ~filtered_df["name"].str.contains("Test", case=False, na=False)
            ]
        else:
            filtered_df = filtered_df[~filtered_df["canceled"]]

    filtered_df = filtered_df[filtered_df['experience'].isin(show_exp_levels)]

    # Plot the clicked points
    fig, ax = plt.subplots()

    plots.plot_best_color_params(
            ax, filtered_df,
            use_color_bar=False,
            show_means=show_means,
            show_dispersion=show_dispersion,
            show_exp_levels=show_exp_levels,
            show_aggregation=show_aggregation,
        )

    st.pyplot(fig)

    # Display the filtered data including the review column
    st.write(filtered_df)

    if ENABLE_PARTICIPANT_LEVEL_VIEWS:
        participant_filtered_df = get_participant_df(filtered_df)
        if participant_filtered_df.empty:
            st.warning(
                "Participant-level views are enabled, but no local identifiable data was found."
            )
        else:
            st.subheader(lr.str.users_stats)
            st.write(
                    participant_filtered_df.groupby("name", as_index=False)
                    .agg(
                        count=("folder_name", "size"),  # equivalent to COUNT(*)
                        folder_names=("folder_name", lambda x: json.dumps(x.tolist()))
                    )
                )

    # from transformers import pipeline

    # # Load a pre-trained summarization pipeline
    # summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # # Example review texts
    # review_texts = filtered_df["review"]

    # # Concatenate all reviews into a single string
    # all_reviews = " ; ".join(review_texts) + """ <SEP>
    # Os textos anteriores estão em português do Brasil e são revisões feitas por usuários
    # de uma aplicação voltada para visualização de recortes de lâminas petrográficas.
    # Os usuários interagem com a aplicação designando cortes de canais de cores ciano e preto,
    # através das coordenadas x e y, que indicam os locais em que clicam nas imagens.
    # Com a participação de múltiplos usuários, os dados coletados sobre os cortes dos canais
    # de cores são analisados de forma estatística, com a média e o desvio padrão calculados
    # para indicar os cortes mais representativos.
    # """

    # # Generate a summary
    # summary = summarizer(all_reviews, max_length=1500, min_length=50, do_sample=False)
    # st.text(summary[0]['summary_text'])


if ENABLE_PARTICIPANT_LEVEL_VIEWS and selected_label == "params_c_k_data_by_user":
    
    st.sidebar.header(lr.str.best_color_params_user_data)

    names = np.sort(df["name"][df["name"].notnull()].unique())

    selected_name = st.sidebar.selectbox(
            lr.str.select_user,
            options=names,  # Internal values
        )

    # Filter data based on user input
    participant_df = get_participant_df(df)
    filtered_df = participant_df[participant_df["name"] == selected_name]

    available_images = data.get_available_images(selected_name)
    filtered_images = [item for item in available_images
            if item.get("name", "").lower() == selected_name.lower()]

    # Display the filtered data including the review column
    st.write(filtered_df)
    
    for image_data in filtered_images:
        st.subheader(f"Session: {image_data.get('folder_name').replace('static/output', '')}")
        
        st.image(image_data.get("param_space_image")[:,:,::-1], caption="main_image.png")

        cols = st.columns(1 + len(image_data.get("tiles", [])))
        cols[0].image(image_data.get("cropped_image")[:,:,::-1], caption="cropped.jpg")
        for it, tile in enumerate(image_data.get("tiles", [])):
            cols[it+1].image(tile.get("image")[:,:,::-1], caption=f"Tile at ({tile['x']}, {tile['y']})")

st.text(lr.str.end_report)
