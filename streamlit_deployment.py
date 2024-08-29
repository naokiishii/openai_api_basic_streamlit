# streamlit run streamlit_deployment.py
import os
from pinecone import Pinecone
import streamlit as st

from openai import OpenAI

os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
client = OpenAI()

pc = Pinecone(api_key=st.secrets["pinecone_api_key"])
index = pc.Index("movies")

# AI helper functions


def generate_blog(topic, additional_text):

    prompt = f"""
    You are a copy writer with years of experience writing impactful blogs that converge and help elevate brands.
    Your task is to write a blog on any topic system provides to you. Make sure to write in a format that works for Medium.
    
    Topic: {topic}
    Additiona pointers: {additional_text}
    """

    response = client.completions.create(
        model = "gpt-3.5-turbo-instruct",
        prompt = prompt,
        max_tokens = 700,
        temperature = 0.9
    )

    return response

def generate_images(prompt, number_of_images):

    response = client.images.generate(
        prompt = prompt,
        n = number_of_images,
        size = "512x512"
    )

    return response

st.set_page_config(layout="wide")

st.title("OpenAI API Demo Webapp")

st.sidebar.title("AI Apps")

ai_app = st.sidebar.radio("Choose an AI App", ("Blog Generator", "Image Generator", "Movie Recommender"))

if ai_app == "Blog Generator":
    st.header("Blog Generator")
    st.write("Input a topic to generate a blog about it using OpenAI API")

    topic = st.text_area("Topic", height=30)
    additional_text = st.text_area("Additional Text", height=30)

    if st.button("Generate Blog"):
        with st.spinner("Generating..."):
            response = generate_blog(topic, additional_text)
            st.text_area("Generated Blog", value = response.choices[0].text, height = 700)

elif ai_app == "Image Generator":
    st.header("Image Generator")
    st.write("Add a prompt to generate an image using OpenAI API and DALLE model")

    prompt = st.text_area("Prompt", height=30)
    number_of_images = st.slider("Number of images", 1, 5, 1)

    if st.button("Generate Image") and prompt != "":
        with st.spinner("Generating..."):
            response = generate_images(prompt, number_of_images)

            for output in response.data:
                st.image(output.url)

elif ai_app == "Movie Recommender":
    st.header("Movie Recommender")
    st.write("Describe a movie that you would like to see")

    movie_description = st.text_area("Movie Description", height=30)

    if st.button("Get Movie Recommendation") and movie_description != "":
        with st.spinner("Generating..."):
            vector = client.embeddings.create(
                model = "text-embedding-ada-002",
                input = movie_description
            )

            result_vector = vector.data[0].embedding

            result = index.query(
                vector = result_vector,
                top_k = 10,
                include_metadata=True
            )

            for movie in result.matches:
                st.write(movie["metadata"]["title"])