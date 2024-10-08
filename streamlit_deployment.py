# streamlit run streamlit_deployment.py
import os
from pinecone import Pinecone
import streamlit as st
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from openai import OpenAI

st.set_page_config(layout="wide")


def open_ai_func():
    try:
        os.environ["OPENAI_API_KEY"] = st.secrets.get("openai_api_key", "")
    except Exception:
        os.environ["OPENAI_API_KEY"] = ""

    client = OpenAI()

    try:
        pc = Pinecone(api_key=st.secrets.get("pinecone_api_key", ""))
        index = pc.Index("movies")
    except Exception:
        index = None

    # AI helper functions

    def generate_blog(topic, additional_text):

        #prompt = f"""
        #You are a copy writer with years of experience writing impactful blogs that converge and help elevate brands.
        #Your task is to write a blog on any topic system provides to you. Make sure to write in a format that works for Medium.

        #Topic: {topic}
        #Additional pointers: {additional_text}
        #"""

        prompt = [
            {"role": "system", "content": "You are a copy writer with years of experience writing impactful blogs that converge and help elevate brands. Your task is to write a blog on any topic system provides to you. Make sure to write in a format that works for Medium."},
            {"role": "user", "content": f"{topic} : Additional pointers: {additional_text}"}
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            #prompt=prompt,
            messages=prompt,
            max_tokens=700,
            temperature=0.9
        )

        return response

    def generate_images(prompt, number_of_images):

        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=number_of_images,
            size="1024x1024",
            quality="standard",
        )

        return response

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
                st.text_area("Generated Blog", value=response.choices[0].message.content, height=700)

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
                    model="text-embedding-ada-002",
                    input=movie_description
                )

                result_vector = vector.data[0].embedding

                result = index.query(
                    vector=result_vector,
                    top_k=10,
                    include_metadata=True
                )

                for movie in result.matches:
                    st.write(movie["metadata"]["title"])

if os.path.isfile('./config.yaml'):
    with open('./config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
    )
else:
    authenticator = stauth.Authenticate(
        dict(st.secrets['credentials']),
        st.secrets['cookie']['name'],
        st.secrets['cookie']['key'],
        st.secrets['cookie']['expiry_days'],
    )

name, authentication_status, username = authenticator.login('main')

if authentication_status:
    authenticator.logout('Logout', 'main')
    st.write(f'Welcome *{name}*')
    open_ai_func()
elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')

