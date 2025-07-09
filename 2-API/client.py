import streamlit as st
import requests

# Page configuration
st.set_page_config(page_title="AI Blog & Poem Generator", layout="centered")

# Header
st.title("🧠 AI Blog & Poem Generator")
st.markdown("Get a creative blog or a poem in seconds! Powered by AI ✨")

# Input
user_input = st.text_area("✍️ Enter a topic:", height=100)



# Button 1: Generate Blog

if st.button("📝 Generate Blog"):
    
        with st.spinner("Generating blog..."):
            try:
                response = requests.post("http://localhost:8000/blog", json={"topic": user_input})
                blog = response.json()["result"]
                st.success("✅ Blog generated successfully!")
                st.markdown("### 📖 Blog Output:")
                st.write(blog)
            except Exception as e:
                st.error(f"❌ Failed to get blog: {e}")
                
                
            if blog:
                st.download_button(
                    label="📥 Download Blog as .txt",
                    data=blog,
                    file_name="generated_blog.txt",
                    mime="text/plain"
                )            

# Button 2: Generate Poem

if st.button("🎤 Generate Poem"):
        with st.spinner("Generating poem..."):
            try:
                response = requests.post("http://localhost:8000/poem", json={"topic": user_input})
                poem = response.json()["result"]
                st.success("✅ Poem generated successfully!")
                st.markdown("### 🎶 Poem Output:")
                st.write(poem)
            except Exception as e:
                st.error(f"❌ Failed to get poem: {e}")
                
            if poem:
                st.download_button(
                    label="📥 Download Poem as .txt",
                    data=poem,
                    file_name="generated_poem.txt",
                    mime="text/plain"
                )            



# Footer
st.markdown("---")
st.markdown("<center><sub>Made By ❤️Abhishek Jyotiba using Streamlit & LangChain</sub></center>", unsafe_allow_html=True)
