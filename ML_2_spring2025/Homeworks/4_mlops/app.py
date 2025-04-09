import streamlit as st

st.title("fldlfdlfldf")
st.header("This is a header")
st.subheader("This is a subheader")
st.text("This is a text")
st.markdown("# This is a markdown header 1")
st.markdown("## This is a markdown header 2")
st.markdown("### This is a markdown header 3")
st.markdown("This is a markdown: *bold* **italic** `inline code` ~strikethrough~")
st.markdown("""This is a code block with syntax highlighting
```python
print("Hello world!")
```
""")
st.html(
    "image from url example with html: "
    "<img src='https://www.wallpaperflare.com/static/450/825/286/kitten-cute-animals-grass-5k-wallpaper.jpg' width=400px>",
)


st.write("Text with write")
st.write(range(10))