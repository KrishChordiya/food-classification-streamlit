import streamlit as st
from PIL import Image
import torch
from torchvision import transforms

classnames= ['pizza', 'risotto', 'steak','sushi']

device = "cuda" if torch.cuda.is_available() else "cpu"
simple_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
model = torch.load('model_SDG.pt').to(device)


st.markdown("<h1 style='text-align: center;'>Basic Food Classifier</h1>", unsafe_allow_html=True)

st.markdown("<h3>Example Images</h3>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.image('ex_pizza.jpg')
with col2:
    st.image('ex_risotto.jpg')
with col3:
    st.image('ex_steak.jpg')
with col4:
    st.image('ex_sushi.jpg')

img = st.file_uploader("Upload Image")
_, _, _, btn_col, _, _, _ = st.columns([1]*6+[1.18])
analyse_btn = btn_col.button("Classify")

if analyse_btn and img is not None:
    model.eval()
    with torch.inference_mode():
        print("Working")
        img_data = Image.open(img)
        transformed_img = simple_transform(img_data).unsqueeze(0).to(device)
        result = torch.argmax(model(transformed_img), 1).item()
        st.image(img)
        st.markdown(f"<h3 style='text-align: center;'>Above Image is of <span style='text-decoration:underline'>{classnames[result].capitalize()}</span></h3>", unsafe_allow_html=True)

