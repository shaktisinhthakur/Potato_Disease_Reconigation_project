import base64
import streamlit as st
import tensorflow as tf
import numpy as np
import plotly.express as px

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img = get_img_as_base64("potato_model/first_1.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://wallpaperaccess.com/full/5721509.jpg");
background-size: 180%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-position: center ; 
background-repeat: repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

def model_prediction(test_image):
    model = tf.keras.models.load_model('potato_model/1.keras')
    image = tf.keras.preprocessing.image.load_img(test_image,target_size =( 256, 256))
    input_arr= tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

st.sidebar.markdown('<h2 style="color: red;">Dashboard</h2>', unsafe_allow_html=True)
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown('<h1 style="color: darkorange;">POTATO DISEASE RECOGNITION SYSTEM</h1>', unsafe_allow_html=True)


if(app_mode=='Home'):
    image_path = "C:potato_model\image _not.JPG"
    st.image(image_path,use_column_width=True)
    st.markdown("""
<span style="color: darkorange; font-size: 20px;">Welcome to the Potato Disease Recognition System! 🌿🔍</span>

<span style="color: red; font-size: 20px;">Our mission is to help in identifying Potato diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!</span>
  ### <span style="color: darkorange;">How It Works</span>""", unsafe_allow_html=True)

    st.markdown(""" :red[1.  Upload Image:] **:orange[Go to the ]** **:orange[Disease Recognition]** **:orange[page and upload an image of a plant with suspected diseases.]**""")
    st.markdown(""" :red[2.  Analysis:] **:orange[Our system will process the image using advanced algorithms to identify potential diseases.]**""")
    st.markdown(""" :red[3.  Results:] **:orange[View the results and recommendations for further action.]**
 ### <span style="color: darkorange;">Why Choose Us</span> """, unsafe_allow_html=True)
   
    st.markdown("""- :red[Accuracy:] **:orange[Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.]**""")
    st.markdown("""- :red[User-Friendly:] **:orange[Simple and intuitive interface for seamless user experience.]**""")
    st.markdown("""- :red[Fast and Efficient:] **:orange[receive results in seconds, allowing for quick decision-making.]**

    ### <span style="color: darkorange;">Get Started</span>""", unsafe_allow_html=True)
    st.markdown("""- :red[Click on the [Disease Recognition] page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!]

    ### <span style="color: darkorange;">About Us</span>""", unsafe_allow_html=True)
    st.markdown(""":red[Learn more about the project, our team, and our goals on the **About** page.]""")

   

    #Aboout Page
elif(app_mode=="About"):

    st.markdown("""
            ### <span style="color: darkorange;">About Us</span>""", unsafe_allow_html=True)
    st.markdown(""":red[This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.]""")
    st.markdown(""":red[This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.]""")
    st.markdown(""":red[A new directory containing 33 test images is created later for prediction purpose.]""")
    st.markdown(""" ### <span style="color: darkorange;">Content</span>""", unsafe_allow_html=True)
    st.markdown(""" :red[1. train]
                            **:red[2. test]** 
                            **:red[3. validation ]**
""")
    
 # prediction page
elif(app_mode=="Disease Recognition"):
    st.header(":red[Disease Recognition]")
    test_image = st.file_uploader(":red[Choose an Image:]")
    if(st.button(":red[Show Image]")):
        st.image(test_image,use_column_width=True)
     
     #Predict Button
    if(st.button("predict")):
        st.balloons()
        st.write(":red[Our Prediction]")
        result_index = model_prediction(test_image)
        
        #define class
        class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
        st.markdown(f'<h4 style="color: orange;">Model is Predicting it\'s a {class_names[result_index]}</h4>', unsafe_allow_html=True)