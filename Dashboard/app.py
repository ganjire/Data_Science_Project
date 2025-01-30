# HOW TO RUN THE DASHBOARD: in Terminal enter the following prompt
# streamlit run app.py


import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# Load model
model_path = "model/final_18.keras"
model = tf.keras.models.load_model(model_path)

# Temporary upload directory
upload_dir = "uploads"
os.makedirs(upload_dir, exist_ok=True)

# Mapping of class indices to class names
class_mapping = {
    0: "ARMA 1 with trend",
    1: "ARMA 1 without trend",
    2: "ARMA 2 with trend",
    3: "ARMA 2 without trend",
    4: "ARMA 3 with trend",
    5: "ARMA 3 without trend",
    6: "AR 1 with trend",
    7: "AR 1 without trend",
    8: "AR 2 with trend",
    9: "AR 2 without trend",
    10: "AR 3 with trend",
    11: "AR 3 without trend",
    12: "MA 1 with trend",
    13: "MA 1 without trend",
    14: "MA 2 with trend",
    15: "MA 2 without trend",
    16: "MA 3 with trend",
    17: "MA 3 without trend"
}

# Preprocess image for the model
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# Dashboard
st.title("Time Series: Image Processing")

file_upload = st.file_uploader("Upload an image or a CSV file", type=["png", "jpg", "jpeg", "csv"])

if file_upload:
    if file_upload.name.endswith(("png", "jpg", "jpeg")):
        st.subheader("Image Upload Successful")

        # Display the uploaded image
        image = Image.open(file_upload)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Process the image with the model
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)[0]

        # Show the top 5 predictions as a bar chart
        st.subheader("Top 5 Predictions")
        top_5_predictions_df = pd.DataFrame({
            "Class": [class_mapping[idx] for idx, probability in sorted(enumerate(predictions), key=lambda x: x[1], reverse=True)[:5]],
            "Probability": sorted(predictions, reverse=True)[:5]
        })
        st.bar_chart(top_5_predictions_df.set_index("Class"))

        # Show all classes with predictions as a table
        st.subheader("All Predictions")
        all_predictions_df = pd.DataFrame({
            "Class": [class_mapping[idx] for idx in range(len(predictions))],
            "Probability": predictions
        }).sort_values(by="Probability", ascending=False)
        st.table(all_predictions_df.style.format({"Probability": "{:.2%}"}))

    elif file_upload.name.endswith("csv"):
        st.subheader("CSV Upload Successful")
        csv_path = os.path.join(upload_dir, file_upload.name)
        with open(csv_path, "wb") as f:
            f.write(file_upload.getbuffer())
        
        # Convert CSV to diagrams
        st.subheader("Generated Diagrams")
        image_paths = plot_csv_to_images(csv_path)
        for img_path in image_paths:
            image = Image.open(img_path)
            st.image(image, caption=f"Diagram: {os.path.basename(img_path)}", use_container_width=True)

        # Process the diagrams with the model
        st.subheader("Process the Diagrams with the Model")
        for img_path in image_paths:
            image = Image.open(img_path)
            processed_image = preprocess_image(image)
            predictions = model.predict(processed_image)[0]

            st.write(f"Prediction for {os.path.basename(img_path)}:")
            prediction_table = sorted(
                [(class_mapping[idx], probability) for idx, probability in enumerate(predictions)],
                key=lambda x: x[1],
                reverse=True
            )
            st.table(pd.DataFrame(prediction_table, columns=["Class", "Probability"]).style.format({"Probability": "{:.2%}"}))



# Additional information
st.markdown("""
<div style="position: fixed; bottom: 10px; left: 10px; line-height: 1.2;">
    <h4>Time Series</h4>
    <p><b>Module:</b> Data Science Project</p>
    <p><b>Students: Moreno Gallo, Rebecca Ganjineh, Josephine Görner</b></p>

</div>
""", unsafe_allow_html=True)


