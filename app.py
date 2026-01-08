import streamlit as st
import requests
import torch
import numpy as np

st.title("Data to Tensor Converter")
st.write("This app pulls API data and converts it into a PyTorch Tensor for AI models.")

if st.button('Fetch and Process Data'):
    url = "https://jsonplaceholder.typicode.com/posts"
    response = requests.get(url)
    data = response.json()

    # Process data
    features = [[item['userId'], item['id']] for item in data]
    data_array = np.array(features, dtype=np.float32)
    data_tensor = torch.from_numpy(data_array)

    # Display on Website
    st.success(f"Successfully fetched {len(data)} items!")
    
    col1, col2 = st.columns(2)
    col1.metric("Tensor Shape", str(list(data_tensor.shape)))
    col2.metric("Device", str(data_tensor.device))

    st.subheader("Raw Tensor Output (First 10 rows)")
    st.code(data_tensor[:10])
streamlit>=1.30.0
requests>=2.31.0
torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu
numpy>=1.24.0
