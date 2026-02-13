import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from models.pixelrnn import PixelRNN
import io

# -----------------------------------------
# Streamlit Page Configuration
# -----------------------------------------

st.set_page_config(page_title="PixelRNN Image Completion", layout="wide")
st.title("🧠 PixelRNN Image Completion")
st.markdown(
    """
    <p style="font-size:16px;">
    Upload an occluded image to see how the trained <b>PixelRNN</b> model reconstructs the missing regions.
    You can optionally upload the original image to compare reconstruction quality.
    </p>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------
# Load Trained Model
# -----------------------------------------

@st.cache_resource
def load_model(checkpoint_path="checkpoints/pixelrnn_epoch20.pth", device="cpu"):
    model = PixelRNN()
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        st.success(f"✅ Model loaded successfully from {checkpoint_path}")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("Please make sure the checkpoint file exists and is compatible.")
        return None
    model.eval()
    return model.to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.write(f"Using device: {device}")

model = load_model()

# -----------------------------------------
# Image Preprocessing Helpers
# -----------------------------------------

def preprocess_image(image: Image.Image, size=64):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

def tensor_to_image(tensor):
    tensor = tensor.squeeze(0).detach().cpu().clamp(0, 1)
    return transforms.ToPILImage()(tensor)

# -----------------------------------------
# Upload Section
# -----------------------------------------

st.sidebar.header("📂 Upload Images")
occluded_file = st.sidebar.file_uploader("Upload Occluded Image", type=["jpg", "png", "jpeg"])
original_file = st.sidebar.file_uploader("Upload Original Image (optional)", type=["jpg", "png", "jpeg"])

# -----------------------------------------
# Processing Section
# -----------------------------------------

if occluded_file and model is not None:
    occluded_img = Image.open(occluded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(occluded_img, caption="Occluded Input", use_container_width=True)

    with st.spinner("🧠 Reconstructing missing regions..."):
        try:
            input_tensor = preprocess_image(occluded_img).to(device)
            with torch.no_grad():
                output_tensor = model(input_tensor)
            reconstructed_img = tensor_to_image(output_tensor)
            st.success("✅ Reconstruction complete!")
        except Exception as e:
            st.error(f"❌ Error during reconstruction: {e}")
            st.stop()

    # -----------------------------------------
    # Display Results
    # -----------------------------------------
    st.subheader("🔍 Visual Comparison")

    if original_file:
        original_img = Image.open(original_file).convert("RGB")
        col1, col2, col3 = st.columns(3)
        col1.image(occluded_img, caption="Occluded Image", use_container_width=True)
        col2.image(reconstructed_img, caption="Reconstructed Output", use_container_width=True)
        col3.image(original_img, caption="Original Ground Truth", use_container_width=True)
        
        # -----------------------------------------
        # Side-by-Side Comparison (using Streamlit's native features)
        # -----------------------------------------
        st.markdown("### 🖼️ Side-by-Side Comparison")
        tab1, tab2 = st.tabs(["Original", "Reconstructed"])
        with tab1:
            st.image(original_img, use_container_width=True)
        with tab2:
            st.image(reconstructed_img, use_container_width=True)
            
    else:
        col1, col2 = st.columns(2)
        col1.image(occluded_img, caption="Occluded Image", use_container_width=True)
        col2.image(reconstructed_img, caption="Reconstructed Output", use_container_width=True)
        st.info("ℹ️ Upload the original image to enable comparison with ground truth.")

    # -----------------------------------------
    # Download Option
    # -----------------------------------------
    buf = io.BytesIO()
    reconstructed_img.save(buf, format="PNG")
    st.download_button(
        label="💾 Download Reconstructed Image",
        data=buf.getvalue(),
        file_name="reconstructed.png",
        mime="image/png"
    )

elif occluded_file and model is None:
    st.error("❌ Model failed to load. Please check the checkpoint file.")
else:
    st.info("👆 Please upload an occluded image from the sidebar to begin.")

# -----------------------------------------
# Additional Information
# -----------------------------------------
with st.sidebar.expander("ℹ️ About this app"):
    st.markdown("""
    This app uses a **PixelRNN** model to reconstruct missing regions in images.
    
    **How to use:**
    1. Upload an occluded image (with missing regions)
    2. Optionally upload the original for comparison
    3. View the reconstruction results
    4. Download the reconstructed image
    
    **Note:** The model works best with 64x64 pixel images and may resize your input.
    """)