import streamlit as st
from image_to_embedding import ImageEmbeddingProcessor
from pathlib import Path
from PIL import Image
import os
import threading

# Page configuration
st.set_page_config(
    page_title="Image Search with CLIP",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'processor' not in st.session_state:
    with st.spinner("Loading CLIP model..."):
        st.session_state.processor = ImageEmbeddingProcessor()
    st.success("Model loaded successfully!")

# Title and description
st.title("🔍 Image Search with CLIP")
st.markdown("Search for images using natural language descriptions")
# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Indexing Section
st.sidebar.markdown("---")
st.sidebar.subheader("📁 Index Images")

# First define these variables
collection_name = st.sidebar.text_input("Collection Name", value="image_embeddings", key="collection_name_input")
qdrant_host = st.sidebar.text_input("Qdrant Host", value=os.environ.get("QDRANT_HOST", "localhost"), key="qdrant_host_input")
qdrant_port = st.sidebar.number_input("Qdrant Port", value=int(os.environ.get("QDRANT_PORT", "6333")), min_value=1, max_value=65535, key="qdrant_port_input")

folder_path = st.sidebar.text_input(
    "Folder Path to Index",
    value="./images",
    help="Enter the path to folder containing images to index"
)

if st.sidebar.button("🚀 Start Indexing", type="primary"):
    if folder_path and Path(folder_path).exists():
        with st.spinner(f"Indexing images from {folder_path}..."):
            try:
                st.session_state.processor.process_folder_to_qdrant(
                    folder_path=folder_path,
                    collection_name=collection_name,
                    qdrant_host=qdrant_host,
                    qdrant_port=qdrant_port
                )
                st.sidebar.success(f"✓ Successfully indexed images from {folder_path}")
            except Exception as e:
                st.sidebar.error(f"Indexing failed: {str(e)}")
    else:
        st.sidebar.error("Invalid folder path!")

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Search Settings")

num_results = st.sidebar.slider("Number of Results", min_value=1, max_value=200, value=10)

# Search interface
st.markdown("---")
col1, col2 = st.columns([3, 1])

with col1:
    query_text = st.text_input(
        "Enter your search query:",
        placeholder="e.g., aadhaar card, passport photo, person smiling...",
        key="search_query"
    )

with col2:
    st.write("")  # Spacing
    st.write("")  # Spacing
    search_button = st.button("🔍 Search", type="primary", use_container_width=True)

# Search and display results
if search_button and query_text:
    with st.spinner(f"Searching for '{query_text}'..."):
        try:
            # Perform search
            results = st.session_state.processor.search_by_text(
                query_text=query_text,
                collection_name=collection_name,
                qdrant_host=qdrant_host,
                qdrant_port=qdrant_port,
                limit=num_results
            )
            
            if not results:
                st.warning("No results found. Try a different query.")
            else:
                st.success(f"Found {len(results)} matching images!")
                st.markdown("---")
                
                # Display results in a grid
                cols_per_row = 3
                for i in range(0, len(results), cols_per_row):
                    cols = st.columns(cols_per_row)
                    
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx < len(results):
                            result = results[idx]
                            
                            with col:
                                # Try to load and display image
                                try:
                                    image_path = Path(result['image_path'])
                                    if image_path.exists():
                                        img = Image.open(image_path)
                                        st.image(img, use_container_width=True)
                                        
                                        # Display metadata
                                        st.markdown(f"**Rank:** {idx + 1}")
                                        st.markdown(f"**Score:** {result['score']:.4f}")
                                        st.markdown(f"**File:** `{result['filename']}`")
                                        
                                        # Expandable path
                                        with st.expander("View full path"):
                                            st.code(result['image_path'])
                                    else:
                                        st.error(f"Image not found: {result['filename']}")
                                except Exception as e:
                                    st.error(f"Error loading image: {e}")
                                
                                st.markdown("---")
                
        except Exception as e:
            st.error(f"Search failed: {str(e)}")
            st.exception(e)

elif search_button:
    st.warning("Please enter a search query.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <small>Powered by OpenAI CLIP and Qdrant Vector Database</small>
    </div>
    """,
    unsafe_allow_html=True
)
