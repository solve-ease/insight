import streamlit as st
from image_to_embedding import ImageEmbeddingProcessor
from video_to_embedding import VideoEmbeddingProcessor
from pathlib import Path
from PIL import Image
import os
import threading

# Page configuration
st.set_page_config(
    page_title="Image & Video Search with CLIP",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'image_processor' not in st.session_state:
    with st.spinner("Loading CLIP model for images..."):
        st.session_state.image_processor = ImageEmbeddingProcessor()
    st.success("Image model loaded successfully!")

if 'video_processor' not in st.session_state:
    with st.spinner("Loading CLIP model for videos..."):
        st.session_state.video_processor = VideoEmbeddingProcessor()
    st.success("Video model loaded successfully!")

# Title and description
st.title("🔍 Image & Video Search with CLIP")
st.markdown("Search for images and videos using natural language descriptions")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Search mode selection
st.sidebar.markdown("---")
search_mode = st.sidebar.radio(
    "Search Mode",
    ["Images", "Videos", "Both"],
    help="Choose whether to search images, videos, or both"
)

# Qdrant configuration
st.sidebar.subheader("🗄️ Database Settings")
qdrant_host = st.sidebar.text_input("Qdrant Host", value=os.environ.get("QDRANT_HOST", "localhost"), key="qdrant_host_input")
qdrant_port = st.sidebar.number_input("Qdrant Port", value=int(os.environ.get("QDRANT_PORT", "6333")), min_value=1, max_value=65535, key="qdrant_port_input")

# Image collection settings
image_collection_name = st.sidebar.text_input("Image Collection Name", value="image_embeddings", key="image_collection_name_input")

# Video collection settings
video_collection_name = st.sidebar.text_input("Video Collection Name", value="video_embeddings", key="video_collection_name_input")

# Indexing Section
st.sidebar.markdown("---")
st.sidebar.subheader("📁 Index Content")

# Indexing Section
st.sidebar.markdown("---")
st.sidebar.subheader("📁 Index Content")

# Image indexing
image_folder_path = st.sidebar.text_input(
    "Image Folder Path",
    value="./images",
    help="Enter the path to folder containing images to index"
)

if st.sidebar.button("🚀 Index Images", type="primary"):
    if image_folder_path and Path(image_folder_path).exists():
        with st.spinner(f"Indexing images from {image_folder_path}..."):
            try:
                st.session_state.image_processor.process_folder_to_qdrant(
                    folder_path=image_folder_path,
                    collection_name=image_collection_name,
                    qdrant_host=qdrant_host,
                    qdrant_port=qdrant_port
                )
                st.sidebar.success(f"✓ Successfully indexed images from {image_folder_path}")
            except Exception as e:
                st.sidebar.error(f"Indexing failed: {str(e)}")
    else:
        st.sidebar.error("Invalid folder path!")

# Video indexing
video_folder_path = st.sidebar.text_input(
    "Video Folder Path",
    value="./videos",
    help="Enter the path to folder containing videos to index"
)

if st.sidebar.button("🎬 Index Videos", type="primary"):
    if video_folder_path and Path(video_folder_path).exists():
        with st.spinner(f"Indexing videos from {video_folder_path}..."):
            try:
                st.session_state.video_processor.process_videos_to_qdrant(
                    folder_path=video_folder_path,
                    collection_name=video_collection_name,
                    qdrant_host=qdrant_host,
                    qdrant_port=qdrant_port
                )
                st.sidebar.success(f"✓ Successfully indexed videos from {video_folder_path}")
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
            all_results = []
            
            # Search images
            if search_mode in ["Images", "Both"]:
                with st.spinner("Searching images..."):
                    image_results = st.session_state.image_processor.search_by_text(
                        query_text=query_text,
                        collection_name=image_collection_name,
                        qdrant_host=qdrant_host,
                        qdrant_port=qdrant_port,
                        limit=num_results
                    )
                    for result in image_results:
                        result['type'] = 'image'
                    all_results.extend(image_results)
            
            # Search videos
            if search_mode in ["Videos", "Both"]:
                with st.spinner("Searching videos..."):
                    video_results = st.session_state.video_processor.search_videos_by_text(
                        query_text=query_text,
                        collection_name=video_collection_name,
                        qdrant_host=qdrant_host,
                        qdrant_port=qdrant_port,
                        limit=num_results
                    )
                    for result in video_results:
                        result['type'] = 'video'
                    all_results.extend(video_results)
            
            # Sort all results by score
            all_results.sort(key=lambda x: x['score'], reverse=True)
            
            if not all_results:
                st.warning("No results found. Try a different query or index some content.")
            else:
                st.success(f"Found {len(all_results)} matching results!")
                st.markdown("---")
                
                # Display results based on type
                if search_mode == "Images" or (search_mode == "Both" and image_results):
                    st.subheader("🖼️ Image Results")
                    
                    image_only = [r for r in all_results if r['type'] == 'image']
                    if image_only:
                        cols_per_row = 3
                        for i in range(0, len(image_only), cols_per_row):
                            cols = st.columns(cols_per_row)
                            
                            for j, col in enumerate(cols):
                                idx = i + j
                                if idx < len(image_only):
                                    result = image_only[idx]
                                    
                                    with col:
# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <small>Powered by OpenAI CLIP and Qdrant Vector Database<br>
    Supporting both image and video search with semantic understanding</small>
    </div>
    """,
    unsafe_allow_html=True
)                                               
                                                with st.expander("View full path"):
                                                    st.code(result['image_path'])
                                            else:
                                                st.error(f"Image not found: {result['filename']}")
                                        except Exception as e:
                                            st.error(f"Error loading image: {e}")
                                        
                                        st.markdown("---")
                
                if search_mode == "Videos" or (search_mode == "Both" and video_results):
                    st.subheader("🎬 Video Results")
                    
                    video_only = [r for r in all_results if r['type'] == 'video']
                    if video_only:
                        # Group by video
                        videos_dict = {}
                        for result in video_only:
                            video_name = result['video_name']
                            if video_name not in videos_dict:
                                videos_dict[video_name] = []
                            videos_dict[video_name].append(result)
                        
                        # Display each video
                        for video_name, segments in videos_dict.items():
                            with st.expander(f"📹 {video_name} ({len(segments)} matching segments)", expanded=True):
                                st.markdown(f"**Path:** `{segments[0]['video_path']}`")
                                st.markdown(f"**Total Frames:** {segments[0]['total_frames']}")
                                
                                # Display segments
                                for i, segment in enumerate(segments, 1):
                                    cols = st.columns([1, 2, 2])
                                    
                                    with cols[0]:
                                        st.metric("Segment", f"#{i}")
                                        st.metric("Score", f"{segment['score']:.4f}")
                                    
                                    with cols[1]:
                                        st.markdown(f"**Cluster ID:** {segment['cluster_id']}")
                                        st.markdown(f"**Frames in Cluster:** {segment['num_frames_in_cluster']}")
                                    
                                    with cols[2]:
                                        frame_indices = segment['frame_indices']
                                        if len(frame_indices) <= 10:
                                            st.markdown(f"**Frame Indices:** {frame_indices}")
                                        else:
                                            st.markdown(f"**Frame Indices:** {frame_indices[:10]} ... (+{len(frame_indices)-10} more)")
                                    
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
