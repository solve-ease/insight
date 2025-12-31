"""
Example script demonstrating the complete video search workflow
"""

import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import the video processor
from video_to_embedding import VideoEmbeddingProcessor

def main():
    print("\n" + "="*70)
    print(" VIDEO SEARCH - COMPLETE WORKFLOW EXAMPLE")
    print("="*70)
    
    # Configuration
    VIDEO_FOLDER = "./videos"
    COLLECTION_NAME = "video_embeddings"
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
    
    # Step 1: Initialize the processor
    print("\n[Step 1] Initializing Video Embedding Processor...")
    processor = VideoEmbeddingProcessor()
    print("✓ Processor initialized")
    
    # Step 2: Check if video folder exists
    print(f"\n[Step 2] Checking video folder: {VIDEO_FOLDER}")
    video_path = Path(VIDEO_FOLDER)
    
    if not video_path.exists():
        print(f"⚠ Video folder doesn't exist, creating: {VIDEO_FOLDER}")
        video_path.mkdir(parents=True, exist_ok=True)
        print(f"\nPlease add video files (.mp4, .avi, .mov, .mkv) to {VIDEO_FOLDER}")
        print("Then run this script again.")
        return
    
    # Count videos
    video_files = list(video_path.rglob("*.mp4")) + \
                  list(video_path.rglob("*.avi")) + \
                  list(video_path.rglob("*.mov")) + \
                  list(video_path.rglob("*.mkv"))
    
    if not video_files:
        print(f"⚠ No video files found in {VIDEO_FOLDER}")
        print("Please add video files and run again.")
        return
    
    print(f"✓ Found {len(video_files)} video file(s)")
    for vf in video_files[:5]:  # Show first 5
        print(f"  - {vf.name}")
    if len(video_files) > 5:
        print(f"  ... and {len(video_files) - 5} more")
    
    # Step 3: Index videos
    print(f"\n[Step 3] Indexing videos to collection '{COLLECTION_NAME}'...")
    print("This process will:")
    print("  1. Sample frames from each video")
    print("  2. Generate CLIP embeddings for each frame")
    print("  3. Cluster similar frames")
    print("  4. Mean-pool clusters")
    print("  5. Store in Qdrant vector database")
    print()
    
    try:
        processor.process_videos_to_qdrant(
            folder_path=VIDEO_FOLDER,
            collection_name=COLLECTION_NAME,
            qdrant_host=QDRANT_HOST,
            qdrant_port=QDRANT_PORT
        )
        print("\n✓ Video indexing completed!")
    except Exception as e:
        print(f"\n✗ Indexing failed: {e}")
        print("\nTroubleshooting:")
        print("  - Make sure Qdrant is running: docker-compose up -d qdrant")
        print("  - Check video files are valid")
        print("  - Ensure sufficient disk space and memory")
        return
    
    # Step 4: Get collection statistics
    print(f"\n[Step 4] Collection Statistics")
    try:
        stats = processor.get_collection_stats(
            collection_name=COLLECTION_NAME,
            qdrant_host=QDRANT_HOST,
            qdrant_port=QDRANT_PORT
        )
        
        print(f"  Collection Name: {stats['collection_name']}")
        print(f"  Total Embeddings: {stats['total_embeddings']}")
        print(f"  Vector Dimension: {stats['vector_dimension']}")
        print(f"  Distance Metric: {stats['distance_metric']}")
    except Exception as e:
        print(f"⚠ Could not get stats: {e}")
    
    # Step 5: Example searches
    print(f"\n[Step 5] Running Example Searches")
    
    example_queries = [
        "person walking",
        "outdoor scene",
        "indoor setting",
        "car or vehicle",
        "people talking"
    ]
    
    for query in example_queries:
        print(f"\n  Query: '{query}'")
        try:
            results = processor.search_videos_by_text(
                query_text=query,
                collection_name=COLLECTION_NAME,
                qdrant_host=QDRANT_HOST,
                qdrant_port=QDRANT_PORT,
                limit=3
            )
            
            if results:
                print(f"  Found {len(results)} results:")
                for i, result in enumerate(results[:3], 1):
                    print(f"    {i}. {result['video_name']} (score: {result['score']:.4f})")
            else:
                print("  No results found")
        except Exception as e:
            print(f"  Search failed: {e}")
    
    # Done
    print("\n" + "="*70)
    print(" WORKFLOW COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  - Use the Streamlit UI for interactive search: streamlit run app.py")
    print("  - Or use CLI search: python3 search_videos.py 'your query'")
    print("  - View Qdrant dashboard: http://localhost:6333/dashboard")
    print()

if __name__ == "__main__":
    main()
