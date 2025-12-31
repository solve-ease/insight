from video_to_embedding import VideoEmbeddingProcessor
import sys
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    # Configuration
    COLLECTION_NAME = os.getenv("VIDEO_COLLECTION_NAME", "video_embeddings")
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
    
    # Get query from command line or use default
    if len(sys.argv) > 1:
        query_text = " ".join(sys.argv[1:])
    else:
        query_text = "person walking"  # Default query
    
    print(f"\n{'='*60}")
    print(f"Searching videos for: '{query_text}'")
    print(f"{'='*60}\n")
    
    # Initialize processor
    try:
        processor = VideoEmbeddingProcessor()
    except Exception as e:
        print(f"Error initializing processor: {e}")
        return
    
    # Search for similar videos
    try:
        results = processor.search_videos_by_text(
            query_text=query_text,
            collection_name=COLLECTION_NAME,
            qdrant_host=QDRANT_HOST,
            qdrant_port=QDRANT_PORT,
            limit=20
        )
    except Exception as e:
        print(f"Error searching videos: {e}")
        return
    
    # Display results
    if not results:
        print("No results found.")
        print("\nTips:")
        print("  - Make sure videos have been indexed using video_to_embedding.py")
        print("  - Try different search terms")
        print("  - Check that Qdrant is running")
    else:
        print(f"Found {len(results)} matching video segments:\n")
        
        # Group results by video
        videos_dict = {}
        for result in results:
            video_name = result['video_name']
            if video_name not in videos_dict:
                videos_dict[video_name] = []
            videos_dict[video_name].append(result)
        
        # Display grouped results
        rank = 1
        for video_name, video_results in videos_dict.items():
            print(f"\n{'─'*60}")
            print(f"Video: {video_name}")
            print(f"Path: {video_results[0]['video_path']}")
            print(f"Total Frames: {video_results[0]['total_frames']}")
            print(f"Matching Segments: {len(video_results)}")
            print(f"{'─'*60}")
            
            for i, result in enumerate(video_results, 1):
                print(f"\n  Segment {i}:")
                print(f"    Rank: #{rank}")
                print(f"    Score: {result['score']:.4f}")
                print(f"    Cluster ID: {result['cluster_id']}")
                print(f"    Frames in Cluster: {result['num_frames_in_cluster']}")
                
                # Show frame indices (first 10 if many)
                frame_indices = result['frame_indices']
                if len(frame_indices) <= 10:
                    print(f"    Frame Indices: {frame_indices}")
                else:
                    print(f"    Frame Indices: {frame_indices[:10]} ... (and {len(frame_indices)-10} more)")
                
                rank += 1
        
        print(f"\n{'='*60}")
        print(f"Total: {len(results)} segments from {len(videos_dict)} video(s)")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
