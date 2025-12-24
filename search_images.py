from image_to_embedding import ImageEmbeddingProcessor
import sys
import shutil
from pathlib import Path

def main():
    # Configuration
    COLLECTION_NAME = "image_embeddings"
    QDRANT_HOST = "localhost"
    QDRANT_PORT = 6333
    RESULTS_FOLDER = "./search_results"
    
    # Get query from command line or use default
    if len(sys.argv) > 1:
        query_text = " ".join(sys.argv[1:])
    else:
        query_text = "aadhaar card"  # Default query
    
    print(f"Searching for: '{query_text}'")
    print("-" * 50)
    
    # Clean and create results folder
    results_path = Path(RESULTS_FOLDER)
    if results_path.exists():
        shutil.rmtree(results_path)
    results_path.mkdir(parents=True, exist_ok=True)
    print(f"Cleared search results folder: {RESULTS_FOLDER}\n")
    
    # Initialize processor
    processor = ImageEmbeddingProcessor()
    
    # Search for similar images
    results = processor.search_by_text(
        query_text=query_text,
        collection_name=COLLECTION_NAME,
        qdrant_host=QDRANT_HOST,
        qdrant_port=QDRANT_PORT,
        limit=100
    )
    
    # Display results and copy images
    if not results:
        print("No results found.")
    else:
        print(f"Top {len(results)} results:\n")
        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result['score']:.4f}")
            print(f"   File: {result['filename']}")
            print(f"   Path: {result['image_path']}")
            
            # Copy image to results folder
            source_path = Path(result['image_path'])
            if source_path.exists():
                # Create filename with rank and score
                dest_filename = f"{i:02d}_score{result['score']:.4f}_{result['filename']}"
                dest_path = results_path / dest_filename
                shutil.copy2(source_path, dest_path)
                print(f"   Copied to: {dest_path}")
            else:
                print(f"   Warning: Source file not found!")
            print()
        
        print(f"\n✓ Copied {len(results)} images to {RESULTS_FOLDER}/")

if __name__ == "__main__":
    main()
