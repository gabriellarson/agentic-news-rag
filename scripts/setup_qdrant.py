"""
Setup script for Qdrant vector database.

This script:
1. Checks if Qdrant is running
2. Creates the news_articles collection with hybrid search support
3. Configures indexes for optimal performance
"""

import sys
from pathlib import Path
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams,
    CollectionInfo, OptimizersConfigDiff,
    HnswConfigDiff
)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import get_config

logger = logging.getLogger(__name__)


class QdrantSetup:
    """Setup and configuration for Qdrant vector database"""
    
    def __init__(self, config=None, host=None, port=None):
        if config is None:
            config = get_config()
        
        self.config = config
        
        # Get Qdrant configuration
        self.host = host or config.get('qdrant', 'host', default='localhost')
        self.port = port or config.get('qdrant', 'port', default=6333)
        self.collection_name = config.get('qdrant', 'collection_name', default='news_articles')
        self.on_disk = config.get('qdrant', 'on_disk', default=True)
        
        # Get embedding dimension from config
        self.vector_size = config.get('embeddings', 'dimension', default=1024)  # Qwen3-0.6B default
        
        self.client = None
    
    def check_connection(self) -> bool:
        """Check if Qdrant server is running and accessible"""
        try:
            self.client = QdrantClient(host=self.host, port=self.port)
            # Try to get collections to verify connection
            collections = self.client.get_collections()
            logger.info(f"Successfully connected to Qdrant at {self.host}:{self.port}")
            logger.info(f"Found {len(collections.collections)} existing collections")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            return False
    
    def create_collection(self, recreate: bool = False) -> bool:
        """Create the news articles collection with hybrid search support"""
        if not self.client:
            logger.error("Not connected to Qdrant")
            return False
        
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            exists = any(c.name == self.collection_name for c in collections.collections)
            
            if exists:
                if recreate:
                    logger.info(f"Deleting existing collection '{self.collection_name}'")
                    self.client.delete_collection(self.collection_name)
                else:
                    logger.info(f"Collection '{self.collection_name}' already exists")
                    return True
            
            logger.info(f"Creating collection '{self.collection_name}'")
            
            # Create collection with both dense and sparse vectors
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                        on_disk=self.on_disk  # Store on disk for large collections
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams()
                },
                # Optimize for accuracy over speed
                hnsw_config=HnswConfigDiff(
                    m=16,  # Number of edges per node
                    ef_construct=200,  # Build-time accuracy
                    full_scan_threshold=10000  # Use HNSW for collections > 10k
                ),
                optimizers_config=OptimizersConfigDiff(
                    default_segment_number=4,  # Number of segments
                    max_segment_size=100000,  # Max points per segment
                    memmap_threshold=50000,  # Use memory mapping for large segments
                    indexing_threshold=20000,  # Start indexing after 20k points
                )
            )
            
            logger.info(f"Collection '{self.collection_name}' created successfully")
            
            # Create payload indexes for filtering
            self._create_indexes()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False
    
    def _create_indexes(self):
        """Create payload indexes for efficient filtering"""
        try:
            # Create index on timestamp for temporal filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="timestamp",
                field_schema="float"
            )
            logger.info("Created index on 'timestamp' field")
            
            # Note: Qdrant doesn't support indexing array fields directly
            # Authors filtering will work without an index, just slightly slower
            logger.info("Skipping index on 'authors' field (array type not supported for indexing)")
            
            # Create index on published date string for exact matching
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="published_date",
                field_schema="keyword"
            )
            logger.info("Created index on 'published_date' field")
            
        except Exception as e:
            logger.warning(f"Failed to create some indexes: {e}")
    
    def get_collection_info(self) -> CollectionInfo:
        """Get information about the collection"""
        if not self.client:
            logger.error("Not connected to Qdrant")
            return None
        
        try:
            info = self.client.get_collection(self.collection_name)
            return info
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return None
    
    def setup(self, recreate: bool = False) -> bool:
        """Run the complete setup process"""
        logger.info("Starting Qdrant setup...")
        
        # Check connection
        if not self.check_connection():
            logger.error("Cannot connect to Qdrant. Is the server running?")
            logger.info("To start Qdrant with Docker:")
            logger.info("  docker run -p 6333:6333 -v ~/qdrant_storage:/qdrant/storage qdrant/qdrant")
            return False
        
        # Create collection
        if not self.create_collection(recreate=recreate):
            return False
        
        # Get collection info
        info = self.get_collection_info()
        if info:
            logger.info(f"Collection status: {info.status}")
            logger.info(f"Points count: {info.points_count}")
            logger.info(f"Vectors config: {info.config.params.vectors}")
        
        logger.info("Qdrant setup completed successfully!")
        return True


def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup Qdrant for News RAG")
    parser.add_argument(
        "--recreate", 
        action="store_true",
        help="Recreate collection if it exists (WARNING: deletes existing data)"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Qdrant host (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6333,
        help="Qdrant port (default: 6333)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get config
    config = get_config()
    
    # Run setup with optional host/port overrides
    setup = QdrantSetup(
        config=config,
        host=args.host if args.host != "localhost" else None,
        port=args.port if args.port != 6333 else None
    )
    success = setup.setup(recreate=args.recreate)
    
    if not success:
        sys.exit(1)
    
    print("\nNext steps:")
    print("1. Run the indexing script to add articles:")
    print("   python scripts/index_articles.py")
    print("\n2. Test search functionality:")
    print("   python scripts/test_search.py")


if __name__ == "__main__":
    main()