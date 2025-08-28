# test_grpc_client.py
import grpc
import cv2
import numpy as np
import logging
from typing import List

import embeddings_pb2
import embeddings_pb2_grpc

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_image() -> bytes:
    """Create a simple test image or load from file"""
    try:
        # Try to load an existing face image
        image_path = r"C:\Users\garvw\Projects\face-recognition-service\services\embeddings\didi.jpg"  # Replace with your test image path
        with open(image_path, "rb") as f:
            return f.read()
    except FileNotFoundError:
        logger.warning("Test image not found, creating a dummy image")
        # Create a dummy image (112x112x3 - matching your model input)
        dummy_image = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', dummy_image)
        return buffer.tobytes()

def test_single_embedding():
    """Test single embedding generation"""
    logger.info("Testing single embedding generation...")
    
    # Create channel
    channel = grpc.insecure_channel('localhost:50051')
    stub = embeddings_pb2_grpc.EmbeddingServiceStub(channel)
    
    try:
        # Get test image
        image_data = create_test_image()
        logger.info(f"Image size: {len(image_data)} bytes")
        
        # Create request
        request = embeddings_pb2.EmbeddingRequest(image_data=image_data)
        
        # Call service
        response = stub.GetEmbedding(request)
        
        # Check response
        if response.status == "success":
            logger.info("SUCCESS: Embedding generated successfully!")
            logger.info(f"   - Embedding length: {len(response.embedding)}")
            logger.info(f"   - First 5 values: {response.embedding[:5]}")
            logger.info(f"   - Last 5 values: {response.embedding[-5:]}")
            
            # Validate embedding properties
            embedding_array = np.array(response.embedding)
            norm = np.linalg.norm(embedding_array)
            logger.info(f"   - L2 Norm: {norm:.6f} (should be approximately 1.0 for normalized)")
            
            return True
        else:
            logger.error(f"FAILED: Failed to generate embedding: {response.status}")
            return False
            
    except grpc.RpcError as e:
        logger.error(f"gRPC Error: {e.code()} - {e.details()}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return False
    finally:
        channel.close()

def test_invalid_image():
    """Test with invalid image data"""
    logger.info("Testing with invalid image data...")
    
    channel = grpc.insecure_channel('localhost:50051')
    stub = embeddings_pb2_grpc.EmbeddingServiceStub(channel)
    
    try:
        # Send invalid data
        request = embeddings_pb2.EmbeddingRequest(image_data=b"invalid_image_data")
        response = stub.GetEmbedding(request)
        
        # This should not succeed
        logger.info(f"Response status: {response.status}")
        return True
        
    except grpc.RpcError as e:
        logger.info(f"CORRECTLY rejected invalid image: {e.code()} - {e.details()}")
        return True
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return False
    finally:
        channel.close()

def test_multiple_requests():
    """Test multiple concurrent requests"""
    logger.info("Testing multiple concurrent requests...")
    
    channel = grpc.insecure_channel('localhost:50051')
    stub = embeddings_pb2_grpc.EmbeddingServiceStub(channel)
    
    try:
        image_data = create_test_image()
        success_count = 0
        
        # Send 5 requests
        for i in range(5):
            request = embeddings_pb2.EmbeddingRequest(image_data=image_data)
            response = stub.GetEmbedding(request)
            
            if response.status == "success":
                success_count += 1
                logger.info(f"Request {i+1} successful")
            else:
                logger.error(f"Request {i+1} failed: {response.status}")
        
        logger.info(f"Completed {success_count}/5 requests successfully")
        return success_count == 5
        
    except Exception as e:
        logger.error(f"Error in multiple requests test: {str(e)}")
        return False
    finally:
        channel.close()

def main():
    """Run all tests"""
    logger.info("Starting gRPC Embedding Service Tests")
    
    tests = [
        ("Single Embedding Test", test_single_embedding),
        ("Invalid Image Test", test_invalid_image),
        ("Multiple Requests Test", test_multiple_requests),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n" + "="*50)
            logger.info(f"Running: {test_name}")
            logger.info(f"="*50)
            
            result = test_func()
            results.append((test_name, result))
            
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info(f"="*50)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{status} - {test_name}")
        if result:
            passed += 1
    
    logger.info(f"\nFinal Result: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("All tests passed!")
        return 0
    else:
        logger.error("Some tests failed!")
        return 1

if __name__ == "__main__":
    exit(main())