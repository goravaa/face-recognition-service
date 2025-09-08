import grpc
import logging
import cv2
import numpy as np

# Import the generated gRPC classes for the face attributes service
import face_attributes_pb2
import face_attributes_pb2_grpc

# Import the gRPC Health Checking modules
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc


# Set up basic logging to see the test output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# The address of the gRPC server to test
GRPC_SERVER_ADDRESS = 'localhost:50052' # Make sure this port matches your server.py config

def create_test_image() -> bytes:
    """
    Loads a real face image from a file.
    If the file is not found, it creates a random dummy image.
    """
    try:
        # IMPORTANT: Replace this with the actual path to a test image of a face
        image_path = r"C:\Users\garvw\Projects\face-recognition-service\services\embeddings\didi.jpg"
        with open(image_path, "rb") as f:
            logger.info(f"Loading test image from: {image_path}")
            return f.read()
    except FileNotFoundError:
        logger.warning(f"Test image not found at the specified path. Creating a dummy 224x224 image.")
        # Create a dummy image that matches the model's expected input size
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        is_success, buffer = cv2.imencode(".jpg", dummy_image)
        if not is_success:
            raise RuntimeError("Failed to create a dummy jpg image.")
        return buffer.tobytes()

def test_health_check():
    """Tests the gRPC health check endpoint to ensure the service reports 'SERVING'."""
    logger.info("--- Testing gRPC Health Check ---")
    service_name = 'faceattributes.AttributeService'
    try:
        with grpc.insecure_channel(GRPC_SERVER_ADDRESS) as channel:
            stub = health_pb2_grpc.HealthStub(channel)
            request = health_pb2.HealthCheckRequest(service=service_name)
            
            # Use a timeout to avoid waiting forever if the server is unresponsive
            response = stub.Check(request, timeout=5)

            if response.status == health_pb2.HealthCheckResponse.SERVING:
                logger.info(f"SUCCESS: Health check passed. Service '{service_name}' is SERVING.")
                return True
            else:
                logger.error(f"FAILED: Service '{service_name}' is not serving. Status: {response.status}")
                return False

    except grpc.RpcError as e:
        logger.error(f"FAILED: Could not connect or run health check. Is the server running? Code: {e.code()}, Details: {e.details()}")
        return False
    except Exception as e:
        logger.error(f"FAILED: An unexpected error occurred during health check: {e}", exc_info=True)
        return False


def test_get_attributes():
    """Tests the GetAttributes RPC with a valid image."""
    logger.info("--- Testing GetAttributes with a valid image ---")
    
    try:
        with grpc.insecure_channel(GRPC_SERVER_ADDRESS) as channel:
            stub = face_attributes_pb2_grpc.AttributeServiceStub(channel)
            image_data = create_test_image()
            request = face_attributes_pb2.AttributeRequest(image_data=image_data)
            
            response = stub.GetAttributes(request, timeout=10) # Added timeout
            
            if response.status == "success":
                logger.info("SUCCESS: Attributes received successfully!")
                logger.info(f"  - Predicted Race:   {response.race}")
                logger.info(f"  - Predicted Gender: {response.gender}")
                logger.info(f"  - Predicted Age:    {response.age}")
                
                # Print the top 3 race probabilities for inspection
                race_probs = sorted(response.race_probs.items(), key=lambda item: item[1], reverse=True)
                logger.info(f"  - Race Probabilities (Top 3): {race_probs[:3]}")

                gender_probs = sorted(response.gender_probs.items(), key=lambda item: item[1], reverse=True)
                logger.info(f"  - Gender Probabilities: {gender_probs}")

                return True
            else:
                logger.error(f"FAILED: Server returned an error: {response.error_message}")
                return False

    except grpc.RpcError as e:
        logger.error(f"FAILED: gRPC Error - Code: {e.code()}, Details: {e.details()}")
        return False
    except Exception as e:
        logger.error(f"FAILED: An unexpected error occurred: {e}", exc_info=True)
        return False

def test_invalid_image():
    """Tests the GetAttributes RPC with invalid (non-image) data."""
    logger.info("--- Testing GetAttributes with invalid image data ---")
    
    try:
        with grpc.insecure_channel(GRPC_SERVER_ADDRESS) as channel:
            stub = face_attributes_pb2_grpc.AttributeServiceStub(channel)
            # Send random bytes that cannot be decoded as an image
            request = face_attributes_pb2.AttributeRequest(image_data=b"this is not an image")
            
            response = stub.GetAttributes(request, timeout=5)

            # The server should return a clear "error" status
            if response.status == "error" and response.error_message:
                logger.info(f"SUCCESS: Server correctly handled invalid data with message: '{response.error_message}'")
                return True
            else:
                logger.error(f"FAILED: Server did not return the expected error status for invalid data.")
                return False

    except grpc.RpcError as e:
        # The server might close the connection with an error code, which is also an acceptable outcome for this test.
        logger.info(f"SUCCESS: Server correctly rejected invalid data with gRPC error: {e.code()} - {e.details()}")
        return True
    except Exception as e:
        logger.error(f"FAILED: An unexpected error occurred: {e}", exc_info=True)
        return False

def main():
    """Runs all defined tests and prints a final summary."""
    logger.info("=" * 60)
    logger.info(f" Starting gRPC Attribute Service Test Client")
    logger.info(f"   Targeting server at: {GRPC_SERVER_ADDRESS}")
    logger.info("=" * 60)

    tests_to_run = {
        "Health Check Test": test_health_check,
        "Get Attributes Test": test_get_attributes,
        "Invalid Image Data Test": test_invalid_image,
    }

    results = {}
    for name, test_func in tests_to_run.items():
        result = test_func()
        results[name] = " PASS" if result else " FAIL"
        logger.info("-" * 60)

    # Final Summary
    logger.info(" Test Summary")
    logger.info("=" * 60)
    passed_count = 0
    for name, status in results.items():
        logger.info(f"- {name}: {status}")
        if status == " PASS":
            passed_count += 1
            
    total_count = len(results)
    logger.info(f"\n Final Result: {passed_count} / {total_count} tests passed.")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
