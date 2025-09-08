# frs/services/face-attributes/grpc_server.py
import grpc
from concurrent import futures
import time
import numpy as np
import cv2

import face_attributes_pb2
import face_attributes_pb2_grpc

from grpc_health.v1 import health
from grpc_health.v1 import health_pb2_grpc

# Import your existing components
from modelmanager import model_manager
from logger import logger
from config import Config as config

class AttributeServiceServicer(face_attributes_pb2_grpc.AttributeServiceServicer):
    """
    Implements the AttributeService gRPC servicer.
    (This class remains unchanged)
    """
    def GetAttributes(self, request, context):
        """
        Handles the GetAttributes RPC call. It receives an image, uses the
        ModelManager to predict attributes, and returns the results including
        confidence scores for each category.
        """
        logger.info("Received GetAttributes request.")

        try:
            # Decoding the incoming image bytes into a NumPy array
            image_np = np.frombuffer(request.image_data, np.uint8)
            face_crop_np = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

            if face_crop_np is None:
                raise ValueError("Failed to decode image data. The data may be corrupt or not a valid image format.")

            # ModelManager to get predictions
            predictions = model_manager.predict_attributes(face_crop_np)
            logger.info(f"Successfully predicted attributes: {predictions['race']}, {predictions['gender']}, {predictions['age_range']}")
            
          
            # taking the lower bound of the range.
            age_range = predictions.get("age_range", "0-0")
            predicted_age = int(age_range.split('-')[0].replace('+', ''))

           # Building the gRPC response, including the probability maps
            return face_attributes_pb2.AttributeResponse(
                status="success",
                race=predictions.get("race", "unknown"),
                gender=predictions.get("gender", "unknown"),
                age=predicted_age,
                race_probs=predictions.get("race_probs", {}),
                gender_probs=predictions.get("gender_probs", {}),
                age_probs=predictions.get("age_probs", {})
            )

        except Exception as e:
            logger.error(f"Error processing request: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return face_attributes_pb2.AttributeResponse(
                status="error",
                error_message=str(e)
            )

def serve():
    """
    Starts the gRPC server and waits for requests.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=config.GRPC_MAX_WORKERS))

    face_attributes_pb2_grpc.add_AttributeServiceServicer_to_server(
        AttributeServiceServicer(), server
    )

 
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

  
    service_name = 'faceattributes.AttributeService'
    health_servicer.set(service_name, 'SERVING')

    server_address = f'[::]:{config.GRPC_PORT}'
    server.add_insecure_port(server_address)
    server.start()
    logger.info(f"Face Attributes gRPC server with health check started successfully on {server_address}")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Server stopping...")
        server.stop(0)

if __name__ == '__main__':
    serve()
