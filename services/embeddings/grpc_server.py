# frs/services/embeddings/grpc_server.py
import grpc
import cv2
import numpy as np
from concurrent import futures

import embeddings_pb2
import embeddings_pb2_grpc

from grpc_health.v1 import health
from grpc_health.v1 import health_pb2_grpc

from config import Config
from modelmanager import model_manager

from logger import logger
config = Config()

class EmbeddingServiceServicer(embeddings_pb2_grpc.EmbeddingServiceServicer):
    """gRPC service implementation for embedding generation"""
    
    def GetEmbedding(self, request, context):
        """
        Generate face embedding from image bytes
        
        Args:
            request: EmbeddingRequest with image_data
            context: gRPC context
            
        Returns:
            EmbeddingResponse with embedding vector
        """
        try:
            logger.info("Received embedding request")
            
            np_arr = np.frombuffer(request.image_data, np.uint8)
            
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if image is None:
                logger.error("Failed to decode image")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Invalid image data provided")
                return embeddings_pb2.EmbeddingResponse(status="error")
            
            embedding = model_manager.get_embedding(image)
            
            logger.info(f"Successfully generated embedding with {len(embedding)} dimensions")
            
            return embeddings_pb2.EmbeddingResponse(
                embedding=embedding,
                status="success"
            )
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal error: {str(e)}")
            return embeddings_pb2.EmbeddingResponse(status="error")

def serve():
    """Start the gRPC server"""
    
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=config.GRPC_MAX_WORKERS),
        options=[
            ('grpc.max_send_message_length', config.GRPC_MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', config.GRPC_MAX_MESSAGE_LENGTH),
        ]
    )
    

    embeddings_pb2_grpc.add_EmbeddingServiceServicer_to_server(
        EmbeddingServiceServicer(), server
    )
    

    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)


    # This service name must match what the client (e.g., the API) will check for.
    service_name = 'embeddings.EmbeddingService'
    health_servicer.set(service_name, 'SERVING')
    
   
    server_address = f'[::]:{config.GRPC_PORT}'
    server.add_insecure_port(server_address)
    
    # Start server
    server.start()
    logger.info(f"Embedding gRPC server with health check started on {server_address}")
    logger.info(f"Max workers: {config.GRPC_MAX_WORKERS}")
    logger.info(f"Max message length: {config.GRPC_MAX_MESSAGE_LENGTH} bytes")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down gRPC server...")
        server.stop(0)

if __name__ == "__main__":
    serve()
