"""
RAG Backend - Modal GPU Service with DynamoDB
==============================================

Clean FastAPI backend with:
- Single persistence initialization
- Proper trace fetching with user ownership verification
- X-User-Anonymous header for business logic decisions
- Save-at-start pattern for abandoned request visibility
- Single message_id (serves as both message and trace identifier)
- Chat title generation endpoint

Endpoints:
- POST /query - Process RAG query (streaming or non-streaming)
- GET /health - Health check
- GET /chats - List user's chats (requires non-anonymous user)
- GET /chats/{chat_id}/messages - Get chat history
- GET /chats/{chat_id}/messages/{message_id}/trace - Get trace for a message
- POST /chats/{chat_id}/generate-title - Generate a title for the chat
"""

import modal
from fastapi import FastAPI, HTTPException, Header, Security, Depends
from fastapi.security import APIKeyHeader
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json
import os

# =============================================================================
# Configuration
# =============================================================================

LOCAL_CODE_DIR = "."
MODAL_REMOTE_CODE_DIR = "/root/app"
MODEL_VOLUME_NAME = "model-weights-vol"
DATA_VOLUME_NAME = "data-storage-vol"

GPU_TYPE = "L4"
CONTAINER_IDLE_TIMEOUT = 900

# Security
API_KEY_NAME = "X-Service-Token"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def download_nltk():
    import nltk
    nltk.download("punkt_tab")


# =============================================================================
# Modal Image
# =============================================================================

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .uv_pip_install("vllm", extra_options="--torch-backend=cu128")
    .uv_pip_install(
        "sentence-transformers==5.2.0",
        "chromadb==1.4.0",
        "nltk==3.9.2",
        "rank-bm25",
        "accelerate",
        "fastapi",
        "pydantic>=2.0",
        "boto3",
    )
    .run_function(download_nltk)
    .env({
        "ENV": "prod",
        "HF_HOME": "/models",
        "PYTHONPATH": MODAL_REMOTE_CODE_DIR,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "SERVICE_AUTH_TOKEN": "dev-secret-123",  # Override in Modal Secrets
        "USE_DYNAMODB": "true",
        "AWS_REGION": "us-east-2",
        "AWS_ROLE_ARN": "arn:aws:iam::649489225731:role/modal-litlens-role",
    })
    .add_local_dir(
        LOCAL_CODE_DIR,
        remote_path=MODAL_REMOTE_CODE_DIR,
        ignore=[".git/", "__pycache__/", ".ipynb_checkpoints/",
                "litlens_inference/", ".files/", ".chainlit/",
                "logs/", "traces/", "*.pyc","aws/"],
    )
)

# =============================================================================
# Modal App
# =============================================================================

app = modal.App("rag-backend-service")
model_vol = modal.Volume.from_name(MODEL_VOLUME_NAME)
data_vol = modal.Volume.from_name(DATA_VOLUME_NAME)

# NEED TO CHANGE THE DEFAULT CONV idn and hallucination check
class ChatCompletionRequest(BaseModel):
    model: str = "litlens-rag"
    messages: list[dict]  # [{"role": "user", "content": "..."}]
    stream: Optional[bool] = True
    # Custom extensions (not in OpenAI spec but useful)
    conversation_id: Optional[str] = "chat1"
    enable_hallucination_check: Optional[bool] = True


class GenerateTitleRequest(BaseModel):
    """Request body for title generation."""
    queries: list[str]  # Frontend provides the queries it already has


# =============================================================================
# RAG Service
# =============================================================================

@app.cls(
    image=image,
    gpu=GPU_TYPE,
    volumes={
        "/models": model_vol,
        "/data": data_vol,
    },
    scaledown_window=CONTAINER_IDLE_TIMEOUT,
    timeout=600,
    min_containers=0,
    max_containers=1,
    # secrets=[modal.Secret.from_name("litlens-config")],
)
@modal.concurrent(max_inputs=10)
class RAGService:
    """Modal class hosting RAG pipeline + FastAPI."""
    
    pipeline = None
    persistence = None

    @modal.enter()
    async def startup(self):
        """Initialize pipeline and persistence."""
        import sys
        sys.path.insert(0, MODAL_REMOTE_CODE_DIR)
        
        from utils.logger import configure_logging
        configure_logging({"logging": {"environment": "production", "level": "INFO"}})
        
        # Initialize persistence
        use_dynamodb = os.environ.get("USE_DYNAMODB", "false").lower() == "true"
        aws_role_arn = os.environ.get("AWS_ROLE_ARN")
        aws_region = os.environ.get("AWS_REGION", "us-east-2")
        
        if use_dynamodb and aws_role_arn:
            print(f"Initializing DynamoDB with OIDC...")
            print(f"  Role ARN: {aws_role_arn}")
            print(f"  Region: {aws_region}")
            
            try:
                from dynamo_persistence import DynamoPersistence
                self.persistence = DynamoPersistence.from_oidc(
                    role_arn=aws_role_arn,
                    region=aws_region,
                )
                print("✓ DynamoDB initialized")
            except Exception as e:
                print(f"ERROR: DynamoDB init failed: {e}")
                self.persistence = None
        else:
            if use_dynamodb:
                print("WARNING: USE_DYNAMODB=true but AWS_ROLE_ARN not set")
            print("Running without persistence")
            
        # Message ID generation
        from dynamo_persistence import _generate_id
        self._generate_id = _generate_id
        
        # Initialize pipeline
        from chat_pipeline import RAGPipelineV2
        
        config_path = os.path.join(MODAL_REMOTE_CODE_DIR, "config.yaml")
        if not os.path.exists(config_path):
            config_path = None
        
        self.pipeline = RAGPipelineV2(
            config_path=config_path,
            use_dynamodb=self.persistence is not None,
            dynamodb_persistence=self.persistence,
        )
        
        await self.pipeline.initialize()
        print("✓ Pipeline ready")

    @modal.exit()
    async def shutdown(self):
        if self.pipeline:
            await self.pipeline.cleanup()
            print("Pipeline cleaned up")

    @modal.asgi_app()
    def web_app(self):
        """FastAPI application."""
        
        web_app = FastAPI(title="RAG Backend API", version="2.1.0")
        
        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        service = self

        async def verify_token(api_key: str = Security(api_key_header)):
            """Verify the service token (authenticates the frontend app)."""
            expected = os.environ.get("SERVICE_AUTH_TOKEN")
            if api_key != expected:
                raise HTTPException(status_code=403, detail="Invalid credentials")
            return api_key

        def parse_anonymous_header(x_user_anonymous: Optional[str] = Header(None)) -> bool:
            """
            Parse the X-User-Anonymous header.
            
            Frontend sends this to indicate if the Firebase user is anonymous.
            Values: "true", "false", or not present (defaults to False).
            """
            if x_user_anonymous is None:
                return False
            return x_user_anonymous.lower() == "true"

        # =================================================================
        # Core Endpoints
        # =================================================================

        @web_app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "pipeline_ready": service.pipeline is not None,
                "persistence": "dynamodb" if service.persistence else "none",
                "queue_depth": service.pipeline.generation_queue_depth if service.pipeline else 0,
            }

        @web_app.post("/v1/chat/completions")
        async def chat_completions(
            request: ChatCompletionRequest,
            token: str = Depends(verify_token),
            x_user_id: Optional[str] = Header(None),
            x_user_anonymous: Optional[str] = Header(None),
        ):
            """
            OpenAI-compatible chat completions endpoint.
            DEBUG MODE: Logging enabled + Sources/Hallucination moved to 'delta' for visibility.
            """
            import time as time_module
            
            if service.pipeline is None:
                raise HTTPException(status_code=503, detail="Pipeline not initialized")
            
            user_id = x_user_id or "anonymous"
            is_anonymous = parse_anonymous_header(x_user_anonymous)
            
            user_messages = [m for m in request.messages if m.get("role") == "user"]
            if not user_messages:
                raise HTTPException(status_code=400, detail="No user message provided")
            query = user_messages[-1].get("content", "")
            
            completion_id = f"chatcmpl-{service._generate_id()}"
            message_id = service._generate_id()
            created = int(time_module.time())
            model = request.model or "litlens-rag"
            conversation_id = request.conversation_id or "default-session"
            
            print(f"--- START STREAM: {conversation_id} ---")

            kwargs = {
                "query": query,
                "conversation_id": conversation_id,
                "user_id": user_id,
                "is_anonymous": is_anonymous,
                "enable_hallucination_check": request.enable_hallucination_check,
                "message_id": message_id,
            }

            if request.stream:
                def format_sse(payload: dict) -> str:
                    return f"data: {json.dumps(payload)}\n\n"

                text_id = f"text-{message_id}"

                def text_start() -> str:
                    return format_sse({"type": "text-start", "id": text_id})

                def text_delta(delta: str) -> str:
                    return format_sse({"type": "text-delta", "id": text_id, "delta": delta})

                def text_end() -> str:
                    return format_sse({"type": "text-end", "id": text_id})

                def source_part(url: str, title: str) -> str:
                    return format_sse(
                        {
                            "type": "source-url",
                            "sourceId": url,
                            "url": url,
                            "title": title,
                        }
                    )

                def data_part(name: str, data: dict) -> str:
                    return format_sse({"type": f"data-{name}", "data": data})

                def error_part(message: str) -> str:
                    return format_sse({"type": "error", "error": message})

                async def stream():
                    try:
                        yield text_start()
                        async for event in service.pipeline.answer_stream(**kwargs):
                            if event.get("type") == "token":
                                yield text_delta(event.get("content", ""))
                            
                            elif event.get("type") == "context":
                                sources = event.get("data", [])
                                seen = set()
                                for source in sources:
                                    metadata = source.get("metadata", {})
                                    title = metadata.get("paper_title") or metadata.get("file_path")
                                    if not title or title in seen:
                                        continue
                                    seen.add(title)
                                    url = metadata.get("url") or metadata.get("file_path") or title
                                    yield source_part(url, title)
                                
                            elif event.get("type") == "hallucination":
                                yield data_part(
                                    "verification",
                                    {
                                        "grounding_ratio": event.get("grounding_ratio"),
                                        "num_claims": event.get("num_claims"),
                                        "num_grounded": event.get("num_grounded"),
                                        "unsupported_claims": event.get("unsupported_claims"),
                                        "verifications": event.get("verifications"),
                                    },
                                )
                            
                            elif event.get("type") == "done":
                                yield data_part(
                                    "completion",
                                    {
                                        "message_id": event.get("message_id"),
                                        "chat_id": event.get("chat_id"),
                                        "total_duration_ms": event.get("total_duration_ms", 0),
                                    },
                                )
                        
                        yield text_end()
                        yield "data: [DONE]\n\n"
                    
                    except Exception as e:
                        print(f"ERROR in Stream: {e}")
                        yield error_part(str(e))
                        yield "data: [DONE]\n\n"

                return StreamingResponse(
                    stream(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                        "x-vercel-ai-ui-message-stream": "v1",
                    },
                )
            else:
                # Non-streaming: collect all events
                answer = ""
                sources = []
                result_message_id = None
                chat_id = None
                total_duration_ms = 0
                
                async for event in service.pipeline.answer_stream(**kwargs):
                    if event.get("type") == "token":
                        answer += event.get("content", "")
                    elif event.get("type") == "context":
                        sources = event.get("data", [])
                    elif event.get("type") == "hallucination":
                        hallucination_result = {
                            "grounding_ratio": event.get("grounding_ratio"),
                            "num_claims": event.get("num_claims"),
                            "num_grounded": event.get("num_grounded"),
                            "unsupported_claims": event.get("unsupported_claims"),
                            "verifications": event.get("verifications"),
                        }
                    elif event.get("type") == "done":
                        result_message_id = event.get("message_id")
                        chat_id = event.get("chat_id")
                        total_duration_ms = event.get("total_duration_ms", 0)
                
                return {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": answer,
                        },
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "total_duration_ms": total_duration_ms,
                    },
                    "sources": sources,  # Extension
                    "hallucination": hallucination_result, # Will be None if not enabled
                    "message_id": result_message_id,
                    "chat_id": chat_id,
                }

        # =================================================================
        # Chat History Endpoints
        # =================================================================

        @web_app.get("/get_chat")
        async def list_chats(
            token: str = Depends(verify_token),
            x_user_id: Optional[str] = Header(None),
            x_user_anonymous: Optional[str] = Header(None),
            limit: int = 20,
        ):
            """
            List user's recent chats (for sidebar).
            
            Anonymous users get an empty list with a message to sign in.
            This is because anonymous users don't have persistent identity
            across sessions that would make chat history meaningful.
            """
            if not service.persistence:
                raise HTTPException(status_code=501, detail="Persistence not configured")
            
            user_id = x_user_id or "anonymous"
            is_anonymous = parse_anonymous_header(x_user_anonymous)
            
            # Anonymous users cannot list past chats
            if is_anonymous or user_id == "anonymous":
                return {
                    "chats": [],
                    "message": "Sign in to view chat history",
                }
            
            chats = await service.persistence.get_user_chats(user_id=user_id, limit=limit)
            return {"chats": chats}

        @web_app.get("/chats/{chat_id}/messages")
        async def get_messages(
            chat_id: str,
            token: str = Depends(verify_token),
            x_user_id: Optional[str] = Header(None),
            limit: int = 100,
        ):
            """
            Get all messages for a chat (for loading conversation).
            
            Verifies that the requesting user owns this chat.
            """
            if not service.persistence:
                raise HTTPException(status_code=501, detail="Persistence not configured")
            
            user_id = x_user_id or "anonymous"
            
            # Verify ownership
            is_owner, actual_owner = await service.persistence.verify_chat_ownership(
                chat_id, user_id
            )
            
            # If chat exists and user doesn't own it, deny access
            if actual_owner is not None and not is_owner:
                raise HTTPException(status_code=403, detail="Access denied")
            
            messages = await service.persistence.get_chat_messages(
                chat_id=chat_id, limit=limit
            )
            return {"chat_id": chat_id, "messages": messages}

        # =================================================================
        # Trace Endpoint (with ownership verification)
        # =================================================================

        @web_app.get("/chats/{chat_id}/messages/{message_id}/trace")
        async def get_message_trace(
            chat_id: str,
            message_id: str,
            token: str = Depends(verify_token),
            x_user_id: Optional[str] = Header(None),
            x_user_anonymous: Optional[str] = Header(None),
        ):
            """
            Get the trace for a specific message.
            
            This is the primary way to inspect what happened during a query.
            
            Security:
            - Verifies that the requesting user owns the chat
            - Returns 403 if user doesn't own the chat
            - Returns 404 if trace doesn't exist
            
            Args:
                chat_id: The chat ID containing the message
                message_id: The message ID (also the trace identifier)
            """
            if not service.persistence:
                raise HTTPException(status_code=501, detail="Persistence not configured")
            
            user_id = x_user_id or "anonymous"
            is_anonymous = parse_anonymous_header(x_user_anonymous)
            
            # Use the auth-aware trace retrieval
            trace, error = await service.persistence.get_trace_with_auth(
                chat_id=chat_id,
                message_id=message_id,
                requesting_user_id=user_id,
                is_anonymous=is_anonymous,
            )
            
            if error == "not_found":
                raise HTTPException(status_code=404, detail="Trace not found")
            elif error == "unauthorized":
                raise HTTPException(status_code=403, detail="Access denied")
            
            return trace

        # =================================================================
        # Chat Title Generation
        # =================================================================

        @web_app.post("/chats/{chat_id}/generate-title")
        async def generate_chat_title(
            chat_id: str,
            request: GenerateTitleRequest,
            token: str = Depends(verify_token),
            x_user_id: Optional[str] = Header(None),
            x_user_anonymous: Optional[str] = Header(None),
        ):
            """
            Generate a descriptive title for a chat.
            
            This endpoint uses the already-loaded LLM to generate a concise,
            specific title based on the queries provided by the frontend.
            
            Workflow:
            1. Frontend tracks message count
            2. When count >= 2 and user is NOT anonymous, frontend calls this endpoint
            3. Frontend provides the queries it already has in the request body
            4. Backend generates title using LLM and updates metadata table
            5. Returns the generated title
            
            Security:
            - Requires non-anonymous user
            - Verifies user owns the chat before updating
            """
            if not service.persistence:
                raise HTTPException(status_code=501, detail="Persistence not configured")
            
            if not service.pipeline:
                raise HTTPException(status_code=503, detail="Pipeline not initialized")
            
            user_id = x_user_id or "anonymous"
            is_anonymous = parse_anonymous_header(x_user_anonymous)
            
            # Anonymous users cannot rename chats
            if is_anonymous or user_id == "anonymous":
                raise HTTPException(
                    status_code=403,
                    detail="Sign in to generate chat titles"
                )
            
            # Verify ownership
            is_owner, actual_owner = await service.persistence.verify_chat_ownership(
                chat_id, user_id
            )
            
            if actual_owner is None:
                raise HTTPException(status_code=404, detail="Chat not found")
            
            if not is_owner:
                raise HTTPException(status_code=403, detail="Access denied")
            
            # Validate queries
            if not request.queries:
                raise HTTPException(status_code=400, detail="No queries provided")
            
            queries = request.queries[:3]  # Cap at 3
            
            # Generate title using the already-loaded LLM
            try:
                title = await service.pipeline.generate_chat_title(queries)
            except Exception as e:
                print(f"Title generation failed: {e}")
                # Fallback to first query truncated
                title = queries[0][:50] if queries else "New Chat"
            
            # Update the title in metadata table
            success = await service.persistence.update_chat_title(
                user_id=user_id,
                chat_id=chat_id,
                title=title,
            )
            
            if not success:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to update chat title"
                )
            
            return {
                "chat_id": chat_id,
                "title": title,
                "generated": True,
            }

        # =================================================================
        # Admin/Debug Endpoints (optional, remove in production)
        # =================================================================

        @web_app.get("/admin/abandoned-traces")
        async def get_abandoned_traces(
            token: str = Depends(verify_token),
            limit: int = 50,
        ):
            """
            Get traces that were abandoned (status still "running").
            
            Useful for monitoring and debugging.
            """
            if not service.persistence:
                raise HTTPException(status_code=501, detail="Persistence not configured")
            
            traces = await service.persistence.get_abandoned_traces(limit=limit)
            return {
                "count": len(traces),
                "traces": traces,
            }

        return web_app
