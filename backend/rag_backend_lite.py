"""
RAG Backend Lite - Modal CPU-only Mock Service
===============================================

Lightweight version of the RAG backend that:
- Runs on CPU only (no GPU, no L4)
- Keeps DynamoDB persistence fully functional
- Mocks the /v1/chat/completions endpoint with streaming responses
- Uses OpenAI-compatible API format (identical to full backend)
- Parses markdown response format with sources and hallucination check

Use case: 
- Development and testing without GPU costs
- Testing DynamoDB persistence and chat management
- Frontend integration testing

Streaming Event Lifecycle:
1. sources    - delta.sources: list of paper titles
2. token(s)   - delta.content: streamed answer tokens
3. hallucination - delta.hallucination: grounding metrics
4. done       - finish_reason: "stop" with usage and message_id
5. [DONE]     - end of stream

Endpoints (identical contract to full backend):
- POST /v1/chat/completions - OpenAI-compatible chat completions (mock)
- GET /health - Health check
- GET /chats - List user's chats
- GET /chats/{chat_id}/messages - Get chat history
- GET /chats/{chat_id}/messages/{message_id}/trace - Get trace for a message
- POST /chats/{chat_id}/generate-title - Generate a mock title
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
import asyncio
import random
import time

# =============================================================================
# Configuration
# =============================================================================

LOCAL_CODE_DIR = "."
MODAL_REMOTE_CODE_DIR = "/root/app"

# No GPU, longer idle timeout since it's cheap
CONTAINER_IDLE_TIMEOUT = 1800  # 30 minutes

# Security
API_KEY_NAME = "X-Service-Token"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
VERCEL_STREAM_HEADER = "x-vercel-ai-ui-message-stream"


# =============================================================================
# Modal Image (CPU-only, minimal dependencies)
# =============================================================================

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi",
        "pydantic>=2.0",
        "boto3",
    )
    .env({
        "ENV": "dev",
        "PYTHONPATH": MODAL_REMOTE_CODE_DIR,
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
                "logs/", "traces/", "*.pyc", "aws/",
                # Ignore GPU-heavy modules
                "inference/", "models/"],
    )
)

# =============================================================================
# Modal App
# =============================================================================

app = modal.App("rag-backend-lite")


# OpenAI-compatible request model (matches full backend)
class ChatCompletionRequest(BaseModel):
    model: str = "litlens-rag"
    messages: list[dict]  # [{"role": "user", "content": "..."}]
    stream: Optional[bool] = True
    # Custom extensions (not in OpenAI spec but useful)
    conversation_id: Optional[str] = "chat1"
    enable_hallucination_check: Optional[bool] = True


class GenerateTitleRequest(BaseModel):
    """Request body for title generation."""
    queries: list[str]


class RenameChatRequest(BaseModel):
    """Request body for renaming a chat."""
    title: str


class FeedbackRequest(BaseModel):
    chat_id: str
    message_id: str
    rating: str
    comment: Optional[str] = None


# =============================================================================
# Mock Data Generators
# =============================================================================

MOCK_SOURCES_FULL = [
    {
        "text": "The fundamental principles of causal inference require careful consideration of confounding variables and selection bias.",
        "metadata": {
            "paper_title": "Introduction to Causal Inference Methods",
            "file_path": "papers/causal_inference_intro.pdf",
            "chroma_id": "chunk_mock_001",
        },
        "score": 0.92,
    },
    {
        "text": "Regression analysis provides a flexible framework for modeling relationships between dependent and independent variables.",
        "metadata": {
            "paper_title": "Statistical Methods in Biomedical Research",
            "file_path": "papers/biostats_methods.pdf",
            "chroma_id": "chunk_mock_002",
        },
        "score": 0.87,
    },
    {
        "text": "The propensity score methodology offers a powerful approach to addressing confounding in observational studies.",
        "metadata": {
            "paper_title": "Propensity Score Methods: A Comprehensive Review",
            "file_path": "papers/propensity_scores.pdf",
            "chroma_id": "chunk_mock_003",
        },
        "score": 0.84,
    },
]

LOREM_SENTENCES = [
    "The analysis reveals significant correlations between the variables under study.",
    "Previous research has established foundational frameworks for understanding this phenomenon.",
    "Statistical methods employed include regression analysis and hypothesis testing.",
    "The findings suggest a moderate effect size with implications for clinical practice.",
    "Further investigation is warranted to validate these preliminary observations.",
    "The methodology adheres to established protocols in the field of biostatistics.",
    "Cross-validation techniques were applied to ensure model robustness.",
    "The confidence intervals indicate reasonable precision in the estimates.",
    "Longitudinal data analysis reveals temporal patterns of interest.",
    "The study population was selected using stratified random sampling.",
    "Causal inference methods were employed to address confounding variables.",
    "The results are consistent with theoretical predictions from prior work.",
    "Sensitivity analyses confirm the stability of the primary findings.",
    "The sample size provides adequate statistical power for the main hypotheses.",
    "Subgroup analyses reveal heterogeneous effects across demographic categories.",
]


def generate_mock_answer(query: str, num_sentences: int = 5) -> str:
    """Generate a mock answer based on the query."""
    random.seed(len(query))
    selected = random.sample(LOREM_SENTENCES, min(num_sentences, len(LOREM_SENTENCES)))
    random.seed()
    return " ".join(selected)


def generate_mock_title(queries: list[str]) -> str:
    """Generate a mock title from queries."""
    if not queries:
        return "New Chat"
    
    first_query = queries[0]
    words = first_query.split()[:6]
    
    if len(words) >= 3:
        return " ".join(words[:5]).title()
    return first_query[:50]


# =============================================================================
# RAG Lite Service
# =============================================================================

@app.cls(
    image=image,
    # No GPU!
    scaledown_window=CONTAINER_IDLE_TIMEOUT,
    timeout=300,
    min_containers=0,
    max_containers=2,
    # secrets=[modal.Secret.from_name("litlens-config")],
)
@modal.concurrent(max_inputs=20)  # Higher concurrency since no GPU bottleneck
class RAGServiceLite:
    """Modal class hosting mock RAG pipeline + FastAPI (CPU only)."""
    
    persistence = None
    _generate_id = None

    @modal.enter()
    async def startup(self):
        """Initialize persistence only (no ML models)."""
        import sys
        sys.path.insert(0, MODAL_REMOTE_CODE_DIR)
        
        print("🚀 Starting RAG Backend Lite (CPU-only mock service)")
        
        # Initialize persistence
        use_dynamodb = os.environ.get("USE_DYNAMODB", "false").lower() == "true"
        aws_role_arn = os.environ.get("AWS_ROLE_ARN")
        aws_region = os.environ.get("AWS_REGION", "us-east-2")
        
        if use_dynamodb and aws_role_arn:
            print(f"Initializing DynamoDB with OIDC...")
            print(f"  Role ARN: {aws_role_arn}")
            print(f"  Region: {aws_region}")
            
            try:
                from dynamo_persistence import DynamoPersistence, TraceBuilder, _generate_id
                self.persistence = DynamoPersistence.from_oidc(
                    role_arn=aws_role_arn,
                    region=aws_region,
                )
                self._generate_id = _generate_id
                self._TraceBuilder = TraceBuilder
                print("✓ DynamoDB initialized")
            except Exception as e:
                print(f"ERROR: DynamoDB init failed: {e}")
                self.persistence = None
                self._TraceBuilder = None
                # Fallback ID generator
                import uuid
                from datetime import datetime
                self._generate_id = lambda: f"{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')[:-3]}_{uuid.uuid4().hex[:8]}"
        else:
            if use_dynamodb:
                print("WARNING: USE_DYNAMODB=true but AWS_ROLE_ARN not set")
            print("Running without persistence")
            # Fallback ID generator
            import uuid
            from datetime import datetime
            self._generate_id = lambda: f"{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')[:-3]}_{uuid.uuid4().hex[:8]}"
            self._TraceBuilder = None
        
        print("✓ RAG Backend Lite ready (mock mode)")

    @modal.asgi_app()
    def web_app(self):
        """FastAPI application."""
        
        web_app = FastAPI(
            title="RAG Backend Lite API",
            version="2.1.0-lite",
            description="CPU-only mock service for development and testing",
        )
        
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
            """Parse the X-User-Anonymous header."""
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
                "mode": "lite",
                "pipeline_ready": True,  # Always ready in mock mode
                "persistence": "dynamodb" if service.persistence else "none",
                "queue_depth": 0,
            }

        @web_app.post("/v1/chat/completions")
        async def chat_completions(
            request: ChatCompletionRequest,
            token: str = Depends(verify_token),
            x_user_id: Optional[str] = Header(None),
            x_user_anonymous: Optional[str] = Header(None),
        ):
            """
            OpenAI-compatible chat completions endpoint (mock version).
            
            Uses the same streaming logic as the full backend - just yields
            mock events instead of real pipeline events.
            """
            user_id = x_user_id or "anonymous"
            is_anonymous = parse_anonymous_header(x_user_anonymous)
            
            # Extract query from messages
            user_messages = [m for m in request.messages if m.get("role") == "user"]
            if not user_messages:
                raise HTTPException(status_code=400, detail="No user message provided")
            query = user_messages[-1].get("content", "")

            # Generate IDs
            completion_id = f"chatcmpl-{service._generate_id()}"
            message_id = service._generate_id()
            created = int(time.time())
            model = request.model or "litlens-rag"
            conversation_id = request.conversation_id or "default-session"
            
            print(f"[MOCK] Query: user={user_id}, anon={is_anonymous}, conv={conversation_id}")
            print(f"--- START STREAM: {conversation_id} ---")

            # Mock pipeline event generator (replaces service.pipeline.answer_stream)
            async def mock_answer_stream(trace=None):
                """Yields events in the same format as the real pipeline."""
                start_time = time.perf_counter()
                
                # Simulate retrieval delay
                await asyncio.sleep(0.1)
                
                # Add mock retrieval stages to trace
                if trace:
                    trace.add_stage("bm25", {"total": 50, "mock": True}, 45.0, papers_retrieved=50)
                    trace.add_stage("embedding_paper", {"total": 50, "mock": True}, 120.0)
                    trace.add_stage("hybrid_fusion", {"mock": True}, 2.0)
                    trace.add_stage("embedding_chunk", {"total": 100, "mock": True}, 80.0, chunks_retrieved=100)
                    trace.add_stage("reranker", {"mock": True}, 150.0, chunks_reranked=5)
                
                # 1. Context event (sources)
                yield {
                    "type": "context",
                    "data": MOCK_SOURCES_FULL,
                }
                
                # 2. Token events (stream the answer word by word)
                mock_answer = generate_mock_answer(query, num_sentences=5)
                words = mock_answer.split()
                for i, word in enumerate(words):
                    token = word + (" " if i < len(words) - 1 else "")
                    yield {"type": "token", "content": token}
                    await asyncio.sleep(random.uniform(0.02, 0.05))
                
                # Add generator stage to trace
                if trace:
                    trace.add_stage("generator", {"answer_length": len(mock_answer), "mock": True}, 500.0)
                
                # 3. Hallucination event (if enabled)
                hal_result = None
                if request.enable_hallucination_check:
                    await asyncio.sleep(0.1)
                    hal_result = {
                        "type": "hallucination",
                        "grounding_ratio": 0.29,
                        "num_claims": 7,
                        "num_grounded": 2,
                        "unsupported_claims": [
                            "The study used MSI to analyze the differential abundance of ions between treatment and control populations.",
                            "The Histology Driven Data Mining study applied PLS-DA to analyze lipid differences.",
                            "MSI data was processed to differentiate between preselected regions of interest.",
                            "These examples show how multi-tissue MSI experiments can detect differentially abundant ions.",
                            "Appropriate statistical methods and experimental design are important in MSI studies.",
                        ],
                    }
                    yield hal_result
                    
                    if trace:
                        trace.add_stage("hallucination", {"mock": True, "grounding_ratio": 0.29}, 200.0, grounding_ratio=0.29)
                
                # 4. Done event
                total_duration = (time.perf_counter() - start_time) * 1000
                yield {
                    "type": "done",
                    "message_id": message_id,
                    "chat_id": conversation_id,
                    "total_duration_ms": total_duration,
                    # Pass back data needed for persistence
                    "_answer": mock_answer,
                    "_hallucination_result": hal_result,
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
                    trace = None
                    accumulated_answer = ""
                    sources_for_storage = []
                    hallucination_result = None
                    
                    try:
                        # Create and save trace start if persistence available
                        if service.persistence and service._TraceBuilder:
                            trace = service._TraceBuilder(
                                chat_id=conversation_id,
                                user_id=user_id,
                                query=query,
                                is_anonymous=is_anonymous,
                                message_id=message_id,
                            )
                            await service.persistence.save_trace_start(trace)
                        
                        # This mirrors the exact logic from the full backend
                        yield text_start()
                        async for event in mock_answer_stream(trace=trace):
                            if event.get("type") not in ["token"]:
                                print(f"[DEBUG] Pipeline Event: {event.get('type')}")

                            if event.get("type") == "token":
                                content = event.get("content", "")
                                accumulated_answer += content
                                yield text_delta(content)
                            
                            elif event.get("type") == "context":
                                sources_for_storage = event.get("data", [])
                                seen = set()
                                for source in sources_for_storage:
                                    metadata = source.get("metadata", {})
                                    title = metadata.get("paper_title") or metadata.get("file_path")
                                    if not title or title in seen:
                                        continue
                                    seen.add(title)
                                    url = metadata.get("url") or metadata.get("file_path") or title
                                    yield source_part(url, title)
                                
                            elif event.get("type") == "hallucination":
                                print("[DEBUG] Yielding Hallucination Chunk")
                                hallucination_result = {
                                    "grounding_ratio": event.get("grounding_ratio"),
                                    "num_claims": event.get("num_claims"),
                                    "num_grounded": event.get("num_grounded"),
                                    "unsupported_claims": event.get("unsupported_claims"),
                                }
                                yield data_part(
                                    "verification",
                                    hallucination_result,
                                )
                            
                            elif event.get("type") == "done":
                                print("[DEBUG] Yielding Done Chunk")
                                # Get final data from done event
                                if "_hallucination_result" in event and event["_hallucination_result"]:
                                    hallucination_result = event["_hallucination_result"]
                                
                                yield data_part(
                                    "completion",
                                    {
                                        "message_id": event.get("message_id"),
                                        "chat_id": event.get("chat_id"),
                                        "total_duration_ms": event.get("total_duration_ms", 0),
                                    },
                                )
                        
                        # Save to DynamoDB if available
                        if service.persistence and trace:
                            trace.complete()
                            await service.persistence.save_turn(
                                trace=trace,
                                answer=accumulated_answer,
                                sources=sources_for_storage,
                                hallucination_result=hallucination_result,
                            )
                            print(f"[MOCK] Saved to DynamoDB: message_id={message_id}")
                        
                        print("--- END STREAM ---")
                        yield text_end()
                        yield "data: [DONE]\n\n"
                    
                    except Exception as e:
                        print(f"ERROR in Stream: {e}")
                        # Mark trace as failed and save
                        if trace:
                            trace.fail("streaming", str(e))
                            if service.persistence:
                                await service.persistence.save_turn(
                                    trace=trace,
                                    answer=accumulated_answer,
                                    sources=sources_for_storage,
                                    hallucination_result=None,
                                )
                        yield error_part(str(e))
                        yield "data: [DONE]\n\n"

                return StreamingResponse(
                    stream(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                        VERCEL_STREAM_HEADER: "v1",
                    },
                )
            else:
                # Non-streaming: collect all events
                trace = None
                answer = ""
                sources = []
                hallucination_result = None
                total_duration_ms = 0
                
                # Create trace if persistence available
                if service.persistence and service._TraceBuilder:
                    trace = service._TraceBuilder(
                        chat_id=conversation_id,
                        user_id=user_id,
                        query=query,
                        is_anonymous=is_anonymous,
                        message_id=message_id,
                    )
                    await service.persistence.save_trace_start(trace)
                
                async for event in mock_answer_stream(trace=trace):
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
                        }
                    elif event.get("type") == "done":
                        total_duration_ms = event.get("total_duration_ms", 0)
                
                # Save to DynamoDB if available
                if service.persistence and trace:
                    trace.complete()
                    await service.persistence.save_turn(
                        trace=trace,
                        answer=answer,
                        sources=sources,
                        hallucination_result=hallucination_result,
                    )
                    print(f"[MOCK] Saved to DynamoDB: message_id={message_id}")
                
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
                    "sources": sources,
                    "hallucination": hallucination_result,
                    "message_id": message_id,
                    "chat_id": conversation_id,
                }

        # =================================================================
        # Chat History Endpoints (unchanged from full backend)
        # =================================================================
        @web_app.get("/get_chats")
        async def list_chats(
            token: str = Depends(verify_token),
            x_user_id: Optional[str] = Header(None),
            x_user_anonymous: Optional[str] = Header(None),
            limit: int = 20,
        ):
            """List user's recent chats (for sidebar)."""
            if not service.persistence:
                raise HTTPException(status_code=501, detail="Persistence not configured")
            
            print(f"DEBUG: Received x_user_id header: {repr(x_user_id)}")
            print(f"DEBUG: Received x_user_anonymous header: {repr(x_user_anonymous)}")
            
            user_id = x_user_id or "anonymous"
            print("User ID:", user_id)
            # is_anonymous = parse_anonymous_header(x_user_anonymous)
            
            # if is_anonymous or user_id == "anonymous":
            #     return {
            #         "chats": [],
            #         "message": "Sign in to view chat history",
            #     }
            
            chats = await service.persistence.get_user_chats(user_id=user_id, limit=limit)
            return {"chats": chats}

        @web_app.get("/chats/{chat_id}/messages")
        async def get_messages(
            chat_id: str,
            token: str = Depends(verify_token),
            x_user_id: Optional[str] = Header(None),
            limit: int = 100,
        ):
            """Get all messages for a chat."""
            if not service.persistence:
                raise HTTPException(status_code=501, detail="Persistence not configured")
            
            user_id = x_user_id or "anonymous"
            
            is_owner, actual_owner = await service.persistence.verify_chat_ownership(
                chat_id, user_id
            )
            
            if actual_owner is not None and not is_owner:
                raise HTTPException(status_code=403, detail="Access denied")
            
            messages = await service.persistence.get_chat_messages(
                chat_id=chat_id, limit=limit
            )
            return {"chat_id": chat_id, "messages": messages}

        # =================================================================
        # Trace Endpoint
        # =================================================================

        @web_app.get("/chats/{chat_id}/messages/{message_id}/trace")
        async def get_message_trace(
            chat_id: str,
            message_id: str,
            token: str = Depends(verify_token),
            x_user_id: Optional[str] = Header(None),
            x_user_anonymous: Optional[str] = Header(None),
        ):
            """Get the trace for a specific message."""
            if not service.persistence:
                raise HTTPException(status_code=501, detail="Persistence not configured")
            
            user_id = x_user_id or "anonymous"
            is_anonymous = parse_anonymous_header(x_user_anonymous)
            
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
        # Chat Title Generation (mock)
        # =================================================================

        @web_app.post("/chats/{chat_id}/generate-title")
        async def generate_chat_title(
            chat_id: str,
            request: GenerateTitleRequest,
            token: str = Depends(verify_token),
            x_user_id: Optional[str] = Header(None),
            x_user_anonymous: Optional[str] = Header(None),
        ):
            """Generate a mock title for a chat."""
            if not service.persistence:
                raise HTTPException(status_code=501, detail="Persistence not configured")
            
            user_id = x_user_id or "anonymous"
            is_anonymous = parse_anonymous_header(x_user_anonymous)
            
            if is_anonymous or user_id == "anonymous":
                raise HTTPException(
                    status_code=403,
                    detail="Sign in to generate chat titles"
                )
            
            is_owner, actual_owner = await service.persistence.verify_chat_ownership(
                chat_id, user_id
            )
            
            if actual_owner is None:
                raise HTTPException(status_code=404, detail="Chat not found")
            
            if not is_owner:
                raise HTTPException(status_code=403, detail="Access denied")
            
            if not request.queries:
                raise HTTPException(status_code=400, detail="No queries provided")
            
            # Generate mock title
            title = generate_mock_title(request.queries[:3])
            
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
                "mock": True,
            }

        @web_app.patch("/chats/{chat_id}")
        async def rename_chat(
            chat_id: str,
            request: RenameChatRequest,
            token: str = Depends(verify_token),
            x_user_id: Optional[str] = Header(None),
            x_user_anonymous: Optional[str] = Header(None),
        ):
            """Rename a chat by updating metadata title."""
            if not service.persistence:
                raise HTTPException(status_code=501, detail="Persistence not configured")

            user_id = x_user_id or "anonymous"
            is_anonymous = parse_anonymous_header(x_user_anonymous)

            if is_anonymous or user_id == "anonymous":
                raise HTTPException(status_code=403, detail="Sign in to rename chats")

            is_owner, actual_owner = await service.persistence.verify_chat_ownership(
                chat_id, user_id
            )

            if actual_owner is None:
                raise HTTPException(status_code=404, detail="Chat not found")

            if not is_owner:
                raise HTTPException(status_code=403, detail="Access denied")

            title = request.title.strip()
            if not title:
                raise HTTPException(status_code=400, detail="Title cannot be empty")

            success = await service.persistence.update_chat_title(
                user_id=user_id,
                chat_id=chat_id,
                title=title,
            )

            if not success:
                raise HTTPException(status_code=500, detail="Failed to update chat title")

            return {"chat_id": chat_id, "title": title, "renamed": True}

        @web_app.delete("/chats/{chat_id}")
        async def hide_chat(
            chat_id: str,
            token: str = Depends(verify_token),
            x_user_id: Optional[str] = Header(None),
            x_user_anonymous: Optional[str] = Header(None),
        ):
            """Soft-delete a chat (hide from the user's sidebar)."""
            if not service.persistence:
                raise HTTPException(status_code=501, detail="Persistence not configured")

            user_id = x_user_id or "anonymous"
            is_anonymous = parse_anonymous_header(x_user_anonymous)

            if is_anonymous or user_id == "anonymous":
                raise HTTPException(status_code=403, detail="Sign in to delete chats")

            is_owner, actual_owner = await service.persistence.verify_chat_ownership(
                chat_id, user_id
            )

            if actual_owner is None:
                raise HTTPException(status_code=404, detail="Chat not found")

            if not is_owner:
                raise HTTPException(status_code=403, detail="Access denied")

            success = await service.persistence.hide_chat(
                user_id=user_id,
                chat_id=chat_id,
                is_hidden=True,
            )

            if not success:
                raise HTTPException(status_code=500, detail="Failed to delete chat")

            return {"chat_id": chat_id, "hidden": True}

        # =================================================================
        # Feedback Endpoint
        # =================================================================

        @web_app.post("/feedback")
        async def submit_feedback(
            request: FeedbackRequest,
            token: str = Depends(verify_token),
            x_user_id: Optional[str] = Header(None),
            x_user_anonymous: Optional[str] = Header(None),
        ):
            """Store feedback for a specific assistant message."""
            if not service.persistence:
                raise HTTPException(status_code=501, detail="Persistence not configured")

            user_id = x_user_id or "anonymous"
            is_anonymous = parse_anonymous_header(x_user_anonymous)

            is_owner, actual_owner = await service.persistence.verify_chat_ownership(
                request.chat_id, user_id
            )
            print("chat id:", request.chat_id)
            print("user id:", user_id)
            print("message id:", request.message_id)
            if actual_owner is not None and not is_owner:
                raise HTTPException(status_code=403, detail="Access denied")

            rating = request.rating.strip().lower()
            if rating not in {"positive", "negative"}:
                raise HTTPException(status_code=400, detail="Invalid rating")

            success = await service.persistence.update_message_feedback(
                chat_id=request.chat_id,
                message_id=request.message_id,
                rating=rating,
                comment=request.comment,
                user_id=None if is_anonymous else user_id,
                is_anonymous=is_anonymous,
            )

            if not success:
                raise HTTPException(status_code=404, detail="Message not found")

            return {"ok": True}

        # =================================================================
        # Admin/Debug Endpoints
        # =================================================================

        @web_app.get("/admin/abandoned-traces")
        async def get_abandoned_traces(
            token: str = Depends(verify_token),
            limit: int = 50,
        ):
            """Get traces that were abandoned (status still 'running')."""
            if not service.persistence:
                raise HTTPException(status_code=501, detail="Persistence not configured")
            
            traces = await service.persistence.get_abandoned_traces(limit=limit)
            return {
                "count": len(traces),
                "traces": traces,
            }

        return web_app