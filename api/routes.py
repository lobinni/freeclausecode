"""FastAPI route handlers."""

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from loguru import logger

from config.settings import Settings
from core.anthropic import get_token_count
from core.trace import trace_event
from providers.registry import ProviderRegistry

from . import dependencies
from .dependencies import get_settings, require_api_key
from .gateway_model_ids import decode_gateway_model_id, gateway_model_id, no_thinking_gateway_model_id
from .models.anthropic import MessagesRequest, TokenCountRequest
from .models.openai_responses import OpenAIResponsesRequest
from .models.responses import ModelResponse, ModelsListResponse
from .services import ClaudeProxyService

router = APIRouter()

DISCOVERED_MODEL_CREATED_AT = "1970-01-01T00:00:00Z"


def _normalize_model_ref(model_ref: str) -> str:
    """Normalize a model reference by stripping gateway ID prefixes.
    Returns the underlying provider/model format for gateway IDs, or the original string otherwise.
    """
    decoded = decode_gateway_model_id(model_ref)
    if decoded:
        return f"{decoded.provider_id}/{decoded.provider_model}"
    return model_ref


def _is_free_model(model_ref: str, settings: Settings) -> bool:
    """Return True if the model is considered free.
    Free if:
    - model_ref ends with ":free", "-free", or "/free"
    - model_ref is in the user-provided free models list (with gateway ID normalization)
    """
    if (
        model_ref.endswith(":free")
        or model_ref.endswith("-free")
        or model_ref.endswith("/free")
    ):
        return True

    normalized_model_ref = _normalize_model_ref(model_ref)
    for free_model in settings.free_models_list:
        if normalized_model_ref == _normalize_model_ref(free_model):
            return True

    return False


SUPPORTED_CLAUDE_MODELS = [
    ModelResponse(
        id="claude-opus-4-20250514",
        display_name="Claude Opus 4",
        created_at="2025-05-14T00:00:00Z",
    ),
    ModelResponse(
        id="claude-sonnet-4-20250514",
        display_name="Claude Sonnet 4",
        created_at="2025-05-14T00:00:00Z",
    ),
    ModelResponse(
        id="claude-haiku-4-20250514",
        display_name="Claude Haiku 4",
        created_at="2025-05-14T00:00:00Z",
    ),
    ModelResponse(
        id="claude-3-opus-20240229",
        display_name="Claude 3 Opus",
        created_at="2024-02-29T00:00:00Z",
    ),
    ModelResponse(
        id="claude-3-5-sonnet-20241022",
        display_name="Claude 3.5 Sonnet",
        created_at="2024-10-22T00:00:00Z",
    ),
    ModelResponse(
        id="claude-3-haiku-20240307",
        display_name="Claude 3 Haiku",
        created_at="2024-03-07T00:00:00Z",
    ),
    ModelResponse(
        id="claude-3-5-haiku-20241022",
        display_name="Claude 3.5 Haiku",
        created_at="2024-10-22T00:00:00Z",
    ),
]


def get_proxy_service(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> ClaudeProxyService:
    """Build the request service for route handlers."""
    return ClaudeProxyService(
        settings,
        provider_getter=lambda provider_type: dependencies.resolve_provider(
            provider_type, app=request.app, settings=settings
        ),
        token_counter=get_token_count,
    )


def _probe_response(allow: str) -> Response:
    """Return an empty success response for compatibility probes."""
    return Response(status_code=204, headers={"Allow": allow})


def _discovered_model_response(
    model_id: str, *, display_name: str, is_free: bool = False, supports_thinking: bool = False
) -> ModelResponse:
    if not supports_thinking:
        display_name = f"{display_name} (no thinking)"
    if is_free:
        display_name = f"{display_name} (free)"
    return ModelResponse(
        id=model_id,
        display_name=display_name,
        created_at=DISCOVERED_MODEL_CREATED_AT,
    )


def _append_unique_model(
    models: list[ModelResponse],
    seen: set[str],
    model: ModelResponse,
) -> None:
    if model.id in seen:
        return
    seen.add(model.id)
    models.append(model)


def _append_provider_model_variants(
    models: list[ModelResponse],
    seen: set[str],
    provider_model_ref: str,
    *,
    supports_thinking: bool | None = None,
    is_free: bool = False,
) -> None:
    if supports_thinking is not False:
        _append_unique_model(
            models,
            seen,
            _discovered_model_response(
                gateway_model_id(provider_model_ref),
                display_name=provider_model_ref,
                is_free=is_free,
                supports_thinking=True
            ),
        )
    _append_unique_model(
        models,
        seen,
        _discovered_model_response(
            no_thinking_gateway_model_id(provider_model_ref),
            display_name=provider_model_ref,
            is_free=is_free,
            supports_thinking=False
        ),
    )


def _build_models_list_response(
    settings: Settings, provider_registry: ProviderRegistry | None
) -> ModelsListResponse:
    models: list[ModelResponse] = []
    free_models: list[ModelResponse] = []
    seen: set[str] = set()

    # Determine whether to show free models first
    show_free_models_first = settings.show_free_models_first
    # Determine whether to only show free models
    only_show_free_models = settings.only_show_free_models

    for ref in settings.configured_chat_model_refs():
        supports_thinking = None
        if provider_registry is not None:
            supports_thinking = provider_registry.cached_model_supports_thinking(
                ref.provider_id, ref.model_id
            )
        is_free = _is_free_model(ref.model_ref, settings)
        if only_show_free_models and not is_free:
            continue
        _append_provider_model_variants(
            free_models if show_free_models_first and is_free else models,
            seen,
            ref.model_ref,
            supports_thinking=supports_thinking,
            is_free=is_free,
        )

    if provider_registry is not None:
        for model_info in provider_registry.cached_prefixed_model_infos():
            is_free = _is_free_model(model_info.model_id, settings)
            if only_show_free_models and not is_free:
                continue
            _append_provider_model_variants(
                free_models if show_free_models_first and is_free else models,
                seen,
                model_info.model_id,
                supports_thinking=model_info.supports_thinking,
                is_free=is_free,
            )

    for model in SUPPORTED_CLAUDE_MODELS:
        is_free = _is_free_model(model.id, settings)
        if only_show_free_models and not is_free:
            continue
        claude_model = _discovered_model_response(model_id=model.id, display_name=model.display_name, is_free=is_free, supports_thinking=True)
        claude_model.created_at = model.created_at
        _append_unique_model(
            free_models if show_free_models_first and is_free else models,
            seen,
            claude_model,
        )

    combined_models = free_models + models

    return ModelsListResponse(
        data=combined_models,
        first_id=combined_models[0].id if combined_models else None,
        has_more=False,
        last_id=combined_models[-1].id if combined_models else None,
    )


# =============================================================================
# Routes
# =============================================================================
@router.post("/v1/messages")
async def create_message(
    request_data: MessagesRequest,
    service: ClaudeProxyService = Depends(get_proxy_service),
    _auth=Depends(require_api_key),
):
    """Create a message (always streaming)."""
    return service.create_message(request_data)


@router.api_route("/v1/messages", methods=["HEAD", "OPTIONS"])
async def probe_messages(_auth=Depends(require_api_key)):
    """Respond to Claude compatibility probes for the messages endpoint."""
    return _probe_response("POST, HEAD, OPTIONS")


@router.post("/v1/responses")
async def create_response(
    request_data: OpenAIResponsesRequest,
    service: ClaudeProxyService = Depends(get_proxy_service),
    _auth=Depends(require_api_key),
):
    """Create an OpenAI Responses-compatible response through this proxy."""
    return await service.create_response(request_data)


@router.api_route("/v1/responses", methods=["HEAD", "OPTIONS"])
async def probe_responses(_auth=Depends(require_api_key)):
    """Respond to OpenAI Responses compatibility probes."""
    return _probe_response("POST, HEAD, OPTIONS")


@router.post("/v1/messages/count_tokens")
async def count_tokens(
    request_data: TokenCountRequest,
    service: ClaudeProxyService = Depends(get_proxy_service),
    _auth=Depends(require_api_key),
):
    """Count tokens for a request."""
    return service.count_tokens(request_data)


@router.api_route("/v1/messages/count_tokens", methods=["HEAD", "OPTIONS"])
async def probe_count_tokens(_auth=Depends(require_api_key)):
    """Respond to Claude compatibility probes for the token count endpoint."""
    return _probe_response("POST, HEAD, OPTIONS")


@router.get("/")
async def root(
    settings: Settings = Depends(get_settings), _auth=Depends(require_api_key)
):
    """Root endpoint."""
    return {
        "status": "ok",
        "provider": settings.provider_type,
        "model": settings.model,
    }


@router.api_route("/", methods=["HEAD", "OPTIONS"])
async def probe_root():
    """Respond to unauthenticated local compatibility probes for the root endpoint."""
    return _probe_response("GET, HEAD, OPTIONS")


@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@router.api_route("/health", methods=["HEAD", "OPTIONS"])
async def probe_health():
    """Respond to compatibility probes for the health endpoint."""
    return _probe_response("GET, HEAD, OPTIONS")


@router.get("/v1/models", response_model=ModelsListResponse)
async def list_models(
    request: Request,
    settings: Settings = Depends(get_settings),
    _auth=Depends(require_api_key),
):
    """List the model ids this proxy advertises to Claude-compatible clients."""
    trace_event(stage="ingress", event="api.models.list", source="api")
    registry = getattr(request.app.state, "provider_registry", None)
    provider_registry = registry if isinstance(registry, ProviderRegistry) else None
    return _build_models_list_response(settings, provider_registry)


@router.post("/stop")
async def stop_cli(request: Request, _auth=Depends(require_api_key)):
    """Stop all CLI sessions and pending tasks."""
    handler = getattr(request.app.state, "message_handler", None)
    if not handler:
        # Fallback if messaging not initialized
        cli_manager = getattr(request.app.state, "cli_manager", None)
        if cli_manager:
            await cli_manager.stop_all()
            logger.info("STOP_CLI: source=cli_manager cancelled_count=N/A")
            return {"status": "stopped", "source": "cli_manager"}
        raise HTTPException(status_code=503, detail="Messaging system not initialized")

    count = await handler.stop_all_tasks()
    trace_event(
        stage="ingress",
        event="api.cli.stop_via_handler",
        source="api",
        cancelled_nodes=count,
    )
    logger.info("STOP_CLI: source=handler cancelled_count={}", count)
    return {"status": "stopped", "cancelled_count": count}
