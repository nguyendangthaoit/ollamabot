import phoenix as px
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter


def init_telemetry():
    """Initializes Arize Phoenix and OpenTelemetry tracking globally."""
    print("[SYSTEM] Initializing Phoenix observability instrumentation...")

    # Launch Phoenix backend
    px.launch_app()

    # Setup the OpenTelemetry Tracer Provider
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(
        SimpleSpanProcessor(
            OTLPSpanExporter(endpoint="http://localhost:6006/v1/traces")
        )
    )
    otel_trace.set_tracer_provider(tracer_provider)

    # Automatically instrument LangChain and LangGraph objects
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
