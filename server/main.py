"""ExamGuard server entrypoint - implements docs/api-contract.md"""

from fastapi import FastAPI

from server.routes import auth_check as auth_check_routes
from server.routes import clip as clip_routes
from server.routes import health as health_routes
from server.routes import metadata as metadata_routes
from server.routes import session as session_routes

app = FastAPI(title="ExamGuard Server")
app.include_router(auth_check_routes.router)
app.include_router(health_routes.router)
app.include_router(session_routes.router)
app.include_router(metadata_routes.router)
app.include_router(clip_routes.router)
