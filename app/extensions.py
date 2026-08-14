"""Shared extension instances, created here (uninitialized) so blueprint
modules can import them without triggering circular imports with the app
factory in __init__.py. Bound to the app via .init_app() in create_app()."""

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(key_func=get_remote_address, default_limits=[])
