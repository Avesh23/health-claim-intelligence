from slowapi import Limiter
from slowapi.util import get_remote_address

# Default limit of 5 requests per minute from a single IP
limiter = Limiter(key_func=get_remote_address, default_limits=["5/minute"])
