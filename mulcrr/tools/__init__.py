from mulcrr.tools.base import Tool
from mulcrr.tools.info_database import InfoDatabase
from mulcrr.tools.interaction import InteractionRetriever

TOOL_MAP: dict[str, type] = {
    'info': InfoDatabase,
    'interaction': InteractionRetriever,
}