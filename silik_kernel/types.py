from typing import (
    TypedDict,
    Dict,
    List,
    Literal,
    Optional,
    Any,
    Annotated,
    Union,
    TypeVar,
    Generic,
)


class ExecutionResult(TypedDict):
    """
    Standardized execution result of jupyter kernels. Must be returned
    by do_execute methods of IPykernel wrappers. See :
    https://jupyter-client.readthedocs.io/en/stable/messaging.html#execution-results
    """

    status: Literal["ok", "error", "aborted"]
    execution_count: int
    payload: list[dict]
    user_expressions: dict


"""
Complete TypedDict models for Jupyter messaging protocol content fields
using typing.Annotated for inline documentation.

Covers message types across:
- Shell channel
- Control channel
- IOPub channel
- Stdin channel
- Comm messages
- Debug messages
- History
- Completeness

Spec reference: Jupyter messaging protocol.
"""


# ---------------------------------------------------------------------------
# Shared aliases
# ---------------------------------------------------------------------------

MimeBundle = Annotated[Dict[str, Any], "Mapping of MIME type -> representation"]

Metadata = Annotated[Dict[str, Any], "Arbitrary metadata dictionary"]

JSONDict = Annotated[Dict[str, Any], "Generic JSON object"]


# ---------------------------------------------------------------------------
# IOPub messages
# ---------------------------------------------------------------------------


class StreamContent(TypedDict):
    """stdout/stderr output."""

    name: Annotated[Literal["stdout", "stderr"], "Stream name"]
    text: Annotated[str, "Text emitted"]


class DisplayDataContent(TypedDict):
    """Rich display output."""

    data: Annotated[MimeBundle, "MIME bundle"]
    metadata: Annotated[Metadata, "Metadata"]
    transient: Annotated[Optional[JSONDict], "Transient data"]


class UpdateDisplayDataContent(TypedDict):
    """Update existing display."""

    data: Annotated[MimeBundle, "MIME bundle"]
    metadata: Annotated[Metadata, "Metadata"]
    transient: Annotated[JSONDict, "Must contain display_id"]


class ExecuteInputContent(TypedDict):
    """Code execution started."""

    code: Annotated[str, "Code being executed"]
    execution_count: Annotated[int, "Execution counter"]


class ExecuteResultContent(TypedDict):
    """Execution result."""

    execution_count: Annotated[int, "Execution counter"]
    data: Annotated[MimeBundle, "Result MIME bundle"]
    metadata: Annotated[Metadata, "Metadata"]


class ErrorContent(TypedDict):
    """Execution error."""

    ename: Annotated[str, "Exception name"]
    evalue: Annotated[str, "Exception message"]
    traceback: Annotated[List[str], "Traceback lines"]


class ClearOutputContent(TypedDict, total=False):
    """Clear output request."""

    wait: Annotated[bool, "Wait for new output before clearing"]


class StatusContent(TypedDict):
    """Kernel execution state."""

    execution_state: Annotated[Literal["busy", "idle", "starting"], "Kernel state"]


# ---------------------------------------------------------------------------
# stdin channel
# ---------------------------------------------------------------------------


class InputRequestContent(TypedDict):
    """Kernel requests user input."""

    prompt: Annotated[str, "Prompt text"]
    password: Annotated[bool, "Whether input should be hidden"]


class InputReplyContent(TypedDict):
    """Frontend reply with input."""

    value: Annotated[str, "User input value"]


# ---------------------------------------------------------------------------
# History messages
# ---------------------------------------------------------------------------


class HistoryRequestContent(TypedDict, total=False):
    """Request execution history."""

    output: Annotated[bool, "Include output history"]
    raw: Annotated[bool, "Return raw input"]
    hist_access_type: Annotated[str, "range | tail | search"]
    session: Annotated[int, "Session number"]
    start: Annotated[int, "Start line"]
    stop: Annotated[int, "Stop line"]
    n: Annotated[int, "Number of entries"]
    pattern: Annotated[str, "Search pattern"]
    unique: Annotated[bool, "Exclude duplicates"]


class HistoryReplyContent(TypedDict):
    """History response."""

    history: Annotated[List[Any], "History entries"]


# ---------------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------------


class IsCompleteRequestContent(TypedDict):
    """Check if code is complete."""

    code: Annotated[str, "Code to check"]


class IsCompleteReplyContent(TypedDict, total=False):
    """Completeness result."""

    status: Annotated[
        Literal["complete", "incomplete", "invalid", "unknown"], "Completion status"
    ]
    indent: Annotated[str, "Suggested indent if incomplete"]


# ---------------------------------------------------------------------------
# Comm messages
# ---------------------------------------------------------------------------


class CommOpenContent(TypedDict):
    """Open a comm."""

    comm_id: Annotated[str, "Comm UUID"]
    target_name: Annotated[str, "Target name"]
    data: Annotated[JSONDict, "Initialization data"]


class CommMsgContent(TypedDict):
    """Comm message."""

    comm_id: Annotated[str, "Comm UUID"]
    data: Annotated[JSONDict, "Payload"]


class CommCloseContent(TypedDict):
    """Close comm."""

    comm_id: Annotated[str, "Comm UUID"]
    data: Annotated[JSONDict, "Optional data"]


# ---------------------------------------------------------------------------
# Debug messages
# ---------------------------------------------------------------------------


class DebugRequestContent(TypedDict):
    """
    Debug request following Debug Adapter Protocol.
    Structure varies depending on command.
    """

    command: Annotated[Optional[str], "DAP command"]
    arguments: Annotated[Optional[JSONDict], "DAP arguments"]


class DebugReplyContent(TypedDict):
    """Debug reply."""

    success: Annotated[Optional[bool], "DAP success flag"]
    body: Annotated[Optional[JSONDict], "Response body"]


class DebugEventContent(TypedDict):
    """Debug event."""

    event: Annotated[Optional[str], "DAP event"]
    body: Annotated[Optional[JSONDict], "Event body"]


# ---------------------------------------------------------------------------
# Generic / unspecified (empty content in spec)
# ---------------------------------------------------------------------------


class EmptyContent(TypedDict):
    """Used for messages with unspecified content."""

    pass


class ExecuteRequestContent(TypedDict):
    """
    Content of an `execute_request` message.

    Sent by the frontend to request code execution in the kernel.
    """

    code: Annotated[str, "Code to execute"]

    silent: Annotated[bool, "If True, do not broadcast output on IOPub"]

    store_history: Annotated[bool, "Whether to store command in history"]

    user_expressions: Annotated[
        Optional[Dict[str, str]], "Expressions to evaluate after execution"
    ]

    allow_stdin: Annotated[bool, "Whether kernel may prompt for stdin"]

    stop_on_error: Annotated[bool, "Stop execution queue on error"]


"""
Union groupings for Jupyter message content types by channel.

These unions are useful for:
- Typed message dispatchers
- Pattern matching
- Handler registries
- Static typing of message routers
"""

# ---------------------------------------------------------------------------
# IOPub channel
# ---------------------------------------------------------------------------

IOPubMsg = Union[
    StreamContent,
    DisplayDataContent,
    UpdateDisplayDataContent,
    ExecuteInputContent,
    ExecuteResultContent,
    ErrorContent,
    ClearOutputContent,
    StatusContent,
]
"""
All message content types sent over the IOPub channel.
"""


# ---------------------------------------------------------------------------
# stdin channel
# ---------------------------------------------------------------------------

StdinMsg = Union[
    InputRequestContent,
    InputReplyContent,
]
"""
Messages related to interactive input.
"""


# ---------------------------------------------------------------------------
# shell channel
# ---------------------------------------------------------------------------

ShellMsg = Union[
    HistoryRequestContent,
    HistoryReplyContent,
    IsCompleteRequestContent,
    IsCompleteReplyContent,
    ExecuteRequestContent,
]
"""
Shell channel request/reply message contents.
"""


# ---------------------------------------------------------------------------
# comm messages
# ---------------------------------------------------------------------------

CommMsg = Union[
    CommOpenContent,
    CommMsgContent,
    CommCloseContent,
]
"""
Widget / extension communication messages.
"""


# ---------------------------------------------------------------------------
# debug messages
# ---------------------------------------------------------------------------

DebugMsg = Union[
    DebugRequestContent,
    DebugReplyContent,
    DebugEventContent,
]
"""
Debug Adapter Protocol messages.
"""


# ---------------------------------------------------------------------------
# Global union (all known content types)
# ---------------------------------------------------------------------------

AnyMessageContent = Union[
    IOPubMsg,
    StdinMsg,
    ShellMsg,
    CommMsg,
    DebugMsg,
    EmptyContent,
]
"""
Union of all defined message content schemas.
"""


MSG_TYPE_TO_CONTENT = {
    # =========================
    # IOPub
    # =========================
    "stream": StreamContent,
    "display_data": DisplayDataContent,
    "update_display_data": UpdateDisplayDataContent,
    "execute_input": ExecuteInputContent,
    "execute_result": ExecuteResultContent,
    "error": ErrorContent,
    "clear_output": ClearOutputContent,
    "status": StatusContent,
    # =========================
    # stdin
    # =========================
    "input_request": InputRequestContent,
    "input_reply": InputReplyContent,
    # =========================
    # shell / history
    # =========================
    "execute_request": ExecuteRequestContent,
    "history_request": HistoryRequestContent,
    "history_reply": HistoryReplyContent,
    "is_complete_request": IsCompleteRequestContent,
    "is_complete_reply": IsCompleteReplyContent,
    # =========================
    # comm
    # =========================
    "comm_open": CommOpenContent,
    "comm_msg": CommMsgContent,
    "comm_close": CommCloseContent,
    # =========================
    # debug
    # =========================
    "debug_request": DebugRequestContent,
    "debug_reply": DebugReplyContent,
    "debug_event": DebugEventContent,
}


class MessageHeader(TypedDict):
    """Jupyter message header, contains identifiers and msg type."""

    msg_id: str
    username: str
    session: str
    msg_type: str
    version: str


MsgType = Literal[
    "stream",
    "display_data",
    "update_display_data",
    "execute_input",
    "execute_result",
    "error",
    "clear_output",
    "status",
    "input_request",
    "input_reply",
    "execute_request",
    "history_request",
    "history_reply",
    "is_complete_request",
    "is_complete_reply",
    "comm_open",
    "comm_msg",
    "comm_close",
    "debug_request",
    "debug_reply",
    "debug_event",
]

T = TypeVar("T", bound=Dict[str, Any])


class JupyterMessage(TypedDict, Generic[T]):
    """
    TypedDict for a full Jupyter message envelope.

    Fields:
    - header: contains msg_id, msg_type, and other identifiers
    - msg_id: top-level copy of header['msg_id']
    - msg_type: top-level copy of header['msg_type']
    - parent_header: header of the parent message (empty dict if none)
    - content: message-specific content, typed by T
    - metadata: arbitrary metadata dict
    - buffers: optional binary buffers
    """

    header: MessageHeader
    msg_id: str
    msg_type: MsgType
    parent_header: Dict[str, Any]
    content: T
    metadata: Dict[str, Any]
    buffers: List[bytes]
