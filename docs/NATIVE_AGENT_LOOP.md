# Native Tool Calling and bounded Agent Loop

The chat Tool Agent now uses native model tool calls when an intent model is
configured and `NATIVE_TOOL_CALLING_ENABLED=true`. The existing LangGraph retains
Memory/RAG gathering, Writer and Reviewer. The inner loop sends tool schemas,
executes validated calls, appends matching ToolMessages and asks the model for its
next step. This is an explicit bounded loop, not a migration to `create_agent`.

## Implemented paths

- P0: weather, web_search, memory_snapshot, work_summary, file_parse, image_analyze
  have independent Pydantic argument contracts. The registry exposes inputSchema.
- P1: todo, calendar, image generation/editing and clarification also have typed
  contracts. Missing/invalid required fields stop for clarification. Time values
  for calendar/todo must include an offset. Slot records contain value/source/
  confidence; confidence is conservative attribution metadata, not authorization.
- Intent classification uses structured function output. Scene routing recognizes
  project analysis, research, knowledge, diagnosis, code review, deletion, batch
  file modification and email. Relevant MCP descriptions are ranked, capped at 20.
- P2: results drive subsequent calls; duplicate successful calls are suppressed.
  Retry only explicitly retryable failures, within the same tool-call budget.
  Knowledge search uses only knowledge-base IDs injected by the chat request.
- Delete/batch-edit/email operations use enabled user-owned MCP tools. They require
  an actual configured connector implementing those operations. No SQL, shell or
  SMTP endpoint is invented from model text. Built-in project summaries and health
  checks work without MCP; code/workspace inspection needs an authorized connector.
- P3: existing inspection/release/research templates remain authoritative for the
  separate long-task Planner. Unrestricted dynamic long-task planning is deferred
  as requested; the chat loop dynamically chooses tools within its catalog.

## External write confirmation

MCP write arguments are schema validated and stored in an existing pending operation.
The loop stops immediately, including any later calls in the same model response.
Confirmation revalidates the tool owner, enabled state, schema/risk fingerprint,
server connection fingerprint and current permission policy. High-risk operations
create a second, separate pending confirmation. The chat card appends that second
confirmation; the execution center refreshes its inbox. Destructive reset/drop,
payment and permission tools remain disabled. Delete/remove, batch and send tools
are classified high risk when newly discovered; previously stored `dangerous`
policies remain disabled until the user changes the tool policy.

Before remote I/O, an `executing` claim is committed. A crash or uncertain outcome
does not cause automatic replay. An executing/failed operation must be reconciled
with the external system. This prevents application retries but is not a distributed
exactly-once guarantee for third-party services. The connector determines whether
batch writes are atomic or recoverable; previews do not claim automatic rollback.

Confirmation executes the approved action. It does not resume a persisted model
checkpoint automatically; the user can continue the conversation for subsequent
steps. Existing Workflow/Agent Task recovery continues to serve durable tasks.

## Configuration

Defaults in app/core/config.py:

| Setting | Default |
| --- | --- |
| NATIVE_TOOL_CALLING_ENABLED | true |
| NATIVE_TOOL_CALLING_STRICT | false |
| AGENT_LOOP_ENABLED | true |
| AGENT_LOOP_MAX_STEPS | 5 |
| AGENT_LOOP_MAX_TOOL_CALLS | 8 |
| AGENT_LOOP_MAX_RETRIES | 1 |
| AGENT_LOOP_MAX_NO_PROGRESS | 2 |
| AGENT_LOOP_MAX_SECONDS | 120 |
| AGENT_LOOP_MCP_ENABLED | true |

Use a model supporting native tool calling. Provider strict-schema mode is opt-in
because provider support differs; server-side Pydantic/JSON Schema validation is
always applied. Disabling AGENT_LOOP_ENABLED allows one model turn; disabling
NATIVE_TOOL_CALLING_ENABLED restores the legacy rule/JSON router. A native model
error does not trigger legacy execution, avoiding duplicate or uncertain effects.

The wall-clock bound is checked between synchronous calls; in-flight work is bounded
by provider/connector timeouts. Existing UsageCallbackHandler records model usage
and enforces request token/cost budgets when supplied by the backend.

## Current integration boundaries

- Calendar participants are represented but built-in calendars cannot send invites;
  such requests clarify instead of silently discarding participants. Use a calendar MCP.
- Search time_range is added to the query, not an enforced provider-side date filter.
- Weather requests validate dates against returned short-term forecast dates.
- Image references must be present in the current enriched request; guessed image
  URLs are rejected. File references go through existing path/ownership checks.
- Raw slot values remain internal; existing diagnostic sanitization intentionally
  does not expose arbitrary tool arguments or credentials in persisted chat debug.
- No real deletion, file modification or email delivery was performed during testing.

## Verification

tests/test_native_tool_loop.py covers native ChatOpenAI wire format using a fake HTTP
transport, multi-step observation, bounded retries, confirmation pause, unknown tool
rejection, knowledge scope, typed slots, exact file paths, MCP schema/config changes,
separate approvals, and durable claim/replay behavior. Existing MCP stdio and HTTP
fixture servers test real local protocol interoperability without external services.
