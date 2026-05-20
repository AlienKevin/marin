# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa

"""Qwen3 chat template variant that puts `{% generation %}` markers around *all*
non-system content — user, assistant, and tool — so Levanter computes loss on
the full transcript except the system prompt.

This is the SFT-mode equivalent of ECHO's environment-token prediction: in RL,
ECHO adds a length-normalized CE on terminal-output tokens alongside the GRPO
loss on action tokens; in SFT there is no separate policy-gradient term to
weight against, so the auxiliary-loss formulation collapses to "stop masking
the env tokens." Concretely, this template diverges from
:mod:`experiments.qwen3_chat_template` only in that user and tool blocks are
wrapped in `{% generation %}` rather than rendered raw.

System prompts remain masked: the mini-swe-agent system prompt is identical
across every SWE-ZERO trajectory, so including it in the loss would
overweight static boilerplate.

Used by exp5611_sft_qwen3_1_7b_swe_zero_10k_8k_echo.py (arm (b) of the
TerminalWorld Phase 0 derisk).
"""

QWEN_3_CHAT_TEMPLATE_FULL_TRANSCRIPT = r"""{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if message.role == "system" and not loop.first %}
        {{- '<|im_start|>system\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "user" %}
        {{- '<|im_start|>user\n' }}{% generation %}{{- content + '<|im_end|>' + '\n' }}{% endgeneration %}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\n' }}{% generation %}{{- '<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}{% endgeneration %}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' }}{% generation %}{{- content }}{% endgeneration %}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' }}{% generation %}{{- content }}{% endgeneration %}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {% generation %}{{- '\n' }}{% endgeneration %}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {% generation %}{{- '<tool_call>\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\n</tool_call>' }}{% endgeneration %}
            {%- endfor %}
        {%- endif %}
        {% generation %}{{- '<|im_end|>\n' }}{% endgeneration %}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n' }}{% generation %}{{- '<tool_response>\n' + content + '\n</tool_response>' }}{% endgeneration %}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {% generation %}{{- '<think>\n\n</think>\n\n' }}{% endgeneration %}
    {%- endif %}
{%- endif %}"""
