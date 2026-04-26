# Copyright 2025 Igor Bogdanov
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
from typing import Dict, Any, Set
import os
import json

from .loader import load_template
from .exceptions import MissingPlaceholderError, TemplateNotFoundError


class PromptBuilder:
    def __init__(self, agent_definition_path: str):
        self.agent_template: Dict[str, Any] = load_template(agent_definition_path)
        self.agent_type: str = self.agent_template.get("agent_type", "default")

        # The templates are now located inside the 'builder' package,
        # so we construct the path relative to this file.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        instruction_path = os.path.join(
            current_dir,
            "templates",
            "instructions",
            f"{self.agent_type.lower()}.inst.yaml",
        )
        self.instruction_template: Dict[str, Any] = load_template(instruction_path)

    def build(self, **runtime_kwargs) -> str:
        build_method = getattr(
            self, f"_build_{self.agent_type.lower()}_prompt", self._build_default_prompt
        )
        return build_method(**runtime_kwargs)

    def _build_react_prompt(self, **runtime_kwargs) -> str:
        parts = []

        # 1. Preamble (Role + Instructions + Answer Format)
        preamble_parts = []
        preamble_parts.append(self.agent_template.get("system_message", ""))

        mode_of_operation = "IMPORTANT NOTE: The XML-like tags in this prompt are for context and structure. DO NOT include them in your response.\n\n"

        mode_of_operation += self.instruction_template.get("instructions", "")
        preamble_parts.append(mode_of_operation)
        answer_format = self.agent_template.get("answer_format")
        if answer_format:
            preamble_parts.append(
                f"YOU MUST PROVIDE FINAL ANSWER IN SPECIFIC FORMAT SHOWN BELOW:\n{answer_format}"
            )
        parts.append("\n\n".join(preamble_parts))

        # 2. History (if provided)
        if "history" in runtime_kwargs:
            parts.append(runtime_kwargs["history"])

        # 3. Chain of Thought (if provided)
        include_cot = self.agent_template.get("include_COT_instruction", False)
        cot_instruction = self.agent_template.get("COT_instruction")

        # 4. Tools
        tools = self.agent_template.get("tools", [])

        # Conditionally add tools based on 'include_tool_*' flags
        for key, value in list(self.agent_template.items()):

            if key.startswith("include_tool_") and value:
                tool_name = key.replace("include_tool_", "")
                tool_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "templates",
                    "tools",
                    f"{tool_name}.tool.yaml",
                )
                try:
                    tool_definitions = load_template(tool_path)
                    tools.extend(tool_definitions)
                    # Check if the included tool has its own examples and merge them
                    for tool_def in tool_definitions:
                        if "examples" in tool_def:
                            self.agent_template.setdefault("examples", []).extend(
                                tool_def["examples"]
                            )
                except TemplateNotFoundError:
                    # You might want to log this as a warning or handle it as needed
                    print(
                        f"Warning: Tool definition not found for '{tool_name}' at {tool_path}"
                    )

        if tools:
            tool_strings = []
            for tool in tools:
                tool_str = f"<tool>\n{tool['name']}\ne.g. {tool['example_calling']}\n\n{tool['description']}"

                if tool.get("is_critical"):
                    answer_format = self.agent_template.get("answer_format", "")
                    tool_str += f"\n\n<loop_rules>This tool is critical. If you decide to call it, you must provide Answer immediately after calling this tool. After you receive the Observation from this tool, your ONLY next step is to output the final Answer.</loop_rules>"
                tool_strings.append(tool_str)
                tool_strings.append("</tool>\n")

            reflection_knowledge = self.agent_template.get("reflection_knowledge")
            intro_sentence = "Your available tools are in <tools_list>"
            if reflection_knowledge:
                intro_sentence += (
                    " and your guiding principles are in <reflection_knowledge>"
                )

            tools_section = (
                f"{intro_sentence}\n <tools_list>\n"
                + "\n".join(tool_strings)
                + "\nNo other actions are available to you."
            )
            parts.append(tools_section)
            parts.append("</tools_list>")
        else:
            # Explicitly tell the LLM it has no tools to prevent hallucination
            no_tools_message = (
                "CRITICAL: You have NO external tools available. "
                "Do NOT attempt to use 'Tool:' syntax or call any tools. "
                "You must provide your Answer directly in the required format."
            )
            parts.append(no_tools_message)

        # 2. Reflection Knowledge (if provided)
        reflection_knowledge = self.agent_template.get("reflection_knowledge")
        if reflection_knowledge:
            processed_blocks = []
            # This is now a list of knowledge blocks, which can be lists or dicts
            for block in reflection_knowledge:
                if isinstance(block, dict) and "header" in block:
                    header = block.get("header")
                    content_type = block.get("type", "plain")
                    content = block.get("content", [])
                    root_key = block.get("root")

                    block_str = f"{header}\n"
                    if content_type == "json":
                        if root_key:
                            # Create a single JSON object with the specified root key
                            json_obj = {root_key: content}
                            block_str += json.dumps(json_obj, indent=2)
                        else:
                            # Fallback to formatting each item as a JSON line
                            block_str += "\n".join(
                                f"- {json.dumps(item)}" for item in content
                            )
                    else:  # plain text
                        block_str += "\n".join(f"- {item}" for item in content)
                    processed_blocks.append(block_str)
                elif isinstance(block, dict):
                    processed_blocks.append(f"- {json.dumps(block)}")
                else:  # It's a plain string
                    processed_blocks.append(f"- {str(block)}")

            reflection_knowledge_str = "\n".join(processed_blocks)

            enforcement_statement = (
                "CRITICAL: When using any tool, you MUST ground your reasoning in the following reflection knowledge. "
                "Your thought process must explicitly reference these principles."
            )
            reflection_section = f"\n\n<reflection_knowledge>\n{enforcement_statement}\n{reflection_knowledge_str.strip()}\n</reflection_knowledge>"
            parts.append(reflection_section)

        if include_cot and cot_instruction:
            parts.append(
                f"CRITICAL INSTRUCTION\nYour analysis must meticulously follow these steps:\n{cot_instruction}"
            )
        # 4. Rules
        rules = self.agent_template.get("rules", [])
        if rules:
            parts.append(
                "<CRITICAL_RULES>\n"
                + "\n".join(f"- {rule}" for rule in rules)
                + "\n</CRITICAL_RULES>"
            )

        # 5. Examples
        examples = self.agent_template.get("examples", [])
        if examples:
            example_strings = []
            for i, example in enumerate(examples, 1):
                step_strings = [
                    f"<example number={i} description='{example.get('name', '')}'>",
                    f"<Example_Session description='{example.get('description', '')}' />\n",
                ]
                steps = example.get("steps", [])

                if steps and steps[0]["type"] == "custom":
                    step_strings.append(
                        f"<situation> {steps[0]['content']} </situation>"
                    )

                if steps:
                    step_strings.append("<loop_example>")

                for i, step in enumerate(steps):
                    step_strings.append("<step>")
                    if step["type"] == "thought":
                        step_strings.append(f"Thought: {step['content']}")
                    elif step["type"] == "tool_call":
                        tool_input = step.get("input")
                        if tool_input:
                            step_strings.append(f"Tool: {step['name']}: {tool_input}")
                        else:
                            step_strings.append(f"Tool: {step['name']}")
                        step_strings.append("PAUSE")
                        # Add the "You will be called again" part for all but the last step
                        # if i < len(steps) - 1:
                        #     step_strings.append("\nYou will be called again with this:")
                    elif step["type"] == "observation":
                        step_strings.append(f"Observation: {step['content']}")
                    elif step["type"] == "answer":
                        # The thought preceding the answer is the last thought in the sequence
                        if step_strings and "Thought:" in step_strings[-1]:
                            # No need to add another thought if the last step was one
                            pass
                        step_strings.append("You then output:")
                        step_strings.append(f"Answer: {step['content']}")
                    step_strings.append("</step>")
                step_strings.append("</loop_example>")
                example_strings.append("\n".join(step_strings) + "\n</example>")
            parts.append(
                "\n\n<TOOL_USE_EXAMPLES>\n"
                + "\n\n".join(example_strings)
                + "\n</TOOL_USE_EXAMPLES>"
            )

        return "\n\n".join(parts).strip()

    def _build_cot_prompt(self, **runtime_kwargs) -> str:
        # Implementation for Chain of Thought prompts would go here
        pass

    def _build_few_shot_prompt(self, **runtime_kwargs) -> str:
        # This is a generic builder for few-shot prompts. It assembles parts
        # provided at runtime, without any internal conditional logic.
        parts = []
        parts.append(self.agent_template.get("system_message", ""))

        rules = self.agent_template.get("rules", [])
        if rules:
            parts.append("CRITICAL RULES:\n" + "\n".join(f"- {rule}" for rule in rules))

        # The calling code is responsible for constructing and passing these parts
        if "response_format_rules" in runtime_kwargs:
            parts.append(runtime_kwargs["response_format_rules"])

        if "valid_actions_list" in runtime_kwargs:
            parts.append(runtime_kwargs["valid_actions_list"])

        if "example" in runtime_kwargs:
            parts.append(runtime_kwargs["example"])

        return "\n\n".join(parts)

    def _build_default_prompt(self, **runtime_kwargs) -> str:
        # A fallback for basic string substitution if no type matches
        raw_string = self.agent_template.get("system_message", "")
