# %%

import { OllamaFunctions } from "@langchain/community/experimental/chat_models/ollama_functions";

const model = new OllamaFunctions({
  temperature: 0.1,
  model: "mistral",
});

import { OllamaFunctions } from "@langchain/community/experimental/chat_models/ollama_functions";
import { HumanMessage } from "@langchain/core/messages";

const model = new OllamaFunctions({
  temperature: 0.1,
  model: "mistral",
}).bind({
  functions: [
    {
      name: "get_current_weather",
      description: "Get the current weather in a given location",
      parameters: {
        type: "object",
        properties: {
          location: {
            type: "string",
            description: "The city and state, e.g. San Francisco, CA",
          },
          unit: { type: "string", enum: ["celsius", "fahrenheit"] },
        },
        required: ["location"],
      },
    },
  ],
  // You can set the `function_call` arg to force the model to use a function
  function_call: {
    name: "get_current_weather",
  },
});

const response = await model.invoke([
  new HumanMessage({
    content: "What's the weather in Boston?",
  }),
]);

console.log(response);

import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";
import { OllamaFunctions } from "@langchain/community/experimental/chat_models/ollama_functions";
import { PromptTemplate } from "@langchain/core/prompts";
import { JsonOutputFunctionsParser } from "@langchain/core/output_parsers/openai_functions";

const EXTRACTION_TEMPLATE = `Extract and save the relevant entities mentioned in the following passage together with their p
Passage:
{input}
`;

const prompt = PromptTemplate.fromTemplate(EXTRACTION_TEMPLATE);

// Use Zod for easier schema declaration
const schema = z.object({
  people: z.array(
    z.object({
      name: z.string().describe("The name of a person"),
      height: z.number().describe("The person's height"),
      hairColor: z.optional(z.string()).describe("The person's hair color"),
    })
  ),
});

const model = new OllamaFunctions({
  temperature: 0.1,
  model: "mistral",
}).bind({
  functions: [
    {
      name: "information_extraction",
      description: "Extracts the relevant information from the passage.",
      parameters: {
        type: "object",
        properties: zodToJsonSchema(schema),
      },
    },
  ],
  function_call: {
    name: "information_extraction",
  },
});

// Use a JsonOutputFunctionsParser to get the parsed JSON response directly.
const chain = await prompt.pipe(model).pipe(new JsonOutputFunctionsParser());

const response = await chain.invoke({
  input:
    "Alex is 5 feet tall. Claudia is 1 foot taller than Alex and jumps higher than him. Claudia has orange hair and Alex is blonde.",
});

console.log(response);

import { OllamaFunctions } from "@langchain/community/experimental/chat_models/ollama_functions";
import { HumanMessage } from "@langchain/core/messages";

// Custom system prompt to format tools. You must encourage the model
// to wrap output in a JSON object with "tool" and "tool_input" properties.
const toolSystemPromptTemplate = `You have access to the following tools:
{tools}
To use a tool, respond with a JSON object with the following structure:
{{
"tool": <name of the called tool>,
"tool_input": <parameters for the tool matching the above JSON schema>
}}`;

const model = new OllamaFunctions({
  temperature: 0.1,
  model: "mistral",
  toolSystemPromptTemplate,
}).bind({
  functions: [
    {
      name: "get_current_weather",
      description: "Get the current weather in a given location",
      parameters: {
        type: "object",
        properties: {
          location: {
            type: "string",
            description: "The city and state, e.g. San Francisco, CA",
          },
          unit: { type: "string", enum: ["celsius", "fahrenheit"] },
        },
        required: ["location"],
      },
    },
  ],
  // You can set the `function_call` arg to force the model to use a function
  function_call: {
    name: "get_current_weather",
  },
});

const response = await model.invoke([
  new HumanMessage({
    content: "What's the weather in Boston?",
  }),
]);

console.log(response);