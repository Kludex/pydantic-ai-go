// Package ai provides typed LLM agents and direct model requests.
//
// You create an Agent with a Model, register tools on it, and call Run.
// The agent loops: it sends the conversation to the model, executes any
// tool calls in the response, and repeats until the model produces a
// final output, which is unmarshalled into your Output type.
//
// RequestModel and StreamModel expose the lower-level provider message API
// without tool execution or output validation.
package ai
