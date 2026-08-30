// Package ai is an idiomatic Go library for the LLM agent loop.
//
// You create an Agent with a Model, register tools on it, and call Run.
// The agent loops: it sends the conversation to the model, executes any
// tool calls in the response, and repeats until the model produces a
// final output, which is unmarshalled into your Output type.
package ai
