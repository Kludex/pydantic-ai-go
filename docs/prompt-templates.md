# Prompt templates

Compile a prompt once. Render it against typed run dependencies for each request.

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

type Account struct {
	Name string `json:"name"`
	Plan string `json:"plan"`
}

type Deps struct {
	Account Account
}

func main() {
	instructions := ai.MustParsePromptTemplate[Deps](`You support this account:
{{xml .Account}}`)
	description := ai.MustParsePromptTemplate[Deps]("Support agent for {{.Account.Name}}")

	model := openai.NewModel("gpt-5")
	agent := ai.NewAgent[Deps, string](
		model,
		ai.WithAgentDescriptionFunc(description.Description),
	)
	agent.AddInstructionsFunc(instructions.Instructions)

	result, err := agent.Run(context.Background(), "Which plan am I using?", Deps{
		Account: Account{Name: "Ada", Plan: "business"},
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

`PromptTemplate` uses the standard library `text/template` syntax. The template data is the run's `Deps` value. `Instructions` has the `InstructionsFunc` signature. `Description` has the `AgentDescriptionFunc` signature.

`ParsePromptTemplate` returns syntax errors. `MustParsePromptTemplate` panics on invalid application-owned templates and is useful for package-level configuration. Rendering returns an error for a missing map key or a failing template function instead of inserting `<no value>`.

A parsed template is immutable and safe to render concurrently. Do not parse untrusted user input as a template. Go templates can invoke exported methods on their data.

## XML prompt data

The built-in `xml` template function calls `FormatAsXML` with its defaults. It formats nested prompt data as XML because models can read repeated and semi-structured values more reliably in that form.

`FormatAsXML` supports:

- Strings, booleans, integer and floating-point values.
- `[]byte`, `time.Time`, `time.Duration`, and `encoding.TextMarshaler` values.
- Maps with string or integer keys. Map output is sorted by the formatted key.
- Slices and arrays.
- Exported struct fields. A `json` tag changes the element name, and `json:"-"` excludes a field.
- Pointers, interfaces, and `nil` values.

The default output has no root wrapper, uses `item` for sequence values, writes `null` for nil values, and indents nested elements with two spaces. Configure it with `WithXMLRoot`, `WithXMLItemTag`, `WithXMLNull`, and `WithXMLIndent`. An empty indent emits compact XML.

`FormatAsXML` returns an error for unsupported values, unsupported map keys, invalid element names, text-marshaling failures, and cyclic data. It escapes text content before adding it to the prompt.
