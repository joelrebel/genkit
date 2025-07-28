package workersai

import (
	"context"
	"log"
	"math"
	"os"
	"strings"
	"testing"

	client "github.com/ashishdatta/workers-ai-golang/workers-ai"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func requireEnv(key string) (string, bool) {
	value, ok := os.LookupEnv(key)
	if !ok || value == "" {
		return "", false
	}

	return value, true
}

func TestWorkersAILive(t *testing.T) {
	ctx := context.Background()

	g, err := genkit.Init(ctx,
		genkit.WithPlugins(&WorkersAI{}),
		// TODO: doesn't work with the mistralai model
		//genkit.WithDefaultModel("workersai/@cf/mistralai/mistral-small-3.1-24b-instruct"),
		genkit.WithDefaultModel("workersai/@cf/meta/llama-4-scout-17b-16e-instruct"),
	)
	if err != nil {
		log.Fatal(err)
	}

	t.Run("generate", func(t *testing.T) {
		resp, err := genkit.Generate(ctx, g,
			ai.WithPrompt("Which country was Napoleon the emperor of? Name the country, nothing else"),
		)
		if err != nil {
			t.Fatal(err)
		}

		out := strings.ReplaceAll(resp.Message.Content[0].Text, "\n", "")
		const want = "France"
		if out != want {
			t.Errorf("got %q, expecting %q", out, want)
		}
		if resp.Request == nil {
			t.Error("Request field not set properly")
		}
	})

	// TODO: figure out why this isn't functional

	gablorkenTool := genkit.DefineTool(g, "gablorken", "use this tool when the user asks to calculate a gablorken, carefuly inspect the user input to determine which value from the prompt corresponds to the input structure",
		func(ctx *ai.ToolContext, input struct {
			Value int
			Over  float64
		},
		) (float64, error) {
			return math.Pow(float64(input.Value), input.Over), nil
		},
	)

	t.Run("tool", func(t *testing.T) {

		//tools := genkit.ListTools(g)
		resp, err := genkit.Generate(ctx, g,
			ai.WithPrompt("what is a gablorken of 2 over 3.5? use the gablorken tool"),
			ai.WithTools(gablorkenTool),
			ai.WithMaxTurns(1),
			//ai.WithReturnToolRequests(true),
		)
		if err != nil {
			t.Fatal(err)
		}

		out := resp.Message.Content[0].Text
		const want = "11.31"
		if !strings.Contains(out, want) {
			t.Errorf("got %q, expecting it to contain %q", out, want)
		}
	})

}

func TestToGenkitToolRequestParts(t *testing.T) {
	// Define test cases in a table for clarity and maintainability.
	testCases := []struct {
		name          string
		inputCalls    []client.ToolCall
		expectedParts []*ai.Part
		expectError   bool
		errorContains string
	}{
		{
			name: "should correctly parse simple argument format",
			inputCalls: []client.ToolCall{
				{
					ID:   "call_simple_123",
					Type: "function",
					Function: client.FunctionToCall{
						Name:      "get_weather",
						Arguments: `{"location": "Eindhoven, NL", "unit": "celsius"}`,
					},
				},
			},
			expectedParts: []*ai.Part{
				ai.NewToolRequestPart(&ai.ToolRequest{
					Name: "get_weather",
					Input: map[string]any{
						"location": "Eindhoven, NL",
						"unit":     "celsius",
					},
				}),
			},
			expectError: false,
		},
		{
			name: "should correctly parse and simplify verbose argument format",
			inputCalls: []client.ToolCall{
				{
					ID:   "call_verbose_456",
					Type: "function",
					Function: client.FunctionToCall{
						Name:      "gablorken",
						Arguments: `{"Over": {"type": "number", "value": 3.5}, "Value": {"type": "integer", "value": 2}}`,
					},
				},
			},
			expectedParts: []*ai.Part{
				ai.NewToolRequestPart(&ai.ToolRequest{
					Name: "gablorken",
					Input: map[string]any{
						"Over":  3.5,
						"Value": float64(2), // JSON numbers are unmarshaled as float64.
					},
				}),
			},
			expectError: false,
		},
		{
			name: "should handle a mix of simple and verbose formats in a single call",
			inputCalls: []client.ToolCall{
				{
					Function: client.FunctionToCall{
						Name:      "get_weather_mixed",
						Arguments: `{"location": "Eindhoven, NL", "unit": {"type": "string", "value": "celsius"}}`,
					},
				},
			},
			expectedParts: []*ai.Part{
				ai.NewToolRequestPart(&ai.ToolRequest{
					Name: "get_weather_mixed",
					Input: map[string]any{
						"location": "Eindhoven, NL",
						"unit":     "celsius",
					},
				}),
			},
			expectError: false,
		},
		{
			name: "should return an error for malformed arguments JSON",
			inputCalls: []client.ToolCall{
				{
					Function: client.FunctionToCall{
						Name:      "get_weather",
						Arguments: `{"location": "Eindhoven, NL",`, // Missing closing brace
					},
				},
			},
			expectedParts: nil,
			expectError:   true,
			errorContains: "failed to unmarshal tool arguments",
		},
		{
			name:          "should return an empty slice for no tool calls",
			inputCalls:    []client.ToolCall{},
			expectedParts: []*ai.Part{},
			expectError:   false,
		},
	}

	// Iterate over the test cases and run them as sub-tests.
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Act: Call the function under test.
			parts, err := toGenkitToolRequestParts(tc.inputCalls)

			// Assert: Check the results.
			if tc.expectError {
				require.Error(t, err)
				if tc.errorContains != "" {
					assert.Contains(t, err.Error(), tc.errorContains)
				}
			} else {
				require.NoError(t, err)
				assert.Equal(t, len(tc.expectedParts), len(parts))
				if len(tc.expectedParts) > 0 {
					assert.Equal(t, tc.expectedParts[0].ToolRequest, parts[0].ToolRequest)
				}
			}
		})
	}
}

// TestToClientMessages uses a table-driven approach to validate the conversion
// of Genkit messages to the format expected by the Workers AI client library.
func TestToClientMessages(t *testing.T) {
	// Define the structure for our test cases
	testCases := []struct {
		name      string
		input     []*ai.Message
		expected  []interface{}
		expectErr bool
	}{
		{
			name: "Simple user message",
			input: []*ai.Message{
				ai.NewUserMessage(ai.NewTextPart("Hello, world!")),
			},
			expected: []interface{}{
				client.ChatMessage{Role: "user", Content: "Hello, world!"},
			},
			expectErr: false,
		},
		{
			name: "User and system messages",
			input: []*ai.Message{
				ai.NewSystemMessage(ai.NewTextPart("You are a helpful assistant.")),
				ai.NewUserMessage(ai.NewTextPart("What is Go?")),
			},
			expected: []interface{}{
				client.ChatMessage{Role: "system", Content: "You are a helpful assistant."},
				client.ChatMessage{Role: "user", Content: "What is Go?"},
			},
			expectErr: false,
		},
		{
			name: "Simple model (assistant) response",
			input: []*ai.Message{
				ai.NewModelMessage(ai.NewTextPart("Go is a programming language.")),
			},
			expected: []interface{}{
				client.ChatMessage{Role: "assistant", Content: "Go is a programming language."},
			},
			expectErr: false,
		},
		{
			name: "Full tool call and response sequence",
			input: []*ai.Message{
				// 1. User asks to use a tool
				ai.NewUserMessage(ai.NewTextPart("what is a gablorken of 2 over 3.5?")),
				// 2. Model responds with a request to call the tool
				{
					Role: ai.RoleModel,
					Content: []*ai.Part{
						ai.NewToolRequestPart(&ai.ToolRequest{
							Name:  "gablorken",
							Input: map[string]any{"Value": 2, "Over": 3.5},
							Ref:   "tool-call-id-123", // The crucial ID
						}),
					},
				},
				// 3. The tool is executed and the result is sent back
				{
					Role: ai.RoleTool,
					Content: []*ai.Part{
						ai.NewToolResponsePart(&ai.ToolResponse{
							Name:   "gablorken",
							Output: map[string]any{"result": 11.31},
							Ref:    "tool-call-id-123", // The same ID is passed back
						}),
					},
				},
			},
			expected: []interface{}{
				// User message
				client.ChatMessage{Role: "user", Content: "what is a gablorken of 2 over 3.5?"},
				// Assistant message with the tool call request
				client.ResponseMessage{
					Role: "assistant",
					ToolCalls: []client.ToolCall{
						{
							ID:   "tool-call-id-123",
							Type: "function",
							Function: client.FunctionToCall{
								Name:      "gablorken",
								Arguments: `{"Over":3.5,"Value":2}`,
							},
						},
					},
				},
				// Tool result message
				client.ToolMessage{
					Role:       "tool",
					Content:    `{"result":11.31}`,
					ToolCallID: "tool-call-id-123",
				},
			},
			expectErr: false,
		},
		{
			name: "Error on unmarshalable tool response output",
			input: []*ai.Message{
				{
					Role: ai.RoleTool,
					Content: []*ai.Part{
						ai.NewToolResponsePart(&ai.ToolResponse{
							Name:   "badTool",
							Ref:    "bad-id-1",
							Output: make(chan int), // This type cannot be marshaled to JSON
						}),
					},
				},
			},
			expected:  nil,
			expectErr: true,
		},
	}

	// Run the test cases
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Use require for assertions, which stops the test on failure.
			r := require.New(t)

			got, err := toClientMessages(tc.input)

			if tc.expectErr {
				r.Error(err)
				return // Test ends here if an error is expected.
			}

			r.NoError(err)
			r.NotNil(got)
			r.Len(got, len(tc.expected))

			// Compare each message, handling the special case for tool calls.
			for i := range tc.expected {
				expectedMsg := tc.expected[i]
				gotMsg := got[i]

				// The ResponseMessage contains tool calls with JSON strings that need special handling.
				if expectedResp, ok := expectedMsg.(client.ResponseMessage); ok {
					gotResp, ok := gotMsg.(client.ResponseMessage)
					r.True(ok, "expected message type client.ResponseMessage, but got %T", gotMsg)

					r.Equal(expectedResp.Role, gotResp.Role)
					r.Len(gotResp.ToolCalls, len(expectedResp.ToolCalls))

					for j, expectedCall := range expectedResp.ToolCalls {
						gotCall := gotResp.ToolCalls[j]
						// Compare basic fields
						r.Equal(expectedCall.ID, gotCall.ID)
						r.Equal(expectedCall.Type, gotCall.Type)
						r.Equal(expectedCall.Function.Name, gotCall.Function.Name)
						// Use JSONEq to compare arguments, ignoring key order and whitespace.
						r.JSONEq(expectedCall.Function.Arguments, gotCall.Function.Arguments)
					}
				} else {
					// For all other message types, a direct comparison is sufficient.
					r.Equal(expectedMsg, gotMsg)
				}
			}
		})
	}
}
