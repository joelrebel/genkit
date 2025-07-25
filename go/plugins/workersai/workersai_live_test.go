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
		// genkit.WithDefaultModel("workersai/@cf/mistralai/mistral-small-3.1-24b-instruct"),
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

func TestToClientMessages(t *testing.T) {
	t.Run("should correctly construct history for a full tool-calling cycle", func(t *testing.T) {
		// Arrange: A full conversation history for a tool call.
		toolCallID := "call_123"
		toolName := "get_weather"

		genkitHistory := []*ai.Message{
			// 1. User asks a question.
			{Role: ai.RoleUser, Content: []*ai.Part{ai.NewTextPart("What's the weather in Eindhoven?")}},
			// 2. Model requests a tool call. The ID is stored in the Input map.
			{
				Role: ai.RoleModel,
				Content: []*ai.Part{
					ai.NewToolRequestPart(&ai.ToolRequest{
						Name: toolName,
						Input: map[string]any{
							"__genkit_tool_call_id": toolCallID, // Our special key for the ID.
							"location":              "Eindhoven",
						},
					}),
				},
			},
			// 3. We provide the tool's response.
			{
				Role: ai.RoleTool,
				Content: []*ai.Part{
					ai.NewToolResponsePart(&ai.ToolResponse{
						Name:   toolName, // The simple name.
						Output: map[string]any{"temp": "15C"},
					}),
				},
			},
		}

		// Act: Call the function to convert this history for the API.
		clientMsgs, err := toClientMessages(genkitHistory)
		require.NoError(t, err)
		require.Len(t, clientMsgs, 3)

		// Assert: Check each message in the constructed history.

		// 1. User message should be correct.
		userMsg, ok := clientMsgs[0].(client.ChatMessage)
		require.True(t, ok)
		assert.Equal(t, "user", userMsg.Role)
		assert.Equal(t, "What's the weather in Eindhoven?", userMsg.Content)

		// 2. Assistant message should be reconstructed with the tool call.
		assistantMsg, ok := clientMsgs[1].(client.ChatMessage)
		require.True(t, ok)
		assert.Equal(t, "assistant", assistantMsg.Role)
		require.Len(t, assistantMsg.ToolCalls, 1)
		assert.Equal(t, toolCallID, assistantMsg.ToolCalls[0].ID)
		assert.Equal(t, toolName, assistantMsg.ToolCalls[0].Function.Name)
		// The special ID key should have been removed from the arguments.
		assert.JSONEq(t, `{"location": "Eindhoven"}`, assistantMsg.ToolCalls[0].Function.Arguments)

		// 3. Tool message should be reconstructed with the correct ID and content.
		toolMsg, ok := clientMsgs[2].(client.ToolMessage)
		require.True(t, ok)
		assert.Equal(t, "tool", toolMsg.Role)
		assert.Equal(t, toolCallID, toolMsg.ToolCallID)
		assert.JSONEq(t, `{"temp":"15C"}`, toolMsg.Content)
	})

	t.Run("should handle simple text messages", func(t *testing.T) {
		// Arrange
		genkitHistory := []*ai.Message{
			{Role: ai.RoleUser, Content: []*ai.Part{ai.NewTextPart("Hello")}},
			{Role: ai.RoleModel, Content: []*ai.Part{ai.NewTextPart("Hi there!")}},
		}

		// Act
		clientMsgs, err := toClientMessages(genkitHistory)
		require.NoError(t, err)
		require.Len(t, clientMsgs, 2)

		// Assert
		userMsg, ok := clientMsgs[0].(client.ChatMessage)
		require.True(t, ok)
		assert.Equal(t, "user", userMsg.Role)
		assert.Equal(t, "Hello", userMsg.Content)

		assistantMsg, ok := clientMsgs[1].(client.ChatMessage)
		require.True(t, ok)
		assert.Equal(t, "assistant", assistantMsg.Role)
		assert.Equal(t, "Hi there!", assistantMsg.Content)
	})

	t.Run("should return error if tool response appears without a preceding request", func(t *testing.T) {
		// Arrange: A tool response with no prior model request in the history.
		genkitHistory := []*ai.Message{
			{Role: ai.RoleUser, Content: []*ai.Part{ai.NewTextPart("Hello")}},
			{
				Role: ai.RoleTool,
				Content: []*ai.Part{
					ai.NewToolResponsePart(&ai.ToolResponse{
						Name:   "get_weather",
						Output: "15C",
					}),
				},
			},
		}

		// Act & Assert
		_, err := toClientMessages(genkitHistory)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "could not find tool_call_id for tool response")
	})
}
