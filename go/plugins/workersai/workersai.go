// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

package workersai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/genkit"
)

const provider = "workersai"

// WorkersAI provides configuration options for the Workers AI plugin.
type WorkersAI struct {
	APIToken  string // API token for Cloudflare Workers AI. If empty, the value of the environment variable CLOUDFLARE_API_TOKEN will be used.
	AccountID string // Cloudflare account ID. If empty, the value of the environment variable CLOUDFLARE_ACCOUNT_ID will be used.
	BaseURL   string // Base URL for the API. If empty, defaults to "https://api.cloudflare.com/client/v4".

	httpClient *http.Client
	mu         sync.Mutex
	initted    bool
}

// workersAIRequest represents the request structure for Workers AI API.
type workersAIRequest struct {
	Messages []workersAIMessage `json:"messages,omitempty"`
	Prompt   string             `json:"prompt,omitempty"`
	Stream   bool               `json:"stream,omitempty"`
}

// workersAIMessage represents a message in the Workers AI API format.
type workersAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// workersAIResponse represents the response structure from Workers AI API.
type workersAIResponse struct {
	Success bool   `json:"success"`
	Result  Result `json:"result"`
	Errors  []struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	} `json:"errors"`
	Messages []interface{} `json:"messages"`
}

// Result represents the result structure in the Workers AI response.
type Result struct {
	Response string `json:"response"`
}

// generator holds the configuration for generating responses.
type generator struct {
	model     string
	apiToken  string
	accountID string
	baseURL   string
	client    *http.Client
}

// Name returns the name of the plugin.
func (w *WorkersAI) Name() string {
	return provider
}

// Init initializes the Workers AI plugin.
func (w *WorkersAI) Init(ctx context.Context, g *genkit.Genkit) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.initted {
		return errors.New("Workers AI plugin already initialized")
	}

	if w == nil {
		w = &WorkersAI{}
	}

	// Set API token from environment if not provided
	apiToken := w.APIToken
	if apiToken == "" {
		apiToken = os.Getenv("CLOUDFLARE_API_TOKEN")
		if apiToken == "" {
			return errors.New("Workers AI requires setting CLOUDFLARE_API_TOKEN in the environment or providing APIToken in config")
		}
	}

	// Set account ID from environment if not provided
	accountID := w.AccountID
	if accountID == "" {
		accountID = os.Getenv("CLOUDFLARE_ACCOUNT_ID")
		if accountID == "" {
			return errors.New("Workers AI requires setting CLOUDFLARE_ACCOUNT_ID in the environment or providing AccountID in config")
		}
	}

	// Set base URL if not provided
	baseURL := w.BaseURL
	if baseURL == "" {
		baseURL = "https://api.cloudflare.com/client/v4"
	}

	// Create HTTP client if not provided
	if w.httpClient == nil {
		w.httpClient = &http.Client{
			Timeout: 30 * time.Second,
		}
	}

	w.APIToken = apiToken
	w.AccountID = accountID
	w.BaseURL = baseURL
	w.initted = true

	// Register known models
	models := getKnownModels()
	for modelName, modelInfo := range models {
		w.DefineModel(g, modelName, &modelInfo)
	}

	return nil
}

// DefineModel defines a Workers AI model with the given name and configuration.
func (w *WorkersAI) DefineModel(g *genkit.Genkit, name string, info *ai.ModelInfo) ai.Model {
	w.mu.Lock()
	defer w.mu.Unlock()

	if !w.initted {
		panic("Workers AI plugin not initialized")
	}

	var mi ai.ModelInfo
	if info != nil {
		mi = *info
	} else {
		mi = ai.ModelInfo{
			Label:    "Workers AI - " + name,
			Supports: &ai.ModelSupports{
				Multiturn:  true,
				SystemRole: true,
				Media:      false, // Most Workers AI models don't support media yet
				Tools:      false, // Tool calling support varies by model
			},
			Versions: []string{},
		}
	}

	gen := &generator{
		model:     name,
		apiToken:  w.APIToken,
		accountID: w.AccountID,
		baseURL:   w.BaseURL,
		client:    w.httpClient,
	}

	return genkit.DefineModel(g, provider, name, &mi, gen.generate)
}

// IsDefinedModel reports whether a model is defined.
func IsDefinedModel(g *genkit.Genkit, name string) bool {
	return genkit.LookupModel(g, provider, name) != nil
}

// Model returns the [ai.Model] with the given name.
func Model(g *genkit.Genkit, name string) ai.Model {
	return genkit.LookupModel(g, provider, name)
}

// ModelRef creates a new ModelRef for a Workers AI model.
func ModelRef(name string) ai.ModelRef {
	return ai.NewModelRef(provider+"/"+name, nil)
}

// generate performs the actual generation using the Workers AI API.
func (gen *generator) generate(ctx context.Context, input *ai.ModelRequest, cb func(context.Context, *ai.ModelResponseChunk) error) (*ai.ModelResponse, error) {
	// Build the request
	req := &workersAIRequest{
		Stream: cb != nil,
	}

	// Check if the model supports chat format or requires prompt format
	if supportsChatFormat(gen.model) {
		// Convert messages to Workers AI format
		var messages []workersAIMessage
		for _, msg := range input.Messages {
			content := concatenateTextParts(msg.Content)
			if content != "" {
				messages = append(messages, workersAIMessage{
					Role:    convertRole(msg.Role),
					Content: content,
				})
			}
		}
		req.Messages = messages
	} else {
		// Use prompt format for models that don't support chat
		req.Prompt = buildPrompt(input.Messages)
	}

	// Make the API request
	url := fmt.Sprintf("%s/accounts/%s/ai/run/@cf/%s", gen.baseURL, gen.accountID, gen.model)

	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+gen.apiToken)

	resp, err := gen.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to make request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
	}

	// Handle streaming response
	if cb != nil {
		return gen.handleStreamingResponse(ctx, resp, cb, input)
	}

	// Handle non-streaming response
	return gen.handleResponse(resp, input)
}

// handleResponse processes a non-streaming response from Workers AI.
func (gen *generator) handleResponse(resp *http.Response, input *ai.ModelRequest) (*ai.ModelResponse, error) {
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	var workerResp workersAIResponse
	if err := json.Unmarshal(body, &workerResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	if !workerResp.Success {
		errMsg := "API request failed"
		if len(workerResp.Errors) > 0 {
			errMsg = workerResp.Errors[0].Message
		}
		return nil, fmt.Errorf(errMsg)
	}

	response := &ai.ModelResponse{
		Request:      input,
		FinishReason: ai.FinishReason("stop"),
		Message: &ai.Message{
			Role:    ai.RoleModel,
			Content: []*ai.Part{ai.NewTextPart(workerResp.Result.Response)},
		},
		Usage: &ai.GenerationUsage{}, // Workers AI doesn't provide usage metrics in the response
	}

	return response, nil
}

// handleStreamingResponse processes a streaming response from Workers AI.
func (gen *generator) handleStreamingResponse(ctx context.Context, resp *http.Response, cb func(context.Context, *ai.ModelResponseChunk) error, input *ai.ModelRequest) (*ai.ModelResponse, error) {
	// Note: Workers AI streaming implementation would depend on their specific streaming format
	// For now, we'll fall back to non-streaming behavior
	return gen.handleResponse(resp, input)
}

// ListActions returns the list of available actions for this plugin.
func (w *WorkersAI) ListActions(ctx context.Context) []core.ActionDesc {
	var actions []core.ActionDesc

	models := getKnownModels()
	for modelName, modelInfo := range models {
		metadata := map[string]any{
			"model": map[string]any{
				"supports": map[string]any{
					"media":       modelInfo.Supports.Media,
					"multiturn":   modelInfo.Supports.Multiturn,
					"systemRole":  modelInfo.Supports.SystemRole,
					"tools":       modelInfo.Supports.Tools,
					"toolChoice":  false, // Workers AI doesn't support tool choice
					"constrained": false, // Workers AI doesn't support constrained generation
				},
				"versions": modelInfo.Versions,
				"stage":    string(modelInfo.Stage),
			},
		}
		metadata["label"] = modelInfo.Label

		actions = append(actions, core.ActionDesc{
			Type:     core.ActionTypeModel,
			Name:     fmt.Sprintf("%s/%s", provider, modelName),
			Key:      fmt.Sprintf("/%s/%s/%s", core.ActionTypeModel, provider, modelName),
			Metadata: metadata,
		})
	}

	return actions
}

// ResolveAction resolves an action by type and name.
func (w *WorkersAI) ResolveAction(g *genkit.Genkit, atype core.ActionType, name string) error {
	switch atype {
	case core.ActionTypeModel:
		models := getKnownModels()
		if modelInfo, exists := models[name]; exists {
			w.DefineModel(g, name, &modelInfo)
		} else {
			// Define with default info if model not in known list
			w.DefineModel(g, name, nil)
		}
	}
	return nil
}

// Helper functions

// convertRole converts Genkit roles to Workers AI roles.
func convertRole(role ai.Role) string {
	switch role {
	case ai.RoleUser:
		return "user"
	case ai.RoleModel:
		return "assistant"
	case ai.RoleSystem:
		return "system"
	default:
		return "user"
	}
}

// concatenateTextParts concatenates all text parts from a message.
func concatenateTextParts(parts []*ai.Part) string {
	var builder strings.Builder
	for _, part := range parts {
		if part.IsText() {
			builder.WriteString(part.Text)
		}
	}
	return builder.String()
}

// buildPrompt builds a simple prompt from messages for models that don't support chat format.
func buildPrompt(messages []*ai.Message) string {
	var builder strings.Builder
	for _, msg := range messages {
		content := concatenateTextParts(msg.Content)
		if content != "" {
			switch msg.Role {
			case ai.RoleSystem:
				builder.WriteString("System: " + content + "\n")
			case ai.RoleUser:
				builder.WriteString("User: " + content + "\n")
			case ai.RoleModel:
				builder.WriteString("Assistant: " + content + "\n")
			}
		}
	}
	return builder.String()
}

// supportsChatFormat returns true if the model supports chat format.
func supportsChatFormat(model string) bool {
	return strings.Contains(model, "llama") ||
		   strings.Contains(model, "mistral") ||
		   strings.Contains(model, "qwen") ||
		   strings.Contains(model, "gemma")
}
