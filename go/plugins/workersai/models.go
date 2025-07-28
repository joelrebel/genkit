package workersai

import "github.com/firebase/genkit/go/ai"

const (
	mistralSmall3124BInstruct     = "@cf/mistralai/mistral-small-3.1-24b-instruct"
	metaLlama3370bInstructFp8Fast = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
	metaLlama4scout17b16einstruct = "@cf/meta/llama-4-scout-17b-16e-instruct"
)

var (
	workersAIModels = []string{
		mistralSmall3124BInstruct,
		metaLlama3370bInstructFp8Fast,
		metaLlama4scout17b16einstruct,
	}

	supportedWorkersAIModels = map[string]ai.ModelInfo{
		mistralSmall3124BInstruct: {
			Label: mistralSmall3124BInstruct,
			Supports: &ai.ModelSupports{
				Multiturn:  true,
				Tools:      true,
				ToolChoice: false,
				SystemRole: true,
				Media:      true,
			},
		},

		metaLlama3370bInstructFp8Fast: {
			Label: metaLlama3370bInstructFp8Fast,
			Supports: &ai.ModelSupports{
				Multiturn:  true,
				Tools:      true,
				ToolChoice: false,
				SystemRole: true,
				Media:      true,
			},
		},
		metaLlama4scout17b16einstruct: {
			Label: metaLlama4scout17b16einstruct,
			Supports: &ai.ModelSupports{
				Multiturn:  true,
				Tools:      true,
				ToolChoice: false,
				SystemRole: true,
				Media:      true,
			},
		},
	}
)
