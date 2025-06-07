SHELL := /bin/bash

TEMPLATES_DIR := template
OUTPUT_DIR := generated
MODEL_CONFIG := models.yaml


TEMPLATE_FILES := $(shell find $(TEMPLATES_DIR) -maxdepth 1 -type f)

MODEL_NAMES := $(shell command -v yq >/dev/null 2>&1 && [ -f $(MODEL_CONFIG) ] && yq '.[] | .name' $(MODEL_CONFIG) || echo "")

ALL_OUTPUT_FILES := $(foreach name,$(MODEL_NAMES), \
                      $(patsubst $(TEMPLATES_DIR)/%,$(OUTPUT_DIR)/$(name)/%,$(TEMPLATE_FILES)))


.PHONY: generate clean list build push


build: generate
	@for name in $(MODEL_NAMES); do \
	  echo "🔨 Cog build and push for $$name"; \
	  cd $(OUTPUT_DIR)/$$name && cog build && cd ../..; \
	done
	@echo "✅ All models built successfully."


push: generate
	@for name in $(MODEL_NAMES); do \
	  echo "🔨 Cog build and push for $$name"; \
	  cd $(OUTPUT_DIR)/$$name && cog push && cd ../..; \
	done
	@echo "✅ All models built successfully."


generate: $(ALL_OUTPUT_FILES)
	@for name in $(MODEL_NAMES); do \
		cp -r common/* $(OUTPUT_DIR)/$$name; \
	done
	@echo "✅ All models generated successfully."


list:
	@echo "Discovered Models:"
	@$(foreach name,$(MODEL_NAMES),echo "  - $(name)";)
	@echo "Discovered Templates:"
	@$(foreach tpl,$(TEMPLATE_FILES),echo "  - $(notdir $(tpl))";)
	@echo "Will Generate:"
	@$(foreach file,$(ALL_OUTPUT_FILES),echo "  - $(file)";)


clean:
	@echo "🔥 Removing $(OUTPUT_DIR)..."
	@rm -rf $(OUTPUT_DIR)


.SECONDEXPANSION:

$(ALL_OUTPUT_FILES): $(OUTPUT_DIR)/% : $$(TEMPLATES_DIR)/$$(notdir $$*) | $$(OUTPUT_DIR)/$$(dir $$*)
	$(eval MODEL_NAME := $(word 2,$(subst /, ,$@)))
	$(eval TEMPLATE_FILE_PATH := $(word 1,$^))

	@echo "🔨 Generating $@ from $(TEMPLATE_FILE_PATH)"

	@export $$(yq '.[] | select(.name == "$(MODEL_NAME)") | to_entries | .[] | .key + "=" + .value' models.yaml); \
	envsubst < "$(TEMPLATE_FILE_PATH)" > "$@"


$(OUTPUT_DIR)/%/:
	@mkdir -p $@
