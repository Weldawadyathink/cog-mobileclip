SHELL := /bin/bash

TEMPLATES_DIR := template
OUTPUT_DIR := generated
MODEL_CONFIG := models.yaml


TEMPLATE_FILES := $(shell find $(TEMPLATES_DIR) -maxdepth 1 -type f)

MODEL_NAMES := $(shell command -v yq >/dev/null 2>&1 && [ -f $(MODEL_CONFIG) ] && yq '.[] | .name' $(MODEL_CONFIG) || echo "")

ALL_OUTPUT_FILES := $(foreach name,$(MODEL_NAMES), \
                      $(patsubst $(TEMPLATES_DIR)/%,$(OUTPUT_DIR)/$(name)/%,$(TEMPLATE_FILES)))


.PHONY: generate clean list


generate: $(ALL_OUTPUT_FILES)
	@echo "✅ All models generated successfully."


list:
	@echo "Discovered Models:"
	@$(foreach name,$(MODEL_NAMES),echo "  - $(name)";)
	@echo "\nDiscovered Templates:"
	@$(foreach tpl,$(TEMPLATE_FILES),echo "  - $(notdir $(tpl))";)
	@echo "\nWill Generate:"
	@$(foreach file,$(ALL_OUTPUT_FILES),echo "  - $(file)";)


clean:
	@echo "🔥 Removing $(OUTPUT_DIR)..."
	@rm -rf $(OUTPUT_DIR)


.SECONDEXPANSION:

$(ALL_OUTPUT_FILES): $(OUTPUT_DIR)/% : $$(TEMPLATES_DIR)/$$(notdir $$*) | $$(OUTPUT_DIR)/$$(dir $$*)
	$(eval MODEL_NAME := $(word 2,$(subst /, ,$@)))
	$(eval TEMPLATE_FILE_PATH := $(word 1,$^))

	@echo "🔨 Generating $@ from $(TEMPLATE_FILE_PATH)"

	$(eval YQ_QUERY := '.[] | select(.name == "$(MODEL_NAME)")')

	@export $$(yq '.[] | select(.name == "mobileclip-s1") | to_entries | .[] | .key + "=" + .value' models.yaml)
	@envsubst < "$(TEMPLATE_FILE_PATH)" > "$@"


$(OUTPUT_DIR)/%/:
	@mkdir -p $@
