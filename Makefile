.PHONY: profile clean help

# Default target
help:
	@echo "x-allocator Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make profile    - Generate profile.json, cost.json, and schedule.json"
	@echo "  make clean      - Remove generated files"
	@echo "  make help       - Show this help message"

# Generate profiling outputs
profile:
	@python scripts/generate_profile.py

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	@rm -f data/tmp/*.json
	@echo "Clean complete"

