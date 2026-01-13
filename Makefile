.PHONY: profile clean help test-compiler

# Default target
help:
	@echo "x-allocator Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make profile       - Generate profile.json and schedule.json"
	@echo "  make test-compiler - Run full compiler test (profile + compile + show results)"
	@echo "  make clean         - Remove generated files"
	@echo "  make help          - Show this help message"

# Generate profiling outputs
profile:
	@python scripts/generate_profile.py

# Full dun
run:
	@echo "========================================================================"
	@echo "X-Allocator End to End Run"
	@echo "========================================================================"
	@echo "Generating profile.json, cost.json, and schedule.json..."
	@echo "------------------------------------------------------------------------"
	@$(MAKE) profile
	@echo "Profiling complete!"
	@echo ""
	@echo "Running compiler..."
	@echo "------------------------------------------------------------------------"
	@python scripts/compile.py --schedule data/tmp/schedule.json --src src --output data/tmp/build --verbose
	@echo "Compilation complete!"
	@echo ""
	@echo "========================================================================"
	@echo "Run complete! Check data/tmp/build/ for optimized code."

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	@rm -f data/tmp/*.json
	@rm -rf data/tmp/build
	@echo "Clean complete"

