.PHONY: profile clean help test-compiler

# Default target
help:
	@echo "x-allocator Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make profile       - Generate profile.json, cost.json, and schedule.json"
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
	@echo "Step 1: Generating profile.json, cost.json, and schedule.json..."
	@echo "------------------------------------------------------------------------"
	@$(MAKE) profile
	@echo "Profiling complete!"
	@echo ""
	@echo "Step 2: Running compiler..."
	@echo "------------------------------------------------------------------------"
	@python scripts/compile.py --schedule data/tmp/schedule.json --src src --output data/tmp/build --verbose
	@echo "Compilation complete!"
	@echo ""
	@echo "Step 3: Results Summary"
	@echo "------------------------------------------------------------------------"
	@echo "Generated files:"
	@ls -la data/tmp/*.json 2>/dev/null || echo "No JSON files found"
	@echo ""
	@echo "Compiled output:"
	@ls -la data/tmp/build/ 2>/dev/null || echo "No build directory found"
	@echo ""
	@echo "Step 4: Code Changes"
	@echo "------------------------------------------------------------------------"
	@if [ -d "data/tmp/build" ]; then \
		echo "Checking for @noncontig markers in original vs compiled code..."; \
		echo ""; \
		echo "Original model.py @noncontig lines:"; \
		grep -n "@noncontig" src/model.py || echo "No @noncontig markers found"; \
		echo ""; \
		echo "Compiled model.py (should have .contiguous() calls added):"; \
		if [ -f "data/tmp/build/model.py" ]; then \
			echo "Lines around @noncontig markers:"; \
			grep -n -A2 -B2 "@noncontig\|\.contiguous()" data/tmp/build/model.py || echo "No changes found"; \
		else \
			echo "model.py not found in build directory"; \
		fi; \
	else \
		echo "Build directory not created"; \
	fi
	@echo ""
	@echo "========================================================================"
	@echo "Test complete! Check data/tmp/build/ for optimized code."
	@echo "========================================================================"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	@rm -f data/tmp/*.json
	@rm -rf data/tmp/build
	@echo "Clean complete"

