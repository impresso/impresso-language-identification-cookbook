# GitHub Copilot Instructions for Impresso Language Identification

This repository contains the language identification (LID) cookbook for the Impresso project, which processes historical multilingual newspaper collections.

## Project Overview

This is a Python 3.11 project that uses a Make-based build system for language identification in historical newspapers. The project handles multilingual content (primarily German, French, English, Italian, and Luxembourgish) with OCR noise and historical spelling variations.

## Key Technologies and Tools

- **Python 3.11**: Primary programming language
- **Make**: Build automation and workflow orchestration
- **Pipenv**: Dependency management
- **AWS S3**: Data storage and synchronization via rclone
- **FastText**: Machine learning-based language identification
- **Multiple LID Systems**: langid, wp_ft, impresso_ft, lingua, impresso_langident_pipeline

## Code Style and Conventions

### Python
- Follow PEP 8 style guidelines
- Use flake8 for linting (configuration in `.flake8`)
- Type hints are encouraged (project uses pyright for type checking - see `pyrightconfig.json`)
- Use descriptive variable names appropriate for academic/research context

### File Organization
- `lib/`: Core Python scripts for language identification
  - `impresso_langident_systems.py`: Stage 1a - LID predictions
  - `newspaper_statistics.py`: Stage 1b - Collection statistics
  - `impresso_ensemble_lid.py`: Ensemble stage - Final decisions
- `configs/`: Configuration files for different processing runs
- `test/`: Test files
- `cookbook/`: Impresso Make-Based Cookbook (submodule)

## Important Patterns and Practices

### Make-Based Workflow
- The project uses a sophisticated Make-based build system with parallel processing
- All data processing should be integrated via Makefile targets, not standalone scripts
- Use stamp files for tracking build progress
- Support parallel processing with `COLLECTION_JOBS` and `NEWSPAPER_JOBS` parameters

### Data Handling
- All data is stored on S3 and synchronized locally
- Input: Newspaper text data in JSONL format (often bzip2 compressed)
- Output: Language identification JSON with confidence scores
- Handle missing, partial, or incorrect metadata gracefully

### Processing Pipeline
The system follows a three-stage approach:
1. **Stage 1a**: Apply multiple LID classifiers to generate predictions
2. **Stage 1b**: Aggregate statistics per newspaper collection
3. **Ensemble**: Make final language decisions using rule-based voting

### Error Handling
- Handle OCR noise and historical spelling variations
- Deal with mixed-language content appropriately
- Validate JSON output against Impresso schemas
- Log processing statistics and diagnostics

## When Making Changes

### Adding New Features
- Consider impact on all three processing stages
- Update both Python code and Makefile targets
- Document changes in README.md if user-facing
- Test with sample newspaper data

### Modifying LID Systems
- Changes to `impresso_langident_systems.py` affect Stage 1a
- Update ensemble voting logic in `impresso_ensemble_lid.py` if needed
- Maintain backward compatibility with existing output formats

### Performance Considerations
- Stage 1a is computationally expensive (runs LID models on all text)
- Stages 1b and ensemble are relatively fast (process existing predictions)
- Consider parallelization options for batch processing
- Optimize for I/O when working with S3 data

## Testing
- Run existing tests with `make test` or via tox
- Test with small subsets of data before full collection processing
- Validate JSON output format compliance
- Check statistics and diagnostics files for sanity

## Configuration
- Local configuration in `config.local.mk` (not committed)
- S3 credentials in `.env` file (not committed)
- Processing parameters configurable via Make variables
- Use sample files as templates: `config.local.mk.sample`, `dotenv.sample`

## Dependencies
- Add new dependencies via Pipfile, not requirements.txt
- Use `pipenv install <package>` to add dependencies
- Lock dependencies with `pipenv lock`
- System dependencies (rclone, jq, parallel) required for full functionality

## Common Pitfalls to Avoid
- Don't commit sensitive credentials (S3 keys, access tokens)
- Don't modify the cookbook submodule directly
- Don't assume metadata is always present or correct
- Don't ignore language identification confidence scores
- Don't break parallel processing capability
- Don't hardcode file paths (use Make variables)

## Helpful Context
- Historical newspapers have unique challenges: OCR errors, Gothic fonts, mixed languages
- Luxembourgish detection is particularly important for this project
- The ensemble approach compensates for individual classifier weaknesses
- Processing can be distributed across multiple machines via S3
