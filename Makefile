##########################################################################################
# Makefile for impresso language identification
#
# Read the README.md for more information on how to use this Makefile.
# Or run `make` for online help.

#### ENABLE LOGGING FIRST
# USER-VARIABLE: LOGGING_LEVEL
# Defines the logging level for the Makefile.

# Load our make logging functions
include cookbook/log.mk


# USER-VARIABLE: CONFIG_LOCAL_MAKE
# Defines the name of the local configuration file to include.
#
# This file is used to override default settings and provide local configuration. If a
# file with this name exists in the current directory, it will be included. If the file
# does not exist, it will be silently ignored. Never add the file called config.local.mk
# to the repository! If you have stored config files in the repository set the
# CONFIG_LOCAL_MAKE variable to a different name.
CONFIG_LOCAL_MAKE ?= config.local.mk
ifdef CFG
  CONFIG_LOCAL_MAKE := $(CFG)
  $(info Overriding CONFIG_LOCAL_MAKE to $(CONFIG_LOCAL_MAKE) from CFG variable)
endif
# Load local config if it exists (ignore silently if it does not exists)
-include $(CONFIG_LOCAL_MAKE)


# Now we can use the logging function to show the current logging level
  $(call log.info, LOGGING_LEVEL)


#: Show help message
help::
	@echo "Usage: make <target>"
	@echo "Targets:"
	@echo "  setup                 # Prepare the local directories"
	@echo "  collection            # Call make all for each newspaper found in the file $(NEWSPAPERS_TO_PROCESS_FILE)"
	@echo "  all                   # Resync the data from the S3 bucket to the local directory and process all years of a single newspaper"
	@echo "  newspaper             # Process a single newspaper for all years"
	@echo "  sync                  # Sync the data from the S3 bucket to the local directory"
	@echo "  resync                # Remove the local synchronization file stamp and sync again."
	@echo "  clean-build           # Remove the entire build directory"
	@echo "  clean-newspaper       # Remove the local directory for a single newspaper"
	@echo "  help                  # Show this help message"

# Default target when no target is specified on the command line
.DEFAULT_GOAL := help
.PHONY: help


# SETTINGS FOR THE MAKE PROGRAM
include cookbook/make_settings.mk



# SETUP SETTINGS AND TARGETS
include cookbook/setup.mk
include cookbook/setup_python.mk
#include cookbook/setup_langident.mk

# Load newspaper list configuration and processing rules
include cookbook/newspaper_list.mk


# SETUP PATHS
include cookbook/paths_canonical.mk
include cookbook/paths_rebuilt.mk
include cookbook/paths_langident.mk






# SYNCHRONIZATION TARGETS
include cookbook/sync.mk
include cookbook/sync_canonical.mk
include cookbook/sync_rebuilt.mk
include cookbook/sync_langident.mk


include cookbook/clean.mk


# PROCESSING TARGETS
include cookbook/processing.mk
include cookbook/processing_langident.mk

# MAIN TARGETS
include cookbook/main_targets.mk

# FUNCTION
include cookbook/local_to_s3.mk


# FURTHER ADDONS
include cookbook/aggregators_langident.mk

# configure for aws client access
include cookbook/aws.mk

# Add help target with parallelization documentation
help::
	@echo ""
	@echo "PARALLELIZATION CONFIGURATION:"
	@echo "  COLLECTION_JOBS   #  Number of newspapers to process in parallel (default: 2)"
	@echo "                    #  Higher values increase parallelism but consume more memory"
	@echo "                    #  Recommended: 2-8 depending on available RAM and CPU cores"
	@echo ""
	@echo "  NEWSPAPER_JOBS    #  Number of parallel jobs per newspaper (default: NPROC/COLLECTION_JOBS)"
	@echo "                    #  Controls fine-grained parallelism within each newspaper"
	@echo "                    #  Auto-calculated to balance with COLLECTION_JOBS"
	@echo ""
	@echo "  MAX_LOAD          #  Maximum system load average (default: NPROC)"
	@echo "                    #  Prevents system overload by limiting concurrent processes"
	@echo "                    #  Set lower if system becomes unresponsive"
	@echo ""
	@echo "  NPROC             #  Number of CPU cores (auto-detected)"
	@echo "                    #  Override if auto-detection fails or for resource limiting"
	@echo ""
	@echo "PERFORMANCE TUNING:"
	@echo "  • For CPU-bound tasks: COLLECTION_JOBS ≤ NPROC"
	@echo "  • For I/O-bound tasks: COLLECTION_JOBS can exceed NPROC"
	@echo "  • High memory usage: Reduce COLLECTION_JOBS"
	@echo "  • System lag: Reduce MAX_LOAD to 70-80% of NPROC"
	@echo ""
	@echo "EXAMPLES:"
	@echo "  make collection COLLECTION_JOBS=4          # Process 4 newspapers in parallel"
	@echo "  make collection COLLECTION_JOBS=1 NEWSPAPER_JOBS=8  # Single newspaper, max internal parallelism"
	@echo "  make all MAX_LOAD=8                        # Limit system load to 8"
	@echo "  make newspaper NEWSPAPER_JOBS=4            # Use 4 jobs for single newspaper"
	@echo "  make collection NPROC=16 COLLECTION_JOBS=8 # Override CPU count and use 8 parallel newspapers"
	@echo ""
	@echo "MONITORING:"
	@echo "  tail -f build/collection.joblog            # Monitor collection progress"
	@echo "  htop -u ${USER}                            # Monitor system resources"
	@echo "  iostat -x 5                                # Monitor I/O performance"
	@echo ""
