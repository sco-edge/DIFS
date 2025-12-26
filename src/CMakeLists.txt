project(infaas-src)

# ------------------------------------------------------------
# Build selection
# ------------------------------------------------------------
if(BUILD_ONLY_WORKER)
  message(STATUS "Building - worker")
  add_subdirectory(worker)

elseif(BUILD_ONLY_MD)
    message(STATUS "Metadata-store handled at root")

else()
  message(STATUS "Building - all")
  add_subdirectory(worker)
  add_subdirectory(master)
  add_subdirectory(cli-tools)
endif()
