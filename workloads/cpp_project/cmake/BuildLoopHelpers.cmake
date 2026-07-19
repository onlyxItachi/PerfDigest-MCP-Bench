# BuildLoopHelpers.cmake — the include()d helper the cmake_profile leg is
# meant to see show up as its own "script" unit (function bl_apply_target_defaults,
# distinct from the try_compile/find_package noise of compiler detection).

message(STATUS "BuildLoopHelpers.cmake loaded")

function(bl_apply_target_defaults tgt)
  if(NOT TARGET ${tgt})
    return()
  endif()
  target_compile_options(${tgt} PRIVATE -O2 -Wall -Wextra)
  set_target_properties(${tgt} PROPERTIES
    CXX_EXTENSIONS OFF
  )
endfunction()
