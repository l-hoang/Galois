# Linking against the Boost libs via target_link_libraries
# can result in LLVM headers installed alongside Boost
# taking precedence over our vendored ones.
# avoid this by putting our vendored LLVM include paths
# first when compiling the executables that use them.
include_directories(BEFORE
  ${CMAKE_SOURCE_DIR}/libllvm/include
  ${CMAKE_CURRENT_BINARY_DIR}/../libllvm/include
)

add_library(lonestar STATIC src/BoilerPlate.cpp)
target_include_directories(lonestar PUBLIC 
  ${CMAKE_SOURCE_DIR}/libgalois/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(lonestar galois_shmem gllvm)

function(versionGen name file)
  app(${name}_offline ${file})
  target_compile_definitions(${name}_offline PRIVATE OPTVERSION=1)

  app(${name}_filegraph ${file})
  target_compile_definitions(${name}_filegraph PRIVATE OPTVERSION=2)

  app(${name}_direct ${file})
  target_compile_definitions(${name}_direct PRIVATE OPTVERSION=3)

  app(${name}_buffered ${file})
  target_compile_definitions(${name}_buffered PRIVATE OPTVERSION=4)

  app(${name}_mmap ${file})
  target_compile_definitions(${name}_mmap PRIVATE OPTVERSION=5)

  app(${name}_od ${file})
  target_compile_definitions(${name}_od PRIVATE OPTVERSION=6)

  app(${name}_async ${file})
  target_compile_definitions(${name}_async PRIVATE OPTVERSION=7)
endfunction()

add_subdirectory(betweennesscentrality) 
add_subdirectory(connectedcomponents)
add_subdirectory(kcore)
add_subdirectory(pagerank)
add_subdirectory(sssp)
#add_subdirectory(practice)
