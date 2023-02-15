add_library(threadpool threadpool.h)
target_link_libraries(threadpool locker)
target_link_libraries(threadpool CGI)

target_include_directories(threadpool PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})