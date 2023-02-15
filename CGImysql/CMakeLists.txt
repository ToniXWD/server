add_library(CGI sql_connection_pool.cpp)
target_link_libraries(CGI locker)