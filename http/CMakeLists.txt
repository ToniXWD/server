add_library(http http_conn.cpp)

target_link_libraries(http CGI)
target_link_libraries(http locker)
target_link_libraries(http mysqlclient)