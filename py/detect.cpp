#include <unistd.h>
#include <string.h>
#include <sys/wait.h>

void detect(char *img = nullptr) {
    char *myaargs[4];
    myaargs[0] = strdup("/home/toni/miniconda3/envs/torch/bin/python");
    myaargs[1] = strdup("/home/toni/TinyWebServer-raw_version/py/use_mmdet.py");
    if (img == nullptr) {
        myaargs[2] = strdup("/home/toni/TinyWebServer-raw_version/py/input.jpg");
    } else {
        myaargs[2] = strdup(img);
    }
    myaargs[3] = NULL;

    int rc = fork();

    if (rc == 0) {
        execvp(myaargs[0], myaargs);
    } else {
        int wc = wait(NULL);
    }
}