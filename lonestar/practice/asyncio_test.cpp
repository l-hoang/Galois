#include <stdio.h>
#include <aio.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <signal.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#define NO_THREADS 5

int fd;

void *read_char(void *char_ptr)
{
    int char_no = (long) char_ptr;

    struct aiocb cb;
    bzero(&cb, sizeof(cb));

    size_t BUF_SIZE = sizeof(char);
    cb.aio_fildes = fd;
    cb.aio_buf = malloc(BUF_SIZE);
    cb.aio_nbytes = BUF_SIZE;
    cb.aio_offset = char_no*sizeof(char);

    int ret = aio_read(&cb);

    while (aio_error(&cb) == EINPROGRESS) ;
    //cb.aio_sigevent.sigev_value.
    if ((ret = aio_return(&cb)) > 0) {
        printf("Hello world, Target char number: %d\n", char_no);
        printf("Read char: %c\n", *(char *)cb.aio_buf);
    } else {
        fprintf(stderr, "Read fail\n");
    }
}

int main(void)
{
    pthread_t pth[NO_THREADS];

    fd = open("input", O_RDONLY);

    for (int i = 0; i < NO_THREADS; i++)
    {
        pthread_create(&pth[i], NULL, read_char, (void *)i);
    }

    pthread_exit(NULL);
    return 0;
}
