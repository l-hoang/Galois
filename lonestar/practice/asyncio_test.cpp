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
#define SIG_AIO SIGRTMIN+1

int fd;

//void read_handler(sigval_t sigval)
void read_handler(int s, siginfo_t * info, void *ctx)
{
    printf("AIO handler is called\n");

    struct aiocb *req;

    req = (struct aiocb *)info->si_value.sival_ptr;

    if (aio_error (req) == 0) {
        int ret = aio_return(req);
        printf("results: %c\n", *(char *) req->aio_buf);
    }
    else {
        printf("Fail to get return\n");
    }
}


void *read_char(void *char_ptr)
{
    int char_no = (long) char_ptr;

    struct aiocb cb;
    bzero(&cb, sizeof(cb));

    struct sigaction action;
    action.sa_sigaction = read_handler;
    action.sa_flags = SA_SIGINFO;
    sigemptyset(&action.sa_mask);
    sigaction(SIG_AIO, &action, NULL);
    size_t BUF_SIZE = sizeof(char);
    //cb.aio_fildes = open("input", O_RDONLY);//fd;
    cb.aio_fildes = fd;
    cb.aio_buf = malloc(BUF_SIZE);
    cb.aio_nbytes = BUF_SIZE;
    cb.aio_offset = char_no*sizeof(char);
    //cb.aio_sigevent.sigev_notify = SIGEV_THREAD;
    cb.aio_sigevent.sigev_notify = SIGEV_SIGNAL;
    cb.aio_sigevent.sigev_signo = SIG_AIO;
    //cb.aio_sigevent.sigev_notify_function = read_handler;
    cb.aio_sigevent.sigev_notify_attributes = NULL;
    cb.aio_sigevent.sigev_value.sival_ptr = &cb;

    printf("Thread! offset:%d\n", char_no*sizeof(char));

    int ret = aio_read(&cb);

    /*
    while (aio_error(&cb) == EINPROGRESS) ;
    //cb.aio_sigevent.sigev_value.
    if ((ret = aio_return(&cb)) > 0) {
        printf("Hello world, Target char number: %d\n", char_no);
        printf("Read char: %c\n", *(char *)cb.aio_buf);
    } else {
        fprintf(stderr, "Read fail\n");
    }
    */
}

int main(void)
{
    pthread_t pth[NO_THREADS];

    fd = open("input", O_RDONLY);

    for (int i = 0; i < NO_THREADS; i++)
    {
        pthread_create(&pth[i], NULL, read_char, (void *)i);
    }

    // [NOTE] IT iS NECESSARY! So, blocking AIO is preferable!
    for (int i = 0; i < 5; i ++) sleep(1);
    pthread_exit(NULL);
    return 0;
}
